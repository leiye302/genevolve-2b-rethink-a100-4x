"""Optional anonymous public-pack/tokenizer + real tiny-Qwen gradient smoke.

Not a substitute for check-route against the actual SFT-550 parent or GPU tests.
Downloads only the ~39 MB training pack and tokenizer files, never model weights.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genevolve_repro.common import source_lock, write_json
from genevolve_repro.route_data import MARKER_ID, load_pack, recipe, register_marker, verify_tokenizer
from genevolve_repro.route_objective import supervision_layout
from genevolve_repro.route_release import prepare_route
from genevolve_repro.sft_export import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    cfg = recipe()
    folder = prepare_route(work)
    rows, batches = load_pack(folder, cfg)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    import torch
    from genevolve_repro.route_sft import RouteScorer
    from genevolve_repro.marker_seed import apply_marker_seed, decision_feature, fit_marker

    source = source_lock()['base']
    tokenizer_dir = work / 'public-tokenizer'
    snapshot_download(source['repo'], revision=source['revision'], token=False, local_dir=tokenizer_dir,
                      allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'chat_template.json',
                                      'chat_template.jinja', 'merges.txt', 'vocab.json', 'config.json'])
    tokenizer = register_marker(AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True))
    register_marker(tokenizer)  # Second registration must be harmless.
    verify_tokenizer(rows, tokenizer)
    layout = supervision_layout(rows, tokenizer, MARKER_ID)
    for row, metadata in zip(rows, layout):
        row.update(metadata)
    counts = Counter('answer' if '<answer>' in row['target'][:32] else 'tool' for row in rows)
    tag_lengths = Counter(row['action_tag_end'] - row['target_start'] - row['route_label'] for row in rows)
    print({'rows': len(rows), 'batches': len(batches), 'kinds': dict(counts),
           'actual_tag_lengths': dict(tag_lengths)}, flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model_cfg = Qwen3VLConfig(
        tie_word_embeddings=True, image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        text_config={'vocab_size': 151936, 'hidden_size': 32, 'intermediate_size': 64,
                     'num_hidden_layers': 1, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'head_dim': 8, 'max_position_embeddings': 16384, 'tie_word_embeddings': True,
                     'rope_scaling': {'rope_type': 'default', 'mrope_section': [2, 1, 1]},
                     'bos_token_id': 151643, 'eos_token_id': 151645, 'pad_token_id': 151643},
        vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                       'out_hidden_size': 32, 'deepstack_visual_indexes': [], 'num_position_embeddings': 16},
    )
    model_cfg._attn_implementation = 'sdpa'
    model = Qwen3VLForConditionalGeneration(model_cfg).eval().requires_grad_(False)

    def kind(row):
        return row['route_label'], '<answer>' in row['target'][:32]

    eligible = [batch for batch in batches if len({kind(rows[k]) for k in batch}) == 4]
    batch = min(eligible, key=lambda ix: sum(len(rows[k]['input_ids']) +
                8 * (len(rows[k]['input_ids']) - rows[k]['target_start']) for k in ix))
    selected = [rows[k] for k in batch]
    features = [decision_feature(model, r['input_ids'], r['target_start'], MARKER_ID, torch.device('cpu'))
                for r in selected]
    weight = model.lm_head.weight
    initial = .5 * weight[151657] + .25 * weight[27] + .25 * weight[9217]
    seed, fit_report = fit_marker(torch.stack([x[0] for x in features]), torch.stack([x[1] for x in features]),
                                  torch.tensor([r['route_label'] for r in selected]), initial,
                                  weight[MARKER_ID].clone(), selected, cfg['marker_initialization'])
    before_seed = weight.clone()
    apply_marker_seed(model, seed, MARKER_ID)
    keep = torch.arange(len(weight)) != MARKER_ID
    torch.testing.assert_close(weight[keep], before_seed[keep], rtol=0, atol=0)
    model.train().requires_grad_(True)
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    scorer = RouteScorer(model, projection_chunk=64)
    optimizer = torch.optim.AdamW((p for p in scorer.parameters() if p.requires_grad), lr=cfg['lr'],
                                  betas=cfg['betas'], weight_decay=cfg['weight_decay'])
    initial_weights = weight.detach().clone()
    for row in selected:
        loss, *_ = scorer(torch.tensor([row['input_ids']]), row['target_start'], row['route_label'],
                          MARKER_ID, cfg['route_lambda'], tag_end=row['action_tag_end'],
                          kind_weight=row['action_kind_weight'], content_weight=cfg['content_ce_weight'],
                          tag_weight=cfg['action_tag_ce_weight'])
        if not torch.isfinite(loss):
            raise ValueError('Non-finite smoke loss.')
        (loss / len(batch)).backward()
    norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg['clip_grad'])
    if not torch.isfinite(norm):
        raise ValueError('Non-finite smoke gradient.')
    optimizer.step()
    delta = (weight.detach() - initial_weights).abs().max().item()
    if not delta > 0:
        raise ValueError('Smoke optimizer did not update the model.')
    report = {'rows': len(rows), 'batches': len(batches), 'kinds': dict(counts),
              'tag_lengths': dict(tag_lengths), 'public_tokenizer_source': source,
              'artifact_sha256': {f: file_sha256(folder / f) for f in ('train.jsonl', 'batches.jsonl')},
              'smoke_batch': batches.index(batch), 'all_four_categories': True,
              'tiny_random_model_only': True, 'actual_sft550_or_gpu_verified': False,
              'marker_fit': fit_report, 'gradient_norm': float(norm), 'weight_max_delta': delta}
    write_json(work / 'cpu_smoke_receipt.json', report)
    print({'smoke': 'passed', 'actual_sft550_or_gpu_verified': False}, flush=True)


if __name__ == '__main__':
    main()
