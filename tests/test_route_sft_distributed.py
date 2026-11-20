"""Four-rank CPU/FSDP check with different Good/Bad and action types per rank."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration

from genevolve_repro.distributed import copy_actor, load_checkpoint, save_checkpoint
from genevolve_repro.route_data import MARKER_ID, recipe
from genevolve_repro.route_sft import RouteScorer, update
from genevolve_repro.marker_seed import apply_marker_seed, internal_fit_mask, prepare_marker_seed
from test_distributed import sharded
from test_pipeline import tiny_policy


def native_tiny_policy():
    config = tiny_policy().config
    config.text_config.vocab_size = 151936
    config.tie_word_embeddings = True
    config.text_config.tie_word_embeddings = True
    model = Qwen3VLForConditionalGeneration(config)
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def fixtures():
    rows = []
    # Six Good + two Bad, but opposing action types in simultaneous rank calls.
    for i, (bad, answer) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1),
                                      (0, 1), (0, 0), (0, 1), (0, 0))):
        ids = [3, 4, 5] + ([MARKER_ID] if bad else [])
        ids += ([27, 9217, 397] if answer else [151657]) + [8] * (i % 3 + 1) + [2]
        rows.append({'id': str(i), 'source_task_id': 'fixture-fit',
                     'input_ids': ids, 'target_start': 3, 'route_label': bad,
                     'action_tag_end': 3 + bad + (3 if answer else 1), 'action_kind_weight': 1.})
    return rows


def route_rank(rank, directory):
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root / 'rendezvous').as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(37)
        device = torch.device('cpu')
        cfg, rows = recipe(), fixtures()
        group = 0
        while not internal_fit_mask([{'id': 'fixture', 'source_task_id': str(group)}], cfg['marker_initialization'])[0]:
            group += 1
        for row in rows:
            row['source_task_id'] = str(group)
        model = native_tiny_policy()
        with patch('genevolve_repro.marker_seed.load_model', return_value=deepcopy(model)):
            seed = prepare_marker_seed(root, Path('unused'), rows, {'config': cfg}, MARKER_ID,
                                       rank, 4, device)
        apply_marker_seed(model, seed, MARKER_ID)
        reference = RouteScorer(deepcopy(model), projection_chunk=128)
        actor = sharded(RouteScorer(model), device)
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)

        for step in range(2):
            stats, norm = update(actor, optimizer, rows, list(range(8)), cfg, rank, 4, device)
            assert len(stats) == 2 and norm > 0
            ref_optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
            ref_optimizer.zero_grad(set_to_none=True)
            for row in rows:
                loss, *_ = reference(torch.tensor([row['input_ids']]), row['target_start'],
                                     row['route_label'], MARKER_ID, cfg['route_lambda'],
                                     tag_end=row['action_tag_end'], kind_weight=row['action_kind_weight'],
                                     content_weight=cfg['content_ce_weight'], tag_weight=cfg['action_tag_ce_weight'])
                (loss / 8).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg['clip_grad'])
            ref_optimizer.step()
            replica = native_tiny_policy()
            copy_actor(actor, replica)
            for actual, expected in zip(replica.parameters(), reference.model.parameters()):
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-4)
            scheduler.step()
        # Verify new objective identity and optimizer state survive checkpointing.
        save_checkpoint(root, actor, optimizer, scheduler, 2, {'config': cfg}, rank, device, verify=True)
        saved = load_checkpoint(root / 'checkpoint-00002')
        assert saved['contract']['config']['action_tag_ce_weight'] == 0.5
        assert saved['contract']['config']['ce_objective'] == cfg['ce_objective']
        assert saved['optimizer']['param_groups'][0]['lr'] == scheduler.get_last_lr()[0]
    finally:
        dist.destroy_process_group()


class RouteSftDistributedTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and dist.is_gloo_available(), 'Linux Gloo is required')
    def test_fsdp_split_ce_matches_global_sample_mean(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(route_rank, args=(directory,), nprocs=4)


if __name__ == '__main__':
    unittest.main()
