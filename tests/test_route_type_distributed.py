"""Four-rank CPU/FSDP check with different Good/Bad and action types per rank."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration

from genevolve_repro.distributed import copy_actor, load_checkpoint, save_checkpoint
from genevolve_repro.route_data import MARKER_ID, recipe
from genevolve_repro.route_sft import RouteScorer, update
from test_distributed import sharded
from test_pipeline import tiny_policy


def native_tiny_policy():
    config = tiny_policy().config
    config.text_config.vocab_size = 151936
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
        rows.append({'input_ids': ids, 'target_start': 3, 'route_label': bad})
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
        model = native_tiny_policy()
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
                                     type_lambda=cfg['type_lambda'])
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
        assert saved['contract']['config']['type_lambda'] == 0.1
        assert saved['contract']['config']['type_objective'] == cfg['type_objective']
        assert saved['optimizer']['param_groups'][0]['lr'] == scheduler.get_last_lr()[0]
    finally:
        dist.destroy_process_group()


class RouteTypeDistributedTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and dist.is_gloo_available(), 'Linux Gloo is required')
    def test_fsdp_type_bce_matches_global_sample_mean(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(route_rank, args=(directory,), nprocs=4)


if __name__ == '__main__':
    unittest.main()
