from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from genevolve_repro.marker_seed import (apply_marker_seed, calibration_loss, decision_feature,
    fit_marker, internal_fit_mask, load_tensor_payload, prepare_marker_seed, save_tensor_payload)
from genevolve_repro.route_data import MARKER_ID, recipe, register_marker


class Backbone(torch.nn.Module):
    def __init__(self, vocab=16, hidden=6):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append(input_ids.clone())
        return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(1))


class TiedPolicy(torch.nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.model = Backbone(vocab)
        self.lm_head = torch.nn.Linear(6, vocab, bias=False)
        self.lm_head.weight = self.model.embed.weight
        self.other = torch.nn.Parameter(torch.ones(3))

    def get_input_embeddings(self):
        return self.model.embed


class MarkerSeedTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.cfg = recipe()['marker_initialization']
        self.rows = [{'id': str(i), 'source_task_id': str(i // 2), 'route_label': i % 2}
                     for i in range(100)]

    def test_only_tied_row_changes_no_parent_parameter_is_unfrozen(self):
        model = TiedPolicy().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        new = torch.arange(6.)
        apply_marker_seed(model, new, 11)
        self.assertIs(model.lm_head.weight, model.get_input_embeddings().weight)
        for key, tensor in model.state_dict().items():
            wanted = before[key]
            if key in ('model.embed.weight', 'lm_head.weight'):
                wanted[11] = new
            torch.testing.assert_close(tensor, wanted, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        with self.assertRaisesRegex(ValueError, 'tied'):
            apply_marker_seed(model, new, 11)

    def test_features_see_prefix_only_and_competitor_matches_full_vocab(self):
        model = TiedPolicy().eval().requires_grad_(False)
        ids = [2, 3, 4, 11, 7, 8]
        state, other = decision_feature(model, ids, 3, 11, torch.device('cpu'))
        a, b = decision_feature(model, ids[:3] + [7, 1, 2], 3, 11, torch.device('cpu'))
        torch.testing.assert_close(state, a)
        torch.testing.assert_close(other, b)
        self.assertTrue(all(x.tolist() == [[2, 3, 4]] for x in model.model.calls))
        logits = model.lm_head(state)
        mask = torch.arange(len(logits)) != 11
        torch.testing.assert_close(other, torch.logsumexp(logits[mask], 0))
        self.assertFalse(state.requires_grad)

    def test_group_split_deterministic_fit_and_diagnostic_labels_do_not_select_weights(self):
        fit = internal_fit_mask(self.rows, self.cfg)
        self.assertGreater(fit.sum().item(), 0)
        self.assertGreater((~fit).sum().item(), 0)
        self.assertTrue(torch.equal(fit[::2], fit[1::2]))
        labels = torch.tensor([x['route_label'] for x in self.rows]).float()
        states = torch.randn(100, 6)
        states[:, 0] = labels * 2 - 1
        competitors = torch.zeros(100)
        initial = torch.zeros(6)
        result, report = fit_marker(states, competitors, labels, initial, initial, self.rows, self.cfg)
        changed = labels.clone()
        changed[~fit] = 1 - changed[~fit]
        again, other = fit_marker(states, competitors, changed, initial, initial, self.rows, self.cfg)
        torch.testing.assert_close(result, again, rtol=0, atol=0)
        self.assertLess(report['fit_final_objective'], report['fit_initial_objective'])
        self.assertFalse(report['diagnostic_used_for_selection'])
        self.assertGreater(report['calibrated_diagnostic']['auc'], .9)
        self.assertLess(other['calibrated_diagnostic']['auc'], .1)
        row = torch.randn(6, requires_grad=True)
        loss = calibration_loss(row, states, competitors, labels, initial, self.cfg)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(states @ row, labels,
                        pos_weight=torch.tensor(.5)) + .005 * (row - initial).square().sum()
        torch.testing.assert_close(loss, expected)

    def test_cache_receipt_identity_and_missing_resume_are_fail_closed(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'seed.pt'
            identity = {'fixture': 1}
            self.assertIsNone(load_tensor_payload(path, identity))
            save_tensor_payload(path, {'identity': identity, 'row': torch.ones(6)})
            torch.testing.assert_close(load_tensor_payload(path, identity)['row'], torch.ones(6))
            with self.assertRaisesRegex(ValueError, 'different parent'):
                load_tensor_payload(path, {'fixture': 2})
            path.write_bytes(b'incomplete')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                load_tensor_payload(path, identity)
            with self.assertRaisesRegex(ValueError, 'missing its original'):
                prepare_marker_seed(Path(temp) / 'resume', Path('unused'), self.rows,
                                    {'config': recipe()}, MARKER_ID, 0, 1, torch.device('cpu'), resume=True)

    def test_real_prepare_pipeline_is_cached_and_resume_never_refits(self):
        model = TiedPolicy(151936)
        original = deepcopy(model.state_dict())
        rows = [{**x, 'input_ids': [3, 4, 5 + x['route_label'], MARKER_ID, 8], 'target_start': 3}
                for x in self.rows]
        with TemporaryDirectory() as temp, patch('genevolve_repro.marker_seed.load_model', return_value=model) as load:
            out = Path(temp)
            contract = {'config': recipe(), 'parent': 'fixture-parent', 'data': 'fixture-data'}
            first = prepare_marker_seed(out, Path('unused'), rows, contract, MARKER_ID, 0, 1, torch.device('cpu'))
            load.assert_called_once()
            with patch('genevolve_repro.marker_seed.fit_marker', side_effect=AssertionError('must not refit')):
                second = prepare_marker_seed(out, Path('unused'), rows, contract, MARKER_ID, 0, 1,
                                             torch.device('cpu'), resume=True)
            load.assert_called_once()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue((out / 'marker_initialization' / 'seed_provenance.json').is_file())

    def test_registration_is_idempotent_but_wrong_id_is_rejected(self):
        class Tokenizer:
            eos_token_id = 151645
            all_special_ids = []
            marker = MARKER_ID

            def add_special_tokens(self, *args, **kwargs):
                added = int(self.marker not in self.all_special_ids)
                self.all_special_ids = [self.marker]
                return added

            def encode(self, *args, **kwargs):
                return [self.marker]

        tokenizer = Tokenizer()
        register_marker(tokenizer)
        register_marker(tokenizer)
        tokenizer.marker += 1
        with self.assertRaises(ValueError):
            register_marker(tokenizer)


if __name__ == '__main__':
    unittest.main()
