from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from genevolve_repro.route_data import recipe
from genevolve_repro.route_sft import RouteScorer, summarize_losses
from genevolve_repro.route_type import action_boundary, validate_type_tokenizer
from genevolve_repro.route_objective import OBJECTIVE, action_tag_end, split_ce, supervision_layout, validate_objective

TOKENS = (7, 5, 9)
MARKER = 11


class RecordingBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 6)
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append(input_ids.detach().clone())
        return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(dim=1))


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RecordingBackbone()
        self.lm_head = torch.nn.Linear(6, 16)


def fixture(is_bad, is_answer, body=(8, 12)):
    prefix = [3, 4, 2]
    tag = [5, 9, 6] if is_answer else [7]
    ids = prefix + ([MARKER] if is_bad else []) + tag + list(body) + [2]
    return torch.tensor([ids]), len(prefix), len(prefix) + is_bad + len(tag)


def explicit_reference(model, ids, start, end, is_bad, weight):
    logits = model.lm_head(model.model(ids).last_hidden_state[0])
    ce = F.cross_entropy(logits[start - 1:-1], ids[0, start:], reduction="none")
    a, b = int(is_bad), end - start
    tag = ce[a:b].mean()
    content = torch.cat((ce[:a], ce[b:])).mean()
    route = F.binary_cross_entropy(logits[start - 1].softmax(-1)[MARKER], logits.new_tensor(float(is_bad)))
    return .15 * content + .5 * weight * tag + .1 * route, ce.mean(), route, content, tag


class RouteObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_recipe_changes_only_requested_sft_components(self):
        cfg = recipe()
        self.assertEqual(cfg["ce_objective"], OBJECTIVE)
        self.assertEqual((cfg["content_ce_weight"], cfg["action_tag_ce_weight"]), (.15, .5))
        self.assertNotIn("type_lambda", cfg)
        self.assertEqual((cfg['route_lambda'], cfg['reference_kl_coef']), (.1, 0.))
        self.assertEqual((cfg['good_per_batch'], cfg['bad_per_batch'], cfg['epochs']), (6, 2, 1))
        self.assertEqual((cfg['lr'], cfg['batches_per_epoch'], cfg['max_length']), (2e-6, 297, 12288))
        for weight in (-.1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                validate_objective({**cfg, 'action_tag_ce_weight': weight})
        with self.assertRaises(ValueError):
            validate_objective({**cfg, 'type_lambda': .1})

    def test_complete_native_tag_includes_indivisible_closing_token(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'<tool_call>': [151657], '<answer': [27, 9217]}[text]

            def decode(self, ids, **kwargs):
                return ''.join({151657: '<tool_call>', 27: '<', 9217: 'answer',
                                397: '>\n', 88863: '>{"', 29: '>', 99: 'bad'}[x] for x in ids)

        tokenizer = Tokenizer()
        validate_type_tokenizer(tokenizer)
        for bad in (0, 1):
            for close in (397, 88863, 29):
                ids = [3] + ([151669] if bad else []) + [27, 9217, close, 8, 2]
                self.assertEqual(action_tag_end(ids, 1, bad, 151669, tokenizer), 4 + bad)
            self.assertEqual(action_tag_end([3, 151657, 8, 2], 1, 0, 151669, tokenizer), 2)
        with self.assertRaises(ValueError):
            action_tag_end([3, 27, 9217, 99, 2], 1, 0, 151669, tokenizer)
        rows = [{'input_ids': [3, 151657, 8, 2], 'target_start': 1, 'route_label': 0}] * 3
        rows += [{'input_ids': [3, 151669, 27, 9217, 397, 8, 2], 'target_start': 1, 'route_label': 1}]
        before = deepcopy(rows)
        layout = supervision_layout(rows, tokenizer, 151669)
        self.assertEqual(rows, before)
        self.assertEqual([x['action_kind_weight'] for x in layout], [2/3] * 3 + [2.])
        self.assertAlmostEqual(sum(x['action_kind_weight'] for x in layout[:3]), layout[3]['action_kind_weight'])

    def test_all_four_labels_predictive_positions_and_complete_gradients(self):
        for bad in (0, 1):
            for answer in (False, True):
                model = TinyPolicy()
                reference = deepcopy(model)
                scorer = RouteScorer(model, projection_chunk=2, type_tokens=TOKENS)
                ids, start, end = fixture(bad, answer)
                weight = 1.9 if answer else .67
                loss, ce, route, _, stats = scorer(ids, start, bad, MARKER, .1, tag_end=end,
                                                 kind_weight=weight, with_metrics=True)
                expected, ece, eroute, econtent, etag = explicit_reference(reference, ids, start, end, bad, weight)
                for actual, wanted in ((loss, expected), (ce, ece), (route, eroute),
                                       (stats['content_ce'], econtent), (stats['action_tag_ce'], etag)):
                    torch.testing.assert_close(actual, wanted)
                self.assertEqual(len(model.model.calls), 1)
                loss.backward()
                expected.backward()
                for actual, wanted in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-5, atol=2e-6)

    def test_tag_token_coefficient_ignores_body_length_and_marker_is_not_dropped(self):
        for bad in (0, 1):
            for body_length in (1, 300):
                ce = torch.ones(bad + 3 + body_length + 1, requires_grad=True)
                loss, _ = split_ce(ce, bad, bad + 3, 1.9, .15, .5)
                loss.backward()
                torch.testing.assert_close(ce.grad[bad:bad+3], torch.full((3,), .5 * 1.9 / 3))
                self.assertGreater(ce.grad[-1].item(), 0)  # EOS
                if bad:
                    self.assertGreater(ce.grad[0].item(), 0)  # Marker keeps CE as well as route BCE.

    def test_real_tiny_qwen_checkpointed_gradients_match_reference(self):
        from test_pipeline import tiny_policy
        model = tiny_policy()
        model.model.visual.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        reference = deepcopy(model)
        scorer = RouteScorer(model, projection_chunk=2, type_tokens=TOKENS)
        ids, start, end = fixture(1, False)
        actual, *_ = scorer(ids, start, 1, MARKER, .1, tag_end=end, kind_weight=.67)
        expected, *_ = explicit_reference(reference, ids, start, end, 1, .67)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            if a.grad is None or b.grad is None:
                self.assertIs(a.grad, b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=4e-6, rtol=3e-4)

    def test_global_sample_mean_applied_once(self):
        row = dict(route_label=0, is_answer=False, loss=.6, suffix_ce=1., route_bce=1.,
                   content_ce=1., action_tag_ce=1., content_weighted=.15,
                   action_tag_weighted=.35, weighted_ce=.5, p_rethink=.01)
        one, many = (summarize_losses([row] * n, recipe()) for n in (1, 8))
        for key in ('loss', 'suffix_ce', 'route_bce', 'weighted_ce', 'route_weighted'):
            self.assertEqual(one[key], many[key])
        self.assertAlmostEqual(many['loss'], many['weighted_ce'] + many['route_weighted'])
        self.assertIsNone(many['target_answer_tag_ce'])


if __name__ == '__main__':
    unittest.main()
