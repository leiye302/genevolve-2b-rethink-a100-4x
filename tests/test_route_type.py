from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from genevolve_repro.route_data import recipe
from genevolve_repro.route_sft import RouteScorer, summarize_losses
from genevolve_repro.route_type import (
    TYPE_OBJECTIVE, TYPE_TOKENS, action_boundary, action_type_bce,
    joint_prefix_logit, validate_type_config, validate_type_tokenizer,
)


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
    prefix = [3, 4, 2]  # A stand-in for the fully masked H/R prefix.
    action = [5, 9] if is_answer else [7]
    ids = prefix + ([MARKER] if is_bad else []) + action + list(body) + [2]
    return torch.tensor([ids]), len(prefix)


def explicit_reference(model, ids, start, is_bad, is_answer):
    logits = model.lm_head(model.model(ids).last_hidden_state[0])
    ce = F.cross_entropy(logits[start - 1:-1], ids[0, start:])
    route = F.binary_cross_entropy(logits[start - 1].softmax(-1)[MARKER],
                                   logits.new_tensor(float(is_bad)))
    a = start + int(is_bad)
    candidate = torch.cat((ids[:, :a], ids.new_tensor([[5]])), 1)
    after = model.lm_head(model.model(candidate).last_hidden_state[0, -1])
    p_tool = logits[a - 1].softmax(-1)[7]
    p_answer = logits[a - 1].softmax(-1)[5] * after.softmax(-1)[9]
    tool = F.binary_cross_entropy(p_tool, logits.new_tensor(float(not is_answer)))
    answer = F.binary_cross_entropy(p_answer, logits.new_tensor(float(is_answer)))
    return ce + 0.1 * route + 0.1 * (tool + answer) / 2, ce, route, tool, answer


class RouteTypeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_config_keeps_original_recipe_and_adds_only_type_objective(self):
        cfg = recipe()
        self.assertEqual((cfg['type_lambda'], cfg['type_objective']), (0.1, TYPE_OBJECTIVE))
        self.assertEqual((cfg['route_lambda'], cfg['reference_kl_coef']), (0.1, 0.0))
        self.assertEqual((cfg['good_per_batch'], cfg['bad_per_batch'], cfg['epochs']), (6, 2, 1))
        self.assertEqual(cfg['batches_per_epoch'] * cfg['epochs'], 297)
        self.assertEqual((cfg['lr'], cfg['loss_reduction']), (2e-6, 'sample_mean'))
        for weight in (-0.1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                validate_type_config({**cfg, 'type_lambda': weight})

    def test_native_prefixes_verified_without_retokenizing_or_adding_tokens(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'<tool_call>': [151657], '<answer': [27, 9217]}[text]
        validate_type_tokenizer(Tokenizer())
        wrong = SimpleNamespace(encode=lambda *a, **kw: [27])
        with self.assertRaises(ValueError):
            validate_type_tokenizer(wrong)
        for closing in (397, 88863):  # '>\n' and '>{"' must not enter type BCE.
            self.assertEqual(action_boundary([3, 27, 9217, closing], 1, 0, 151669), (1, True))

    def test_all_four_labels_and_marker_shift(self):
        for bad in (0, 1):
            for answer in (False, True):
                ids, start = fixture(bad, answer)
                self.assertEqual(action_boundary(ids[0], start, bad, MARKER, TOKENS),
                                 (start + bad, answer))
        for ids, start, bad in (([3, MARKER, 7], 1, 0), ([3, 7], 1, 1),
                                ([3, MARKER], 1, 1), ([3, 5, 8], 1, 0)):
            with self.assertRaises(ValueError):
                action_boundary(ids, start, bad, MARKER, TOKENS)

    def test_two_ovr_bces_are_not_a_candidate_softmax(self):
        first, second = torch.zeros(16, requires_grad=True), torch.zeros(16, requires_grad=True)
        for answer in (False, True):
            loss, stats = action_type_bce(first, second, answer, TOKENS)
            p_t, p_a = first.softmax(-1)[7], first.softmax(-1)[5] * second.softmax(-1)[9]
            expected = (F.binary_cross_entropy(p_t, first.new_tensor(float(not answer)))
                        + F.binary_cross_entropy(p_a, first.new_tensor(float(answer)))) / 2
            torch.testing.assert_close(loss, expected)
            self.assertLess((stats['p_tool'] + stats['p_answer']).item(), 0.1)
            shifted, _ = action_type_bce(first + 300, second - 100, answer, TOKENS)
            torch.testing.assert_close(loss, shifted)

    def test_prefix_logit_is_finite_with_extreme_confidence(self):
        for x, y in ((1000, 1000), (-1000, -1000), (1000, -1000), (-1000, 1000)):
            for target in (0., 1.):
                a, b = torch.tensor(float(x), requires_grad=True), torch.tensor(float(y), requires_grad=True)
                r = joint_prefix_logit(a, b)
                loss = F.binary_cross_entropy_with_logits(r, r.new_tensor(target))
                grads = torch.autograd.grad(loss, (a, b))
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(all(torch.isfinite(g).item() for g in grads))
        a, b = torch.tensor(1.2), torch.tensor(-0.8)
        torch.testing.assert_close(joint_prefix_logit(a, b).sigmoid(), a.sigmoid() * b.sigmoid())

    def test_causal_positions_complete_gradient_and_symmetric_replay(self):
        for bad in (0, 1):
            for answer in (False, True):
                with self.subTest(bad=bad, answer=answer):
                    model = TinyPolicy()
                    reference = deepcopy(model)
                    scorer = RouteScorer(model, projection_chunk=2, type_tokens=TOKENS)
                    ids, start = fixture(bad, answer)
                    loss, ce, route, _, stats = scorer(ids, start, bad, MARKER, 0.1,
                                                      type_lambda=0.1, with_type_metrics=True)
                    expected, ece, eroute, etool, eanswer = explicit_reference(
                        reference, ids, start, bad, answer,
                    )
                    for actual, wanted in ((loss, expected), (ce, ece), (route, eroute),
                                           (stats['tool_bce'], etool), (stats['answer_bce'], eanswer)):
                        torch.testing.assert_close(actual, wanted)
                    self.assertEqual(len(model.model.calls), 2)
                    replay = model.model.calls[1]
                    torch.testing.assert_close(replay[:, :-1], ids[:, :start + bad])
                    self.assertEqual(replay[0, -1].item(), TOKENS[1])
                    loss.backward()
                    expected.backward()
                    for actual, wanted in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-5, atol=2e-6)

    def test_type_probabilities_and_coefficients_ignore_action_body_length(self):
        for bad in (0, 1):
            for answer in (False, True):
                model = TinyPolicy()
                scorer = RouteScorer(model, type_tokens=TOKENS)
                short, start = fixture(bad, answer, (8,))
                long, _ = fixture(bad, answer, (12,) * 30)
                reports = []
                for ids in (short, long):
                    total, ce, route, _, stats = scorer(ids, start, bad, MARKER, 0.1,
                                                       type_lambda=0.1, with_type_metrics=True)
                    torch.testing.assert_close(total.detach(), ce + 0.1 * route + 0.1 * stats['type_bce'])
                    reports.append(stats)
                for key in reports[0]:
                    torch.testing.assert_close(reports[0][key], reports[1][key])

    def test_zero_weight_exactly_preserves_original_loss_and_one_forward(self):
        for bad in (0, 1):
            ids, start = fixture(bad, False)
            model = TinyPolicy()
            scorer = RouteScorer(model, type_tokens=TOKENS)
            loss, ce, bce, _ = scorer(ids, start, bad, MARKER, 0.1, type_lambda=0.0)
            self.assertEqual(len(model.model.calls), 1)
            torch.testing.assert_close(loss.detach(), ce + 0.1 * bce)

    def test_real_tiny_qwen_checkpointed_gradients_match_explicit_reference(self):
        from test_pipeline import tiny_policy

        model = tiny_policy()
        model.model.visual.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        reference = deepcopy(model)
        scorer = RouteScorer(model, projection_chunk=2, type_tokens=TOKENS)
        ids, start = fixture(1, False)
        total, *_ = scorer(ids, start, 1, MARKER, 0.1, type_lambda=0.1)
        expected, *_ = explicit_reference(reference, ids, start, 1, False)
        torch.testing.assert_close(total, expected)
        total.backward()
        expected.backward()
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            if actual.grad is None or wanted.grad is None:
                self.assertIs(actual.grad, wanted.grad)
            else:
                torch.testing.assert_close(actual.grad, wanted.grad, atol=4e-6, rtol=3e-4)

    def test_batch_statistics_average_once_and_handle_missing_type(self):
        row = dict(route_label=0, is_answer=False, loss=1.2, suffix_ce=1., route_bce=1.,
                   type_bce=1., tool_bce=0.4, answer_bce=1.6, p_tool=.6, p_answer=.1, p_rethink=.01)
        one = summarize_losses([row], recipe())
        many = summarize_losses([row] * 8, recipe())
        for key in ('loss', 'suffix_ce', 'route_bce', 'type_bce', 'type_weighted'):
            self.assertEqual(one[key], many[key])
        self.assertAlmostEqual(many['type_weighted'], 0.1)
        self.assertIsNone(many['target_answer_mean_probability'])


if __name__ == '__main__':
    unittest.main()
