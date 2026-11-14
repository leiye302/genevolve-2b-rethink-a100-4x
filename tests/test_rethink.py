import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import run
from genevolve_repro import hints, teacher
from genevolve_repro.common import ROOT, read_json, write_json
from genevolve_repro.evolve import update_actor
from genevolve_repro.evaluate import evaluate_case
from genevolve_repro.policy import Turn, Trajectory
from genevolve_repro.route_data import MARKER_ID, cold_start_lock, recipe
from genevolve_repro.route_sft import RouteScorer, verify_route_export
from genevolve_repro.sft_export import sft_manifest
from test_contract import fake_export


class ToyTokenizer:
    all_special_ids = [MARKER_ID]

    def apply_chat_template(self, messages, **kwargs):
        return list("\n".join(x["content"] for x in messages).encode())

    def encode(self, text, **kwargs):
        return list(text.encode())


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.lm_head = torch.nn.Linear(5, 13)


class ToyBackbone(torch.nn.Embedding):
    def __init__(self):
        super().__init__(13, 5)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=super().forward(input_ids).cumsum(dim=1))


class RethinkTest(unittest.TestCase):
    def fixture(self):
        messages = [{"role": "system", "content": "native instructions\n# Tools\ncallable tools"},
                    {"role": "user", "content": "Draw a gold bird."}]
        tagged = '<think>Check the evidence.</think>\n<RETHINK><tool_call>{"name":"image_search"}</tool_call>'
        normal = '<think>Use the image.</think>\n<answer>{"gen_prompt":"Sketch the bird"}</answer>'
        turn1 = Turn([1, 2], [4, MARKER_ID, 2, 3], messages=messages, response=tagged)
        turn2_messages = messages + [{"role": "assistant", "content": tagged},
                                     {"role": "user", "content": "<tool_response>\nIMG_001\n"
                                      "  local_path: /app/cache/private.png\nhttps://example.test/a?sig=example"
                                      "\n</tool_response>"}]
        turn2 = Turn([1, 3], [1, 8], messages=turn2_messages, response=normal)
        other = Turn([1], [1, 8], messages=messages, response=normal)
        result = SimpleNamespace(
            messages=turn2_messages + [{"role": "assistant", "content": normal}],
            termination="answer", gt_image="should not be shared")
        traj = Trajectory(result, [turn1, turn2])
        group = SimpleNamespace(trajectories=[traj, Trajectory(result, [other])])
        return group, traj, turn1, turn2, other

    def test_route_mean_ce_and_full_vocabulary_bce_use_the_predictive_index(self):
        torch.manual_seed(0)
        model = ToyPolicy()
        scorer = RouteScorer(model, projection_chunk=2)
        # The first trainable target is marker 7 at offset 3. Earlier IDs
        # condition the suffix but must not contribute additional CE terms.
        ids = torch.tensor([[4, 1, 9, 7, 8, 10, 2]])
        lam = recipe()["route_lambda"]
        loss, ce, bce, p = scorer(ids, 3, 1, 7, lam)
        logits = model.lm_head(model.model(ids).last_hidden_state[0])
        expected_ce = F.cross_entropy(logits[2:-1], ids[0, 3:])
        expected_p = logits[2].softmax(0)[7]
        expected = expected_ce - lam * expected_p.log()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(ce, expected_ce.detach())
        torch.testing.assert_close(bce, -expected_p.detach().log())
        torch.testing.assert_close(p, expected_p.detach())
        expected_grad = torch.autograd.grad(expected, model.lm_head.weight)[0]
        torch.testing.assert_close(torch.autograd.grad(loss, model.lm_head.weight)[0], expected_grad)

    def test_only_generated_id_enables_local_teacher_hint(self):
        group, traj, first, later, unrelated = self.fixture()
        original = [(list(t.prefix), list(t.tokens), [m.copy() for m in t.messages])
                    for t in (first, later, unrelated)]
        scored = []

        def client(settings, cfg, jobs):
            self.assertEqual(jobs, [(traj, [1])])
            # Even if the provider attempts to suggest an unselected step, it
            # has no route and cannot add the suggestion to a Teacher row.
            return [({1: "Use the image result.", 2: "INJECTION"}, 1)]

        def fake_score(_, prefix, tokens, temperature, device):
            self.assertFalse(torch.is_grad_enabled())
            scored.append((bytes(prefix).decode(), tokens))
            return torch.arange(len(tokens), dtype=torch.float32)

        cfg = read_json(ROOT / "configs" / "evolve.json")
        with patch("genevolve_repro.teacher.score_turn", side_effect=fake_score):
            stats = teacher.score_groups([group], None, ToyTokenizer(), cfg, "cpu", None, client)
        self.assertEqual((stats["turns"], stats["generated_marker"], stats["eligible"],
                          stats["api_attempts"], stats["local_used"], stats["teacher_turns"]), (3, 1, 1, 1, 1, 3))
        self.assertEqual(stats["native_boundary"], 1)
        self.assertTrue(all("[Privileged Skill Information]" in prefix for prefix, _ in scored))
        self.assertIn("Use the image result.", scored[0][0])
        self.assertTrue(all("[Local Decision Hint]" not in prefix for prefix, _ in scored[1:]))
        self.assertTrue(all("INJECTION" not in prefix for prefix, _ in scored))
        self.assertEqual([ids for _, ids in scored], [t.tokens for t in (first, later, unrelated)])
        self.assertEqual(original, [(t.prefix, t.tokens, t.messages) for t in (first, later, unrelated)])
        self.assertTrue(all(not t.teacher.requires_grad for t in (first, later, unrelated)))

    def test_no_api_for_unmarked_and_malformed_marker_is_not_a_second_gate(self):
        group, trajectory, first, _, _ = self.fixture()
        first.response = first.response.replace("<RETHINK>", "", 1) + "<RETHINK>"
        self.assertFalse(teacher.is_routed(first))

        def client(settings, cfg, jobs):
            self.assertEqual(jobs, [(trajectory, [1])])
            return [({}, 1)]

        _, stats = teacher.plan_hints([group], teacher.recipe(), None, client)
        self.assertEqual((stats["generated_marker"], stats["native_boundary"], stats["eligible"]), (1, 0, 1))
        first.tokens.remove(MARKER_ID)
        with patch("genevolve_repro.hints.request_trajectories") as api:
            incoming, stats = teacher.plan_hints([group], teacher.recipe(), None)
            api.assert_not_called()
        self.assertFalse(incoming)
        self.assertEqual(stats["generated_marker"], 0)
        self.assertEqual(stats["eligible"], 0)

    def test_missing_or_ungrounded_hint_falls_back(self):
        group, traj, first, _, _ = self.fixture()
        with patch("genevolve_repro.teacher.score_turn", return_value=torch.zeros(len(first.tokens))) as score:
            stats = teacher.score_groups([group], None, ToyTokenizer(), {"context_tokens": 65536, "temperature": 0.7},
                                         "cpu", None, lambda *_: [({}, 2)])
        self.assertEqual((stats["eligible"], stats["static_fallback"], stats["local_used"]), (1, 1, 0))
        prefix, has_local = teacher.preferred_prefix(ToyTokenizer(), first, "x" * 97, 65536, 96)
        self.assertIsNotNone(prefix)
        self.assertFalse(has_local)
        _, has_local = teacher.preferred_prefix(ToyTokenizer(), first, "Use IMG_001.", 65536, 96)
        self.assertFalse(has_local)
        _, has_local = teacher.preferred_prefix(ToyTokenizer(), traj.turns[1], "Use IMG_001.", 65536, 96)
        self.assertTrue(has_local)
        self.assertEqual(teacher.preferred_prefix(ToyTokenizer(), first, None, 3, 96), (None, False))
        self.assertEqual(score.call_count, 3)

    def test_api_request_contains_only_redacted_trajectory_and_parses_selected_steps(self):
        _, trajectory, _, _, _ = self.fixture()
        body = hints.build_payload(trajectory, [1])
        dumped = json.dumps(body)
        self.assertNotIn("private.png", dumped)
        self.assertNotIn("sig=", dumped)
        self.assertNotIn("should not be shared", dumped)
        self.assertEqual(body["selected_turns"], [1])
        self.assertEqual([m["turn"] for m in body["timeline"]], [0, 1, 1, 2])
        content = '{"hints":{"1":"Search for the wing pattern.","2":"not selected"}}'
        reply = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        self.assertEqual(hints.parse_reply(reply, [1]), {1: "Search for the wing pattern."})
        reply["choices"][0]["message"]["content"] = '{"hints":{"1":"a","1":"b"}}'
        with self.assertRaises(ValueError):
            hints.parse_reply(reply, [1])

    def test_eval_marker_telemetry_cannot_change_rendering_score(self):
        _, traj, first, _, _ = self.fixture()
        traj.result.gen_prompt = "Sketch the bird"
        traj.result.reference_images = []
        traj.result.to_dict = lambda: {}
        example = SimpleNamespace(id=1, gt=None, prompt="Draw a gold bird")
        feedback = SimpleNamespace(renderer=object(), score_image=lambda *_: {"overall": 0.64})
        with patch("genevolve_repro.evaluate.render_for_evaluation", return_value={"success": True, "render_fallback": True}):
            marked = evaluate_case(SimpleNamespace(collect=lambda _: traj), feedback, example, {}, Path("fixture.png"))
            first.tokens.remove(MARKER_ID)
            plain = evaluate_case(SimpleNamespace(collect=lambda _: traj), feedback, example, {}, Path("fixture.png"))
        self.assertEqual(marked.pop("route_telemetry"),
                         {"turns": 2, "marker_turns": 1, "native_boundary_turns": 1})
        self.assertEqual(plain.pop("route_telemetry"),
                         {"turns": 2, "marker_turns": 0, "native_boundary_turns": 0})
        self.assertEqual(marked, plain)

    def test_sdar_gradient_gate_is_detached_even_when_teacher_tensor_has_grad(self):
        student = torch.tensor([-2.0, -2.0, -2.0], requires_grad=True)
        target = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        terms, gate, gap = teacher.sdar_token_terms(student, target, 5.0)
        torch.testing.assert_close(gate, torch.sigmoid(5 * (target.detach() - student.detach())))
        torch.testing.assert_close(terms.detach(), gate * gap)
        terms.mean().backward()
        self.assertIsNone(target.grad)
        torch.testing.assert_close(student.grad, -gate / 3)

    def test_mixed_turn_update_keeps_all_grpo_tokens(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        a = Turn([1], [1, 1], old=torch.zeros(2), reference=torch.zeros(2), teacher=torch.zeros(2))
        b = Turn([1], [1, 1, 1], old=torch.zeros(3), reference=torch.zeros(3))
        group = SimpleNamespace(trajectories=[SimpleNamespace(turns=[a]), SimpleNamespace(turns=[b])],
                                outcomes=[SimpleNamespace(reward=1), SimpleNamespace(reward=0)])
        cfg = read_json(ROOT / "configs" / "evolve.json")

        def scorer(actor, prefix, tokens, temperature, device):
            return actor(torch.ones(len(tokens), 1)).flatten() * 0

        with patch("genevolve_repro.evolve.score_turn", side_effect=scorer) as score:
            _, norm, opsd = update_actor(model, optimizer, [group], cfg, torch.device("cpu"), 2,
                                        opsd=teacher.recipe())
            self.assertEqual(score.call_count, 2)
        self.assertEqual(opsd["sdar/tokens"], 2)
        self.assertEqual(opsd["sdar/gate_mean"], 0.5)
        self.assertEqual(norm, 0.0)

    def test_stage2_clamps_to_exact_complete_route_export(self):
        cfg = recipe()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "route"
            fake_export(path)
            with self.assertRaisesRegex(ValueError, "two-epoch"):
                verify_route_export(path)
            write_json(path / "completed.json", {"step": 594, "algorithm": "route-sft",
                       "contract": {"config": cfg, "manifest_sha256": cfg["manifest_sha256"],
                                    "parent_sha256": cold_start_lock()}})
            write_json(path / "export_sha256.json", sft_manifest(path))
            self.assertEqual(verify_route_export(path)["completed"]["step"], 594)
            (path / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_route_export(path)


if __name__ == "__main__":
    unittest.main()
