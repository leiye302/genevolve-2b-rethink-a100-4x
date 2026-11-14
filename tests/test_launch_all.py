from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

import run


class LaunchAllTest(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        root = Path(directory).resolve()
        self.work = root / "work"
        self.parent = root / "checkpoint-550"
        self.pack = root / "route-sft-v3"
        self.final = self.work / "route_sft" / "hf-step-00594"
        self.args = Namespace(command="all", parent=self.parent, data=self.pack, steps=30, eval=False)
        stack = self.enterContext(ExitStack())
        stack.enter_context(redirect_stdout(StringIO()))
        self.sequence = Mock()
        for name, target in {
            "upstream": "run.load_upstream",
            "services": "run.check_rethink_services",
            "env": "run.required_env",
            "export": "run.export_files",
            "launch": "run.distributed",
            "verify_data": "genevolve_repro.data.verify_data",
            "prepare": "genevolve_repro.data.prepare",
            "prepare_embedding": "genevolve_repro.data.prepare_embedding",
            "embedding_ready": "genevolve_repro.data.embedding_ready",
            "manifest": "genevolve_repro.route_data.pack_manifest",
            "parent": "genevolve_repro.route_data.verify_parent",
            "identity": "genevolve_repro.route_data.route_identity",
            "hardware": "genevolve_repro.runtime.preflight",
            "handoff": "genevolve_repro.route_sft.verify_route_export",
        }.items():
            self.sequence.attach_mock(stack.enter_context(patch(target)), name)
        self.sequence.embedding_ready.return_value = False

    def test_download_then_route_sft_then_verify_then_stage2_then_eval(self):
        self.sequence.verify_data.side_effect = FileNotFoundError()
        self.args.eval = True
        run.execute(self.args, self.work)
        self.sequence.assert_has_calls([
            call.verify_data(self.work), call.prepare(self.work),
            call.launch("_route-sft", self.work, "--parent", str(self.parent), "--data", str(self.pack)),
        ])
        self.sequence.assert_has_calls([
            call.handoff(self.final, expected_contract=self.sequence.identity.return_value),
            call.embedding_ready(self.work), call.prepare_embedding(self.work),
            call.hardware("rethink", self.work),
            call.launch("_rethink", self.work, "--sft-checkpoint", str(self.final), "--steps", "30"),
        ])
        self.assertEqual(self.sequence.launch.call_args_list[-1],
                         call("_eval", self.work, "--split", "bench", "--policy",
                              str(self.work / "rethink" / "final")))

    def test_resume_reuses_ready_stage1_and_pinned_data(self):
        self.final.mkdir(parents=True)
        for name in ("completed.json", "export_sha256.json"):
            (self.final / name).touch()
        self.sequence.embedding_ready.return_value = True
        run.execute(self.args, self.work)
        self.sequence.prepare.assert_not_called()
        self.sequence.prepare_embedding.assert_not_called()
        self.sequence.handoff.assert_called_once_with(
            self.final, expected_contract=self.sequence.identity.return_value,
        )
        self.sequence.launch.assert_called_once_with(
            "_rethink", self.work, "--sft-checkpoint", str(self.final), "--steps", "30",
        )

    def test_stage1_exit_failure_never_starts_stage2(self):
        self.sequence.launch.side_effect = CalledProcessError(1, "fixture")
        with self.assertRaises(CalledProcessError):
            run.execute(self.args, self.work)
        self.sequence.launch.assert_called_once()
        self.sequence.handoff.assert_not_called()
        self.sequence.prepare_embedding.assert_not_called()

    def test_bad_route_export_blocks_stage2(self):
        self.sequence.handoff.side_effect = ValueError("fixture export mismatch")
        with self.assertRaisesRegex(ValueError, "fixture export mismatch"):
            run.execute(self.args, self.work)
        self.assertEqual([record.args[0] for record in self.sequence.launch.call_args_list], ["_route-sft"])

    def test_invalid_steps_fail_before_any_stage(self):
        self.args.steps = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            run.execute(self.args, self.work)
        self.sequence.verify_data.assert_not_called()
        self.sequence.launch.assert_not_called()

    def test_wrong_parent_stops_before_downloading_data_or_launching(self):
        self.sequence.parent.side_effect = ValueError("not a 2B SFT export")
        with self.assertRaisesRegex(ValueError, "not a 2B SFT export"):
            run.execute(self.args, self.work)
        self.sequence.verify_data.assert_not_called()
        self.sequence.launch.assert_not_called()
