import argparse
import json
from pathlib import Path
import subprocess
import sys

from genevolve_repro.common import ROOT, WORLD_SIZE, required_env
from genevolve_repro.sft_export import export_files, sft_manifest


def distributed(subcommand, work, *arguments):
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}", str(ROOT / "run.py"),
        "--work", str(work), subcommand, *arguments,
    ], cwd=ROOT, check=True)


def evaluation_arguments(args):
    result = ["--split", args.split]
    if args.policy is not None:
        result.extend(["--policy", str(args.policy.resolve())])
    if args.max_cases is not None:
        result.extend(["--max-cases", str(args.max_cases)])
    return result


def eval_flags(parser):
    parser.add_argument("--split", choices=["bench", "rl-val", "wise"], default="bench")
    parser.add_argument("--policy", type=Path, help="HF export (default: WORK/rethink/final)")
    parser.add_argument("--max-cases", type=int, help="Limit a live evaluation for a pilot")


def execute(args, work):
    if args.command in {"grpo", "rethink"}:
        export_files(args.sft_checkpoint)
        if args.steps is not None and args.steps < 1:
            raise ValueError("--steps must be positive.")
    if args.command in {"grpo", "rethink", "eval"}:
        keys = ("WISE_API_KEY",) if getattr(args, "split", None) == "wise" else ("JUDGE_API_KEY",)
        for key in ("SERPER_API_KEY", *keys):
            required_env(key)
    if args.command == "rethink":
        from genevolve_repro.hints import settings_from_env
        from genevolve_repro.route_sft import verify_route_export

        settings_from_env()
        verify_route_export(args.sft_checkpoint.resolve())
    if args.command == "prepare":
        from genevolve_repro.data import prepare
        prepare(work)
    if args.command == "route-sft":
        from genevolve_repro.route_data import pack_manifest, recipe
        from genevolve_repro.runtime import preflight

        pack_manifest(args.data.resolve(), recipe())
        preflight("route-sft", work)
        distributed("_route-sft", work, "--parent", str(args.parent.resolve()), "--data", str(args.data.resolve()))
    if args.command in {"grpo", "rethink"}:
        from genevolve_repro.runtime import preflight

        preflight(args.command, work)
        flags = ["--sft-checkpoint", str(args.sft_checkpoint.resolve())]
        if args.steps is not None:
            flags.extend(["--steps", str(args.steps)])
        distributed("_" + args.command, work, *flags)
    if args.command == "eval":
        export_files(args.policy.resolve() if args.policy else work / "rethink" / "final")
        distributed("_eval", work, *evaluation_arguments(args))


def sft_flag(parser):
    parser.add_argument("--sft-checkpoint", required=True, type=Path,
                        help="Qwen3-VL-8B HF export; Route-SFT final for rethink, original SFT-550 for GRPO")


def route_flags(parser):
    parser.add_argument("--data", required=True, type=Path, help="sealed Route-SFT v2 pack")
    parser.add_argument("--parent", required=True, type=Path, help="matching original SFT checkpoint-550 export")


def main():
    parser = argparse.ArgumentParser(description="GenEvolve 8B Route-SFT and Actor-routed OPSD on four GPUs")
    parser.add_argument("--work", type=Path, default=ROOT / "work")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="download pinned RL and benchmark splits")
    sft_flag(commands.add_parser("check-sft", help="check and fingerprint an existing SFT export"))
    for name in ("check-route", "route-sft", "_route-sft"):
        route_flags(commands.add_parser(name))
    for name in ("grpo", "_grpo", "rethink", "_rethink"):
        stage = commands.add_parser(name)
        sft_flag(stage)
        stage.add_argument("--steps", type=int, help="RL updates (default: one pass over the train split)")
    for name in ("eval", "_eval"):
        eval_flags(commands.add_parser(name))
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "check-sft":
        try:
            print(json.dumps(sft_manifest(args.sft_checkpoint.resolve()), indent=2))
        except ValueError as error:
            parser.error(str(error))
    elif args.command == "check-route":
        from transformers import AutoTokenizer

        from genevolve_repro.route_data import (
            load_pack, pack_manifest, recipe, register_marker, verify_parent, verify_tokenizer,
        )

        cfg = recipe()
        m = pack_manifest(args.data.resolve(), cfg)
        verify_parent(args.parent.resolve(), m, cfg)
        rows, batches = load_pack(args.data.resolve(), cfg, m)
        tokenizer = register_marker(AutoTokenizer.from_pretrained(args.parent.resolve(), local_files_only=True))
        verify_tokenizer(rows, tokenizer)
        print(json.dumps({"rows": len(rows), "batches": len(batches), "updates": len(batches) * cfg["epochs"],
                          "max_tokens": max(map(lambda row: len(row["input_ids"]), rows)),
                          "all_online_prefixes_matched": True}, indent=2))
    elif args.command == "_route-sft":
        from genevolve_repro.route_sft import worker
        worker(work, args.parent.resolve(), args.data.resolve())
    elif args.command in {"_grpo", "_rethink"}:
        from genevolve_repro.evolve import worker
        worker(work, args.sft_checkpoint.resolve(), args.steps, rethink=args.command == "_rethink")
    elif args.command == "_eval":
        from genevolve_repro.evaluate import worker
        policy = args.policy.resolve() if args.policy else work / "rethink" / "final"
        worker(work, policy, args.split, args.max_cases)
    else:
        from filelock import FileLock, Timeout
        work.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(work / ".run.lock", timeout=0):
                execute(args, work)
        except Timeout as error:
            raise SystemExit(f"Another run is using {work}.") from error
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
