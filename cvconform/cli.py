"""Command-line interface for cvconform."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cvconform",
        description="The correctness and reliability layer for computer vision AI. "
                    "Answers: is this still the same model?",
    )
    sub = p.add_subparsers(dest="command")

    # verify
    v = sub.add_parser("verify", help="Run differential conformance verification on a model")
    v.add_argument("model", help="Path to the model artifact (.onnx/.pt/.mlmodel)")
    v.add_argument("--reference", default="pytorch",
                   help="Reference runtime ('pytorch','onnx','coreml')")
    v.add_argument("--targets", default=None,
                   help="Comma-separated target runtimes (default: all non-reference)")
    v.add_argument("--dataset", default="synthetic",
                   help="'synthetic' or path to dir of .npy samples")
    v.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic)")
    v.add_argument("--samples", type=int, default=1, help="Number of input samples")
    v.add_argument("--input-shape", default=None,
                   help="Override input shape, e.g. 1,3,224,224")
    v.add_argument("--json", metavar="PATH", default=None,
                   help="Write machine-readable JSON report to PATH")
    v.add_argument("--quiet", action="store_true", help="Suppress human report on success")
    return p


def _parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        from cvconform.verify import verify

        kwargs = {
            "model": args.model,
            "reference": args.reference,
            "dataset": args.dataset,
            "seed": args.seed,
            "num_samples": args.samples,
        }
        if args.targets:
            kwargs["targets"] = [t.strip() for t in args.targets.split(",") if t.strip()]
        if args.input_shape:
            kwargs["input_shape"] = _parse_shape(args.input_shape)
        report = verify(**kwargs)

        from cvconform.report import render_human
        print(render_human(report))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report written to {args.json}")
        return 0
    else:
        print("cvconform: no command given (try 'cvconform verify <model>')", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
