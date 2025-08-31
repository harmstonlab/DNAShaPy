#!/usr/bin/env python3
import argparse
import gzip
import os
import sys

import numba as nb
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from DNAShaPy.Predictor import Predictor
from DNAShaPy.utils import (
    build_cache,
    check_equal_length,
    format_vals,
    get_cache,
    is_dir_like,
    open_maybe_gz,
)


def build_arg_parser():
    """Parse CLI arguments for the DNAShaPy command.

    Returns
    -------
    argparse.Namespace
    Parsed arguments.
    """

    ap = argparse.ArgumentParser(
        description="Predict DNA shape features per FASTA record"
    )

    # Operation group: exactly one of these
    g1 = ap.add_mutually_exclusive_group(required=True)
    g1.add_argument("--input", help="FASTA file (.fa/.fasta or .gz)")
    g1.add_argument(
        "--build-cache",
        action="store_true",
        help="Build the shape cache for the requested feature(s) "
        "(requires --feature <name> or --all; no FASTA needed)",
    )
    g1.add_argument(
        "--get-cache",
        action="store_true",
        help="Download/populate the shape cache to --cache-dir and exit (no FASTA needed)",
    )
    g1.add_argument(
        "--list-features",
        action="store_true",
        help="Print all available features that can be predicted",
    )

    # Features group: NOT required at parse-time (validated after)
    g2 = ap.add_mutually_exclusive_group(required=False)
    g2.add_argument(
        "--feature",
        help="Feature name (e.g., MGW, ProT, Shift, ...) or comma-separated list",
    )
    g2.add_argument(
        "--all",
        action="store_true",
        help="Predict all available features; one output file per feature",
    )

    ap.add_argument(
        "--layer",
        default=4,
        type=int,
        help="Padding radius (bp-step: must be <=4). Default: 4",
        choices=[0, 1, 2, 3, 4, 5],
    )

    ap.add_argument(
        "--output",
        default="-",
        help="Output path. Use '{feature}' as a placeholder for --all "
        "(e.g., 'out/{feature}.txt.gz'). If a directory is given or ends with '/', "
        "files will be named '{feature}.txt'. Default: '-' (stdout; only with --feature)",
    )

    ap.add_argument(
        "--cache-dir",
        help="Directory to store/load np_array .npy arrays. Default: {install_path}/shape_cache",
    )

    ap.add_argument(
        "--bp",
        default="parquets/bp.parquet",
        help="Path to bp.parquet. Default: parquets/bp.parquet",
    )

    ap.add_argument(
        "--bpstep",
        default="parquets/bpstep.parquet",
        help="Path to bpstep.parquet. Default: parquets/bpstep.parquet",
    )

    ap.add_argument(
        "--headers", action="store_true", help="Include FASTA headers in output"
    )

    ap.add_argument(
        "--means-only",
        action="store_true",
        help="Only output the mean values per position. Requires all sequences to be of equal length.",
    )

    ap.add_argument(
        "--quiet", action="store_true", help="Suppress progress bars and messages"
    )

    return ap


def validate_args_or_exit(ap, args):
    """Validate inter-dependent CLI arguments, exiting with usage on error.

    Handles early-exit operations (``--get-cache`` / ``--list-features``) and
    enforces that either ``--feature`` or ``--all`` is supplied for prediction
    or cache-building workflows.
    """

    # If --get-cache: we’re done; no other required args.
    if args.get_cache:
        return

    if args.list_features:
        print("Intra-base (bp) features:")
        print("\n".join(sorted(Predictor().bp_features)))
        print()
        print("Inter-base (bpstep) features:")
        print("\n".join(sorted(Predictor().bpstep_features)))
        exit()

    # For run or build-cache, require one of --feature / --all
    if not (args.feature or args.all):
        ap.error("one of --feature or --all is required (unless --get-cache is used)")


def main():
    """Entry point for the CLI tool.

    The workflow is:
    1. Parse arguments and set up the ``Predictor``.
    2. Resolve the feature set (single feature or ``--all``).
    3. Optionally pre-build np_array cache arrays and exit when ``--build-cache`` is used.
    4. Stream through FASTA records computing predictions, either writing per-record
    values or aggregating means when ``--means-only`` is enabled.
    """

    ap = build_arg_parser()
    args = ap.parse_args()

    validate_args_or_exit(ap, args)
    if args.cache_dir is None:
        from importlib.resources import files

        try:
            args.cache_dir = files("DNAShaPy").joinpath("shape_cache")
        except Exception as e:
            sys.exit(f"Error: could not locate default shape_cache directory: {e}")
    if args.get_cache:
        if args.layer == 5:
            parts = ["5_part_1", "5_part_2"]
            for p in parts:
                get_cache(args.cache_dir, p)
        else:
            get_cache(args.cache_dir, args.layer)
        exit(0)
    if args.output == "-" and not args.build_cache and not args.quiet:
        print("Forcing quiet mode when outputting to stdout", file=sys.stderr)
        args.quiet = True
    pred = Predictor(args.bp, args.bpstep, cache_dir=args.cache_dir, quiet=args.quiet)
    if args.all:
        features = pred.allfeatures
    else:
        features = set()
        for f in args.feature.split(","):
            f = f.strip()
            if f not in pred.allfeatures:
                sys.exit(f"Error: feature '{f}' not found.")
            features.add(f)
    if (features & pred.bpstep_features) and args.layer >= 5:
        sys.exit("Error: for bp-step features, --layer must be <= 4.")
    if (features & pred.bp_features) and args.layer > 5:
        sys.exit("Error: for bp features, --layer must be <= 5.")
    if args.build_cache:
        built, skipped = build_cache(pred, features, args.layer)
        if not args.quiet:
            if built:
                print(
                    f"Built {built} new .npy cache files and placed in {args.cache_dir}.",
                    file=sys.stdout,
                )
            if skipped:
                print(
                    f"Skipped {skipped} existing .npy cache files in {args.cache_dir}.",
                    file=sys.stdout,
                )
        sys.exit(0)
    out_map = {}
    if len(features) > 1:
        if args.output == "-":
            sys.exit(
                "Error: Predicting multiple features requires --output to be a directory or a path template containing '{feature}'."
            )
        if "{feature}" in args.output:
            for f in features:
                path = args.output.replace("{feature}", f)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                out_map[f] = open_maybe_gz(path, "wt")
        elif is_dir_like(args.output):
            os.makedirs(args.output, exist_ok=True)
            for f in features:
                path = os.path.join(args.output, f"{f}.txt")
                out_map[f] = open_maybe_gz(path, "wt")
        else:
            sys.exit(
                "Error: --output must be a directory or include '{feature}' when predicting multiple features."
            )
    else:
        out_map[next(iter(features))] = (
            sys.stdout if args.output == "-" else open_maybe_gz(args.output, "wt")
        )

    if not args.quiet:
        print(f"Loading sequences from {args.input}", file=sys.stdout)

    in_handle = open_maybe_gz(args.input, "rt")
    records = SeqIO.parse(in_handle, "fasta")
    seqs = [(x.id, str(x.seq)) for x in records]
    if args.means_only:
        check_equal_length(seqs, args.quiet)
    try:
        # Preload all arrays/parquets
        for f in features:
            pred._np_array_table(f, pred._k_for(f, args.layer))
        if args.all:
            record_desc = "Predicting all features"
        elif len(features) > 1:
            record_desc = f"Predicting {len(features)} features"
        else:
            record_desc = f"Predicting {next(iter(features))}"

        seqs = tqdm(
            seqs,
            desc=record_desc,
            unit="seq",
            total=len(seqs),
            disable=args.quiet,
            file=sys.stdout,
        )

        sums = {}
        counts = {}
        value_arrays = {}
        for (id, seq) in seqs:
            for f in features:
                vals = pred.predict_seq(seq, f, args.layer)
                if args.means_only:
                    v = np.asarray(vals, dtype=np.float64)
                    mask = ~np.isnan(v)
                    if f not in sums:
                        sums[f] = np.zeros_like(v, dtype=np.float64)
                        counts[f] = np.zeros_like(v, dtype=np.int64)
                    sums[f][mask] += v[mask]
                    counts[f][mask] += 1
                else:
                    if f not in value_arrays:
                        value_arrays[f] = {}
                    value_arrays[f][id] = vals

        if args.means_only:
            for f in features:
                if not args.quiet:
                    print(
                        f"Writing mean values for {f} to {out_map[f].name}",
                        file=sys.stdout,
                    )
                s = sums.get(f)
                c = counts.get(f)
                if s is None:
                    continue
                means = np.full_like(s, np.nan, dtype=np.float64)
                nz = c > 0
                means[nz] = s[nz] / c[nz]
                line = " ".join("NA" if np.isnan(x) else str(x) for x in means)
                if args.headers:
                    print(">means", file=out_map[f])
                print(line, file=out_map[f])
        else:
            for f in features:
                if not args.quiet:
                    print(
                        f"Writing values for {f} to {out_map[f].name}",
                        file=sys.stdout,
                    )
                for rec_id, vals in tqdm(
                    value_arrays[f].items(),
                    desc=f"Writing {f}",
                    unit="seq",
                    disable=args.quiet,
                    file=sys.stdout,
                ):
                    if args.headers:
                        out_map[f].write(f">{rec_id}\n")
                    out_map[f].write(format_vals(vals, " ") + "\n")

    finally:
        for fh in locals().get("out_map", {}).values():
            if fh not in (sys.stdout, sys.stderr):
                fh.close()
        if "in_handle" in locals():
            in_handle.close()


if __name__ == "__main__":
    main()
