import argparse
import glob
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarize and sanity-check prefilter_index.json")
    ap.add_argument("--index", default="/data/rbg/users/seanmurphy/dna_bind/runs_cache/prefilter_index.json",
                    help="Path to prefilter_index.json (default: runs_cache/prefilter_index.json)")
    ap.add_argument("--check-pred-glob", action="store_true",
                    help="Re-glob pred_glob from the index and compare counts")
    ap.add_argument("--require-some-valid", action="store_true",
                    help="Exit nonzero if no valid dirs are present")
    ap.add_argument("--show-invalid", type=int, default=5,
                    help="Show up to N invalid directories (default: 5)")
    args = ap.parse_args(argv)

    index_path = os.path.abspath(args.index)
    if not os.path.exists(index_path):
        print(f"index_not_found: {index_path}")
        return 2

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception as e:
        print(f"index_read_error: {e!r}")
        return 2

    if not isinstance(idx, dict) or "dirs" not in idx or "meta" not in idx:
        print("index_format_error: missing meta/dirs")
        return 2

    meta = idx.get("meta", {})
    dirs = idx.get("dirs", {})
    n_entries = len(dirs)
    valid = [d for d, info in dirs.items() if isinstance(info, dict) and bool(info.get("valid", False))]
    invalid = [d for d in dirs if d not in valid]

    print(f"index_path={index_path}")
    print(f"pred_glob={meta.get('pred_glob', '')}")
    print(f"dir_count_meta={meta.get('dir_count', n_entries)} entries={n_entries}")
    print(f"valid={len(valid)} invalid={len(invalid)} frac_valid={(len(valid) / n_entries) if n_entries else 0.0:.3f}")

    if args.check_pred_glob:
        pg = meta.get("pred_glob", "")
        if pg:
            try:
                matched = sorted(glob.glob(os.path.join(pg)))
                print(f"pred_glob_matched={len(matched)}")
                if int(meta.get("dir_count", -1)) != len(matched):
                    print("warn: meta.dir_count differs from current glob matches")
            except Exception as e:
                print(f"pred_glob_check_error: {e!r}")

    if invalid and args.show_invalid > 0:
        print("invalid_examples:")
        for d in invalid[: int(args.show_invalid)]:
            info = dirs.get(d, {})
            files = info.get("files", {}) if isinstance(info, dict) else {}
            print(f"  {d}")
            if isinstance(files, dict):
                print(f"    files: pt={files.get('pt')} cif={files.get('cif')} pae={files.get('pae')} pde={files.get('pde')}")

    if args.require_some_valid and len(valid) == 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


