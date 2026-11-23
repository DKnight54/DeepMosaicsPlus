import os
import re
import argparse

SYNTH_RE = re.compile(r'^\[\d+/\d+\]\s+(.+?)\s+\([\d.]+s\)\s+->\s+\+0\s+clip\(s\)\s*$')
REAL_RE = re.compile(r'^\[\d+/\d+\]\s+(.+?)\s+\([\d.]+s\)\s+->\s+0\s+clip\(s\)\s*$')


def _index_files(roots):
    """Map exact filename (and stem for real_pair matching) to full paths."""
    by_name, by_stem = {}, {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                by_name.setdefault(fn, []).append(full)
                stem = os.path.splitext(fn)[0]
                by_stem.setdefault(stem, []).append(full)
    return by_name, by_stem


def main():
    parser = argparse.ArgumentParser(description="Delete raw source videos that produced 0 clips in GUI logs.")
    parser.add_argument("--log", required=True, help="Path to pasted GUI log .txt file")
    parser.add_argument("--synthetic_src", default="", help="Raw source video folder for Step 2 (synthetic)")
    parser.add_argument("--real_censored_src", default="", help="Raw censored video folder for Step 1 (real_pair)")
    parser.add_argument("--real_uncensored_src", default="", help="Raw uncensored video folder for Step 1 (real_pair)")
    parser.add_argument("--delete", action="store_true", help="Actually delete matched files (default: dry-run)")
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        print(f"Error: log file not found: {args.log}")
        return

    synth_zero, real_zero = set(), set()
    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_syn = SYNTH_RE.match(line)
            if m_syn:
                synth_zero.add(m_syn.group(1).strip())
                continue
            m_real = REAL_RE.match(line)
            if m_real:
                real_zero.add(m_real.group(1).strip())

    syn_by_name, _ = _index_files([args.synthetic_src])
    _, real_by_stem = _index_files([args.real_censored_src, args.real_uncensored_src])

    targets = []
    for fname in sorted(synth_zero):
        targets.extend(syn_by_name.get(fname, []))
    for pair_stem in sorted(real_zero):
        targets.extend(real_by_stem.get(pair_stem, []))

    # Deduplicate while preserving order
    seen, unique_targets = set(), []
    for p in targets:
        if p not in seen:
            seen.add(p)
            unique_targets.append(p)

    mode = "LIVE DELETE" if args.delete else "DRY RUN (pass --delete to remove)"
    print(f"[{mode}] Parsed {len(synth_zero)} zero-clip synthetic name(s) and {len(real_zero)} zero-clip real-pair stem(s).")
    print(f"Matched {len(unique_targets)} existing file(s) on disk:")

    removed = 0
    for path in unique_targets:
        if not os.path.exists(path):
            print(f"  [skip missing] {path}")
            continue
        if args.delete:
            try:
                os.remove(path)
                removed += 1
                print(f"  [deleted] {path}")
            except OSError as e:
                print(f"  [warn failed] {path} ({e})")
        else:
            print(f"  [would delete] {path}")

    if args.delete:
        print(f"\nDone: safely removed {removed}/{len(unique_targets)} file(s).")


if __name__ == "__main__":
    main()