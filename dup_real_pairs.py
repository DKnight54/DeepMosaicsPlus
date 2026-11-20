import os, shutil

root = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(root, "datasets")

# 1. Auto-detect real-pair folder ('real_pair', 'real_pairs', or 'real')
base = None
for candidate in ("real_pair", "real_pairs", "real"):
    p = os.path.join(datasets_dir, candidate)
    if os.path.isdir(p):
        base = p
        break

if base is None:
    print("Available folders in datasets/:", os.listdir(datasets_dir))
    raise SystemExit("Could not find real_pair(s) folder — check the list above.")

# 2. Count synthetic pool folders to calculate target ratio dynamically
synth_dir = os.path.join(datasets_dir, "synthetic_pool")
n_synth = sum(
    1 for d in os.listdir(synth_dir)
    if os.path.isdir(os.path.join(synth_dir, d))
) if os.path.isdir(synth_dir) else 400

# Target: ~20% real pairs per bucket (e.g., ~100 real folders for ~411 synthetic)
TARGET_REAL_RATIO = 0.20
MAX_EXTRA_COPIES = 8  # Safety cap so tiny buckets aren't over-duplicated

target_real_folders = int(round(n_synth * TARGET_REAL_RATIO / (1.0 - TARGET_REAL_RATIO)))
print(f"Found {n_synth} synthetic folders -> targeting ~{target_real_folders} real folders ({TARGET_REAL_RATIO:.0%} share) per bucket.\n")

# 3. Dynamically balance each bucket
for bucket in sorted(os.listdir(base)):
    bdir = os.path.join(base, bucket)
    if not os.path.isdir(bdir):
        continue

    originals = sorted([
        c for c in os.listdir(bdir)
        if os.path.isdir(os.path.join(bdir, c)) and "_dup" not in c
    ])
    n_orig = len(originals)
    if n_orig == 0:
        continue

    # Calculate multiplier needed to reach target_real_folders, capped at MAX_EXTRA_COPIES
    desired_total = max(n_orig, target_real_folders)
    n_copies = min(MAX_EXTRA_COPIES, max(0, round(desired_total / n_orig) - 1))

    created = 0
    for clip in originals:
        src = os.path.join(bdir, clip)
        for k in range(1, n_copies + 1):
            dst = os.path.join(bdir, f"{clip}_dup{k}")
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
                created += 1

    final_total = len([d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d))])
    actual_pct = 100.0 * final_total / (n_synth + final_total)
    print(f"{bucket}: {n_orig} originals -> {n_copies} extra dupes each -> {final_total} total folders ({actual_pct:.1f}% real share)")