import os
import sys
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import loadmodel
from util import mosaic as mosaic_util


def _measure_clip_pct(cdir):
    orig_dir = os.path.join(cdir, "origin_image")
    mos_dir = os.path.join(cdir, "mosaic_image")
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    if not names:
        return None
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    pairs = []
    for fn in names[::max(1, len(names) // 10)][:10]:
        oi = cv2.imread(os.path.join(orig_dir, fn))
        mi = cv2.imread(os.path.join(mos_dir, fn))
        if oi is None or mi is None or oi.shape != mi.shape:
            continue
        diff = (np.abs(oi.astype(np.int16) - mi.astype(np.int16)).mean(axis=2) > 12).astype(np.uint8) * 255
        mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k_close)
        pairs.append((mi, mask))
    res = mosaic_util.estimate_mosaic_block_pct_multi(pairs, min_confidence=0.12, min_pct=2.0, max_pct=12.0)
    return res["pct"] if res else None


def main():
    parser = argparse.ArgumentParser(description="Re-bucket existing real_pair clips by post-crop mosaic block %.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"))
    parser.add_argument("--manifest", default=os.path.join("pretrained_models", "mosaic", "size_manifest.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    unique_clips = []
    dup_dirs = []
    for b in sorted(os.listdir(args.real_pairs_dir)):
        bdir = os.path.join(args.real_pairs_dir, b)
        if not os.path.isdir(bdir) or not b.startswith("bucket_"):
            continue
        for c in sorted(os.listdir(bdir)):
            cdir = os.path.join(bdir, c)
            if not os.path.isdir(cdir):
                continue
            if "_dup" in c:
                dup_dirs.append(cdir)
            elif os.path.isdir(os.path.join(cdir, "origin_image")):
                unique_clips.append((b, c, cdir))

    print(f"Found {len(unique_clips)} unique clip(s) and {len(dup_dirs)} _dup* folder(s).")
    if dup_dirs and not args.dry_run:
        for d in dup_dirs:
            shutil.rmtree(d, ignore_errors=True)
        print(f"Removed {len(dup_dirs)} old _dup* folder(s) before re-bucketing.")

    moves = []
    counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_measure_clip_pct, cdir): (b, c, cdir) for b, c, cdir in unique_clips}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Measuring post-crop %", unit="clip"):
            old_b, clip, cdir = futs[fut]
            pct = fut.result()
            if pct is None:
                continue
            _, bucket = loadmodel.select_model_by_pct(pct, args.manifest)
            new_b = f"bucket_{bucket:g}"
            counts[new_b] = counts.get(new_b, 0) + 1
            moves.append((cdir, os.path.join(args.real_pairs_dir, new_b, clip), old_b, new_b))

    print("\nTarget post-crop bucket distribution:")
    for b in sorted(counts):
        print(f"  {b:<12}: {counts[b]} unique clip(s)")

    if args.dry_run:
        print("\n[DRY RUN] No folders moved.")
        return

    moved = 0
    for src, dst, old_b, new_b in moves:
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        moved += 1

    # Remove empty legacy bucket_* folders
    for b in os.listdir(args.real_pairs_dir):
        bdir = os.path.join(args.real_pairs_dir, b)
        if os.path.isdir(bdir) and b.startswith("bucket_") and not os.listdir(bdir):
            os.rmdir(bdir)

    print(f"\nDone: moved {moved} clip(s) into post-crop buckets. Re-run your balance script next if desired.")


if __name__ == "__main__":
    main()