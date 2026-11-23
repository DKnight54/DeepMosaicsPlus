import os
import sys
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import mosaic as mosaic_util


def _build_mask(oi, mi, k_close):
    diff = (np.abs(oi.astype(np.int16) - mi.astype(np.int16)).mean(axis=2) > 12).astype(np.uint8) * 255
    return cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k_close)


def _measure_clip_summary(item):
    bucket, clip, cdir = item
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
        pairs.append((mi, _build_mask(oi, mi, k_close)))
    res = mosaic_util.estimate_mosaic_block_pct_multi(pairs, min_confidence=0.12, min_pct=2.0, max_pct=12.0)
    if not res:
        return None
    return (bucket, clip, res["pct"], res["confidence"], res["n_samples"])


def _scan_full_scene(item):
    """Dense frame-by-frame scan across every frame of a single clip to measure intra-scene min/max delta."""
    bucket, clip, cdir = item
    orig_dir = os.path.join(cdir, "origin_image")
    mos_dir = os.path.join(cdir, "mosaic_image")
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    frame_pcts = []
    for fn in names:
        oi = cv2.imread(os.path.join(orig_dir, fn))
        mi = cv2.imread(os.path.join(mos_dir, fn))
        if oi is None or mi is None or oi.shape != mi.shape:
            continue
        est = mosaic_util.estimate_mosaic_block_pct(
            mi, _build_mask(oi, mi, k_close), min_pct=2.0, max_pct=12.0
        )
        if est is not None and est["confidence"] >= 0.15:
            frame_pcts.append(est["pct"])
    if len(frame_pcts) < 15:
        return None
    arr = np.array(frame_pcts, dtype=np.float32)
    p_min, p_med, p_max = float(np.min(arr)), float(np.median(arr)), float(np.max(arr))
    p10, p90 = float(np.percentile(arr, 10)), float(np.percentile(arr, 90))
    return {
        "bucket": bucket, "clip": clip, "n_valid": len(arr), "n_total": len(names),
        "min": p_min, "p10": p10, "med": p_med, "p90": p90, "max": p_max,
        "delta": p_max - p_min, "delta_10_90": p90 - p10,
        "ratio": p_max / max(0.1, p_min), "ratio_10_90": p90 / max(0.1, p10),
    }

def main():
    parser = argparse.ArgumentParser(description="Audit post-crop mosaic block % and intra-scene min/max deltas.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"))
    parser.add_argument("--out_txt", default=os.path.join("tmp", "postcrop_pct_report.txt"))
    parser.add_argument("--scene_samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clips = []
    for b in sorted(os.listdir(args.real_pairs_dir)):
        bdir = os.path.join(args.real_pairs_dir, b)
        if not os.path.isdir(bdir) or not b.startswith("bucket_"):
            continue
        for c in sorted(os.listdir(bdir)):
            if "_dup" in c:
                continue
            cdir = os.path.join(bdir, c)
            if os.path.isdir(os.path.join(cdir, "origin_image")):
                clips.append((b, c, cdir))

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_measure_clip_summary, item) for item in clips]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Sampling all unique clips", unit="clip"):
            res = fut.result()
            if res is not None:
                summaries.append(res)

    random.seed(args.seed)
    chosen = random.sample(clips, min(args.scene_samples, len(clips)))
    dense_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_scan_full_scene, item) for item in chosen]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Full scan ({len(chosen)} scenes)", unit="scene"):
            res = fut.result()
            if res is not None:
                dense_rows.append(res)

    dense_rows.sort(key=lambda r: (r["bucket"], r["clip"]))
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)

    all_pcts = np.array([r[2] for r in summaries], dtype=np.float32)
    lines = [
        f"Post-Crop Mosaic Block % Audit ({len(summaries)}/{len(clips)} unique clips; _dup* excluded)",
        "=" * 86,
        "\n1. OVERALL POST-CROP BLOCK % PERCENTILES (for size_manifest.json bucket design):",
    ]
    for p in (5, 10, 20, 25, 33, 40, 50, 60, 67, 75, 80, 90, 95):
        lines.append(f"  p{p:02d}: {np.percentile(all_pcts, p):6.2f}%")

    lines.append("\n2. CURRENT PRE-CROP BUCKET vs. ACTUAL POST-CROP BLOCK % (Median & p25-p75):")
    for b in sorted({r[0] for r in summaries}):
        sub = np.array([r[2] for r in summaries if r[0] == b], dtype=np.float32)
        lines.append(
            f"  {b:<10} (n={len(sub):3d}): median={np.median(sub):5.2f}%  "
            f"(p10={np.percentile(sub,10):5.2f}%, p25={np.percentile(sub,25):5.2f}%, "
            f"p75={np.percentile(sub,75):5.2f}%, p90={np.percentile(sub,90):5.2f}%)"
        )

    if dense_rows:
        d_full = np.array([r["delta"] for r in dense_rows])
        d_1090 = np.array([r["delta_10_90"] for r in dense_rows])
        r_full = np.array([r["ratio"] for r in dense_rows])
        r_1090 = np.array([r["ratio_10_90"] for r in dense_rows])
        lines.extend([
            f"\n3. INTRA-SCENE BLOCK SIZE VARIATION ({len(dense_rows)} densely scanned scenes, every frame):",
            f"  Full spread (max - min)     : median={np.median(d_full):.2f}%, p25-p75={np.percentile(d_full,25):.2f}%-{np.percentile(d_full,75):.2f}%, p90={np.percentile(d_full,90):.2f}%",
            f"  Robust spread (p90 - p10)   : median={np.median(d_1090):.2f}%, p25-p75={np.percentile(d_1090,25):.2f}%-{np.percentile(d_1090,75):.2f}%, p90={np.percentile(d_1090,90):.2f}%",
            f"  Full stretch (max / min)    : median={np.median(r_full):.2f}x, p25-p75={np.percentile(r_full,25):.2f}x-{np.percentile(r_full,75):.2f}x, p90={np.percentile(r_full,90):.2f}x",
            f"  Robust stretch (p90 / p10)  : median={np.median(r_1090):.2f}x, p25-p75={np.percentile(r_1090,25):.2f}x-{np.percentile(r_1090,75):.2f}x, p90={np.percentile(r_1090,90):.2f}x",
            "\n4. PER-SCENE DENSE SCAN TABLE:",
            f"  {'Bucket':<9} {'Clip':<36} {'Valid':>7} {'Min%':>6} {'p10%':>6} {'Med%':>6} {'p90%':>6} {'Max%':>6} {'Delta':>6} {'Max/Min':>7}",
        ])
        for r in dense_rows:
            cname = r["clip"] if len(r["clip"]) <= 35 else r["clip"][:32] + "..."
            lines.append(
                f"  {r['bucket']:<9} {cname:<36} {r['n_valid']:>3}/{r['n_total']:<3} "
                f"{r['min']:6.2f} {r['p10']:6.2f} {r['med']:6.2f} {r['p90']:6.2f} {r['max']:6.2f} "
                f"{r['delta']:6.2f} {r['ratio']:6.2f}x"
            )

    report = "\n".join(lines) + "\n"
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved report -> {args.out_txt}")


if __name__ == "__main__":
    main()