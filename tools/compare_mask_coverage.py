import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _mask_stats(binary_mask):
    """Return (area_pct, bbox_w_pct, bbox_h_pct, edge_touch_pct) for a binary mask."""
    h, w = binary_mask.shape[:2]
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    area_pct = 100.0 * len(xs) / float(h * w)
    bw_pct = 100.0 * (xs.max() - xs.min() + 1) / float(w)
    bh_pct = 100.0 * (ys.max() - ys.min() + 1) / float(h)
    margin = max(4, int(0.03 * min(h, w)))
    touches_edge = (
        xs.min() <= margin or xs.max() >= (w - 1 - margin)
        or ys.min() <= margin or ys.max() >= (h - 1 - margin)
    )
    return (area_pct, bw_pct, bh_pct, 100.0 if touches_edge else 0.0)


def _sample_synthetic_clip(mask_dir, max_samples=12):
    names = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith('.png'))
    if not names:
        return []
    step = max(1, len(names) // max_samples)
    out = []
    for fn in names[::step][:max_samples]:
        m = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            st = _mask_stats(m > 127)
            if st is not None:
                out.append(st)
    return out


def _sample_real_clip(cdir, max_samples=12):
    orig_dir = os.path.join(cdir, 'origin_image')
    mos_dir = os.path.join(cdir, 'mosaic_image')
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.jpg'))
    if not names:
        return []
    step = max(1, len(names) // max_samples)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    out = []
    for fn in names[::step][:max_samples]:
        oi = cv2.imread(os.path.join(orig_dir, fn), cv2.IMREAD_GRAYSCALE)
        mi = cv2.imread(os.path.join(mos_dir, fn), cv2.IMREAD_GRAYSCALE)
        if oi is None or mi is None or oi.shape != mi.shape:
            continue
        diff = (np.abs(oi.astype(np.int16) - mi.astype(np.int16)) > 14).astype(np.uint8)
        closed = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k_close)
        st = _mask_stats(closed)
        if st is not None:
            out.append(st)
    return out

def _summarize(label, rows):
    arr = np.array(rows, dtype=np.float32)
    area, bw, bh, edge = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    print(f"\n=== {label} ({len(rows):,} sampled frames) ===")
    print(f"  Mask area % of crop : median {np.median(area):.1f}%  (p25–p75: {np.percentile(area, 25):.1f}% – {np.percentile(area, 75):.1f}%)")
    print(f"  BBox width % of crop: median {np.median(bw):.1f}%  (p25–p75: {np.percentile(bw, 25):.1f}% – {np.percentile(bw, 75):.1f}%)")
    print(f"  BBox height % crop  : median {np.median(bh):.1f}%  (p25–p75: {np.percentile(bh, 25):.1f}% – {np.percentile(bh, 75):.1f}%)")
    print(f"  Touches crop border : {np.mean(edge):.1f}% of frames")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare mask coverage between synthetic_pool and real_pair.")
    parser.add_argument("--synthetic_dir", default=os.path.join("datasets", "synthetic_pool"))
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    synth_dirs = [
        os.path.join(args.synthetic_dir, d, "mask")
        for d in os.listdir(args.synthetic_dir)
        if os.path.isdir(os.path.join(args.synthetic_dir, d, "mask"))
    ] if os.path.isdir(args.synthetic_dir) else []

    real_dirs = []
    if os.path.isdir(args.real_pairs_dir):
        for b in os.listdir(args.real_pairs_dir):
            bdir = os.path.join(args.real_pairs_dir, b)
            if os.path.isdir(bdir) and b.startswith("bucket_"):
                for c in os.listdir(bdir):
                    cdir = os.path.join(bdir, c)
                    if os.path.isdir(os.path.join(cdir, "origin_image")):
                        real_dirs.append(cdir)

    synth_rows, real_rows = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs_s = [pool.submit(_sample_synthetic_clip, d) for d in synth_dirs]
        for fut in tqdm(as_completed(futs_s), total=len(futs_s), desc="Sampling synthetic_pool", unit="clip"):
            synth_rows.extend(fut.result())

        futs_r = [pool.submit(_sample_real_clip, d) for d in real_dirs]
        for fut in tqdm(as_completed(futs_r), total=len(futs_r), desc="Sampling real_pair", unit="clip"):
            real_rows.extend(fut.result())

    if synth_rows:
        _summarize("synthetic_pool (YOLO ROI masks)", synth_rows)
    if real_rows:
        _summarize("real_pair (actual censored mosaics)", real_rows)


if __name__ == "__main__":
    main()