import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MIN_FRAMES = 100
BORDER_RATIO = 0.14  # Outer 14% ring around crop is un-mosaicked background


def _border_pixels(bgr):
    """Extract flattened float32 outer border ring pixels (outside central mosaic area)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    bh = max(8, int(h * BORDER_RATIO))
    bw = max(8, int(w * BORDER_RATIO))
    top = gray[:bh, :].ravel()
    bot = gray[h - bh:, :].ravel()
    left = gray[bh:h - bh, :bw].ravel()
    right = gray[bh:h - bh, w - bw:].ravel()
    return np.concatenate([top, bot, left, right])


def _mean_shift_error(orig_borders, mos_borders, shift):
    """Mean L1 border error for shift in (-1, 0, +1). +1 means orig[t+1] matches mos[t]."""
    n = min(len(orig_borders), len(mos_borders))
    if shift == 0:
        pairs = [(i, i) for i in range(n)]
    elif shift == 1:
        pairs = [(i + 1, i) for i in range(n - 1)]
    else:
        pairs = [(i, i + 1) for i in range(n - 1)]
    if not pairs:
        return 999.0
    errs = [float(np.mean(np.abs(orig_borders[oi] - mos_borders[mi]))) for oi, mi in pairs]
    return float(np.mean(errs))

def _audit_one_clip(item, apply_fix):
    bucket, clip, cdir, orig_dir, mos_dir = item
    o_names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    m_names = sorted(f for f in os.listdir(mos_dir) if f.lower().endswith(".jpg"))
    n = min(len(o_names), len(m_names))
    if n < MIN_FRAMES:
        return None

    # Sample up to 24 evenly spaced frames across the clip for fast border comparison
    step = max(1, n // 24)
    sample_idx = list(range(0, n, step))[:24]
    o_borders, m_borders = [], []
    for idx in sample_idx:
        oi = cv2.imread(os.path.join(orig_dir, o_names[idx]))
        mi = cv2.imread(os.path.join(mos_dir, m_names[idx]))
        if oi is None or mi is None or oi.shape != mi.shape:
            return None
        o_borders.append(_border_pixels(oi))
        m_borders.append(_border_pixels(mi))

    err_0 = _mean_shift_error(o_borders, m_borders, 0)
    err_p1 = _mean_shift_error(o_borders, m_borders, 1)
    err_m1 = _mean_shift_error(o_borders, m_borders, -1)

    best_shift, best_err = (1, err_p1) if err_p1 < err_m1 else (-1, err_m1)
    is_shifted = (err_0 > 7.0) and (best_err < err_0 * 0.72) and (best_err < 12.0)

    if not is_shifted:
        return ("ok", err_0, None)

    msg = (
        f"[SHIFT {best_shift:+d}f] {bucket}/{clip}: "
        f"err0={err_0:.2f} -> err({best_shift:+d})={best_err:.2f}"
    )

    if apply_fix:
        # shift == +1: drop orig[0] & mos[-1]; shift == -1: drop mos[0] & orig[-1]
        o_slice = o_names[1:n] if best_shift == 1 else o_names[:n - 1]
        m_slice = m_names[:n - 1] if best_shift == 1 else m_names[1:n]
        tmp_o = os.path.join(cdir, "_tmp_orig")
        tmp_m = os.path.join(cdir, "_tmp_mos")
        os.makedirs(tmp_o, exist_ok=True)
        os.makedirs(tmp_m, exist_ok=True)
        for out_i, (ofn, mfn) in enumerate(zip(o_slice, m_slice), start=1):
            shutil.copy2(os.path.join(orig_dir, ofn), os.path.join(tmp_o, f"{out_i:05d}.jpg"))
            shutil.copy2(os.path.join(mos_dir, mfn), os.path.join(tmp_m, f"{out_i:05d}.jpg"))
        shutil.rmtree(orig_dir, ignore_errors=True)
        shutil.rmtree(mos_dir, ignore_errors=True)
        os.replace(tmp_o, orig_dir)
        os.replace(tmp_m, mos_dir)
        msg += " -> FIXED"

    return ("shifted", err_0, msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit and optionally fix +/-1 frame alignment slips in real_pair.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"))
    parser.add_argument("--fix", action="store_true", help="Shift and renumber misaligned clips in-place")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    candidates = []
    for bucket in sorted(os.listdir(args.real_pairs_dir)):
        bdir = os.path.join(args.real_pairs_dir, bucket)
        if not os.path.isdir(bdir) or not bucket.startswith("bucket_"):
            continue
        for clip in sorted(os.listdir(bdir)):
            if "_dup" in clip:
                continue
            cdir = os.path.join(bdir, clip)
            orig_dir = os.path.join(cdir, "origin_image")
            mos_dir = os.path.join(cdir, "mosaic_image")
            if os.path.isdir(orig_dir) and os.path.isdir(mos_dir):
                candidates.append((bucket, clip, cdir, orig_dir, mos_dir))

    ok_cnt = shifted_cnt = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_audit_one_clip, item, args.fix) for item in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Auditing alignment", unit="clip"):
            res = fut.result()
            if res is None:
                continue
            status, _, msg = res
            if status == "shifted":
                shifted_cnt += 1
                tqdm.write(msg)
            else:
                ok_cnt += 1

    print(f"\nAlignment Summary: {ok_cnt} aligned clip(s), {shifted_cnt} +/-1f shifted clip(s) detected.")


if __name__ == "__main__":
    main()