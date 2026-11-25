import os
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MIN_FRAMES = 100
BORDER_RATIO = 0.15  # Outer 15% ring is un-mosaicked background
SHIFTS = (-3, -2, -1, 0, 1, 2, 3)


def _norm_border_vec(bgr, size=144):
    """Downsample to 144x144 and extract z-normalized outer border ring (immune to codec gamma/gain)."""
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = max(4, int(round(size * BORDER_RATIO)))
    ring = np.concatenate([
        gray[:b, :].ravel(),
        gray[size - b:, :].ravel(),
        gray[b:size - b, :b].ravel(),
        gray[b:size - b, size - b:].ravel(),
    ])
    std = float(np.std(ring))
    if std < 2.0:
        return None
    return (ring - float(np.mean(ring))) / std


def _save_anaglyph_qc(orig_path, mos_path, shifted_orig_path, qc_png, label):
    """Save Red=Orig / Cyan=Mos anaglyph at shift=0 vs best_shift so 1-frame ghosting is unmistakable."""
    oi = cv2.resize(cv2.imread(orig_path), (384, 384))
    mi = cv2.resize(cv2.imread(mos_path), (384, 384))
    oi_s = cv2.resize(cv2.imread(shifted_orig_path), (384, 384))

    def anaglyph(a, b):
        # BGR: Blue+Green from b (Cyan), Red from a (Red) -> zero ghosting = pure grayscale/natural color
        out = np.zeros_like(a)
        out[:, :, 0] = b[:, :, 0]
        out[:, :, 1] = b[:, :, 1]
        out[:, :, 2] = a[:, :, 2]
        return out

    p0 = anaglyph(oi, mi)
    p1 = anaglyph(oi_s, mi)
    d0 = cv2.applyColorMap(np.clip(np.abs(oi.astype(np.int16) - mi.astype(np.int16)).mean(axis=2) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    d1 = cv2.applyColorMap(np.clip(np.abs(oi_s.astype(np.int16) - mi.astype(np.int16)).mean(axis=2) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    cv2.putText(p0, "Shift 0f Anaglyph (Ghost=Slip)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p1, f"{label} Anaglyph (Aligned)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA)
    cv2.putText(d0, "Shift 0f Diff x4", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(d1, f"{label} Diff x4", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(qc_png), exist_ok=True)
    cv2.imwrite(qc_png, np.vstack([np.hstack([p0, p1]), np.hstack([d0, d1])]))


def _audit_one_clip(item, apply_fix, qc_dir):
    bucket, clip, cdir, orig_dir, mos_dir = item
    o_names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    m_names = sorted(f for f in os.listdir(mos_dir) if f.lower().endswith(".jpg"))
    n = min(len(o_names), len(m_names))
    if n < MIN_FRAMES:
        return None

    o_vecs, m_vecs = [], []
    for idx in range(n):
        oi = cv2.imread(os.path.join(orig_dir, o_names[idx]))
        mi = cv2.imread(os.path.join(mos_dir, m_names[idx]))
        if oi is None or mi is None or oi.shape != mi.shape:
            return None
        o_vecs.append(_norm_border_vec(oi))
        m_vecs.append(_norm_border_vec(mi))

    # Compute motion weights so static holds don't dilute motion-frame shift detection
    motion = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        if o_vecs[t] is not None and o_vecs[t - 1] is not None:
            motion[t] = float(np.mean(np.abs(o_vecs[t] - o_vecs[t - 1])))
    motion[0] = motion[1] if n > 1 else 1.0
    weights = np.clip(motion, 0.05, None)

    # Evaluate true frame-by-frame shift s in [-3..+3]: compare orig[t + s] vs mos[t]
    shift_scores = {}
    for s in SHIFTS:
        num, den = 0.0, 0.0
        for t in range(max(0, -s), min(n, n - s)):
            ov, mv = o_vecs[t + s], m_vecs[t]
            if ov is not None and mv is not None:
                corr = float(np.mean(ov * mv))
                w = float(weights[t])
                num += corr * w
                den += w
        shift_scores[s] = (num / den) if den > 0 else 0.0

    # Also scan 60-frame sliding windows to catch mid-clip VFR/dropped-frame drift
    win = 60
    win_best_shifts = []
    for start in range(0, n - win + 1, win // 2):
        best_ws, best_wc = 0, -1.0
        for s in SHIFTS:
            cs = []
            for t in range(max(start, -s), min(start + win, n - s)):
                if o_vecs[t + s] is not None and m_vecs[t] is not None:
                    cs.append(float(np.mean(o_vecs[t + s] * m_vecs[t])))
            if cs:
                mc = float(np.mean(cs))
                if mc > best_wc:
                    best_wc, best_ws = mc, s
        win_best_shifts.append(best_ws)

    corr_0 = shift_scores[0]
    best_shift = max(SHIFTS, key=lambda s: shift_scores[s])
    best_corr = shift_scores[best_shift]
    has_vfr_drift = len(set(win_best_shifts)) > 1 and any(s != 0 for s in win_best_shifts)

    is_global_shift = (best_shift != 0) and ((best_corr - corr_0) >= 0.035) and (best_corr >= 0.80)
    is_corrupt_pair = (best_corr < 0.78)

    if not is_global_shift and not has_vfr_drift and not is_corrupt_pair:
        return ("ok", corr_0, None)

    # Pick highest-motion frame for visual QC proof
    peak_t = int(np.clip(np.argmax(motion), abs(best_shift) + 1, n - abs(best_shift) - 2))
    if qc_dir:
        qc_png = os.path.join(qc_dir, f"{bucket}__{clip}__shift{best_shift:+d}.png")
        _save_anaglyph_qc(
            os.path.join(orig_dir, o_names[peak_t]),
            os.path.join(mos_dir, m_names[peak_t]),
            os.path.join(orig_dir, o_names[peak_t + best_shift]),
            qc_png,
            f"Shift {best_shift:+d}f",
        )

    if is_corrupt_pair:
        return ("corrupt", corr_0, f"[LOW_CORR] {bucket}/{clip}: best_corr={best_corr:.3f} (shift={best_shift:+d}f)")

    if not is_global_shift and has_vfr_drift:
        return ("drift", corr_0, f"[VFR_DRIFT] {bucket}/{clip}: corr0={corr_0:.3f}, window_shifts={win_best_shifts}")

    msg = f"[SHIFT {best_shift:+d}f] {bucket}/{clip}: corr0={corr_0:.3f} -> corr({best_shift:+d})={best_corr:.3f}"
    if apply_fix:
        # best_shift > 0 means orig[t + s] matches mos[t] -> drop first s frames of orig
        # best_shift < 0 means orig[t - |s|] matches mos[t] -> drop first |s| frames of mos
        s = best_shift
        o_slice = o_names[s:n] if s > 0 else o_names[:n + s]
        m_slice = m_names[:n - s] if s > 0 else m_names[-s:n]
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

    return ("shifted", corr_0, msg)

def main():
    parser = argparse.ArgumentParser(description="Dense frame-exact +/-3f alignment & VFR drift auditor for real_pair.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"))
    parser.add_argument("--qc_dir", default="", help="Optional folder to save Red/Cyan anaglyph visual proof PNGs")
    parser.add_argument("--fix", action="store_true", help="Shift and renumber globally-shifted clips in-place")
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

    counts = {"ok": 0, "shifted": 0, "drift": 0, "corrupt": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_audit_one_clip, item, args.fix, args.qc_dir) for item in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Auditing alignment", unit="clip"):
            res = fut.result()
            if res is None:
                continue
            status, _, msg = res
            counts[status] += 1
            if msg:
                tqdm.write(msg)

    print(
        f"\nAlignment Summary: {counts['ok']} aligned, "
        f"{counts['shifted']} global shift (+/-1..3f), "
        f"{counts['drift']} mid-clip VFR drift, "
        f"{counts['corrupt']} low-corr/corrupt."
    )
    if args.qc_dir and (counts["shifted"] + counts["drift"] + counts["corrupt"]) > 0:
        print(f"Visual Red/Cyan anaglyph QC strips saved -> {args.qc_dir}")


if __name__ == "__main__":
    main()