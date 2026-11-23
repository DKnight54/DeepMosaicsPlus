import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MIN_FRAMES = 100          # Matches train.py --M 100 minimum clip length
BOTTOM_BAND_RATIO = 0.65  # Check bottom 35% of the crop where subtitles enter

def _has_horizontal_text_bar(binary_mask, gray_band):
    """Return True if binary_mask contains a compact horizontal text-line bar with sharp character strokes."""
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_line)
    gx = np.abs(cv2.Sobel(gray_band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    for lbl in range(1, num_labels):
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        if bw >= 60 and (8 <= bh <= 55) and (bw / max(1, bh) >= 3.0) and area >= 350:
            roi_mask = binary_mask[y:y + bh, x:x + bw] > 0
            roi_gx = gx[y:y + bh, x:x + bw]
            # Text glyphs have dense vertical stroke transitions (high horizontal Sobel gradient)
            if float(np.mean(roi_gx[roi_mask])) > 65.0:
                return True
    return False

def has_subtitle_in_pair(orig_bgr, mos_bgr):
    """Detect horizontal burned-in subtitle lines in origin_image absent in mosaic_image."""
    h, _ = orig_bgr.shape[:2]
    y0 = int(h * BOTTOM_BAND_RATIO)

    orig_gray = cv2.cvtColor(orig_bgr[y0:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mos_gray = cv2.cvtColor(mos_bgr[y0:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)

    if float(np.mean(np.abs(orig_gray - mos_gray) < 12.0)) < 0.78:
        return False

    bright_diff = (orig_gray > 225.0) & ((orig_gray - mos_gray) > 85.0)
    return _has_horizontal_text_bar(bright_diff, orig_gray)


def detect_single_image_subtitles(bgr_frames, min_persist=5):
    """
    Detect burned-in subtitles in standalone frames (no mosaic reference).
    Looks for high-contrast bright text strokes surrounded by dark outline/shadow
    that persist across >= min_persist consecutive frames.
    """
    raw_flags = []
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))

    for bgr in bgr_frames:
        if bgr is None:
            raw_flags.append(True)
            continue
        h, _ = bgr.shape[:2]
        y0 = int(h * BOTTOM_BAND_RATIO)
        band = bgr[y0:, :]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        bright_core = (gray > 220) & (hsv[:, :, 1] < 95)

        dilated = cv2.dilate(bright_core.astype(np.uint8), k_dilate, iterations=1) > 0
        dark_outline = dilated & (~bright_core) & (gray < 75)

        # Require dark drop-shadow/outline around the bright glyphs AND a horizontal text-line bar
        is_text = (float(np.mean(dark_outline)) > 0.003) and _has_horizontal_text_bar(bright_core, gray)
        raw_flags.append(is_text)

    n = len(raw_flags)
    smoothed = [False] * n
    i = 0
    while i < n:
        if raw_flags[i]:
            j = i
            while j < n and raw_flags[j]:
                j += 1
            if (j - i) >= min_persist:
                s_pad = max(0, i - 2)
                e_pad = min(n, j + 2)
                for k in range(s_pad, e_pad):
                    smoothed[k] = True
            i = j
        else:
            i += 1
    return smoothed


def split_clean_runs(flags, min_len):
    """Return list of (start_idx, end_idx_exclusive) contiguous False runs >= min_len."""
    runs = []
    start = None
    for i, bad in enumerate(flags):
        if not bad and start is None:
            start = i
        elif bad and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(flags) - start >= min_len:
        runs.append((start, len(flags)))
    return runs

def crop_overlaps_boxes(crop_box, boxes):
    """Return True if (cx0, cy0, cx1, cy1) intersects any (bx0, by0, bx1, by1)."""
    cx0, cy0, cx1, cy1 = crop_box
    for bx0, by0, bx1, by1 in boxes:
        if cx0 < bx1 and cx1 > bx0 and cy0 < by1 and cy1 > by0:
            return True
    return False


def find_fullframe_overlay_boxes(orig_bgr_list, mos_bgr_list=None, min_persist=5):
    """Detect full-frame subtitle boxes (bottom 35%) and static corner watermark boxes per frame."""
    n = len(orig_bgr_list)
    if n == 0:
        return []
    h, w = orig_bgr_list.shape[:2]
    y_band = int(h * BOTTOM_BAND_RATIO)
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))

    # 1. Per-frame subtitle bounding boxes in full-frame coordinates
    raw_sub_boxes = [[] for _ in range(n)]
    for idx, bgr in enumerate(orig_bgr_list):
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr[y_band:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if mos_bgr_list is not None and idx < len(mos_bgr_list) and mos_bgr_list[idx] is not None:
            m_gray = cv2.cvtColor(mos_bgr_list[idx][y_band:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if float(np.mean(np.abs(gray - m_gray) < 12.0)) < 0.78:
                continue
            cand = (gray > 225.0) & ((gray - m_gray) > 85.0)
        else:
            hsv = cv2.cvtColor(bgr[y_band:, :], cv2.COLOR_BGR2HSV)
            bright = (gray > 220) & (hsv[:, :, 1] < 95)
            outline = (cv2.dilate(bright.astype(np.uint8), k_dil) > 0) & (~bright) & (gray < 75)
            if float(np.mean(outline)) <= 0.003:
                continue
            cand = bright

        merged = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, k_line)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(merged)
        for lbl in range(1, num_lbl):
            x, y, bw, bh, area = stats[lbl]
            if bw >= 60 and (8 <= bh <= 55) and (bw / max(1, bh) >= 3.0) and area >= 350:
                rm = cand[y:y + bh, x:x + bw] > 0
                if rm.any() and float(np.mean(gx[y:y + bh, x:x + bw][rm])) > 65.0:
                    raw_sub_boxes[idx].append((max(0, x - 6), max(0, y_band + y - 6),
                                               min(w, x + bw + 6), min(h, y_band + y + bh + 6)))

    # Temporal persistence (>= min_persist consecutive frames) for subtitles
    has_sub = [len(b) > 0 for b in raw_sub_boxes]
    persist_boxes = [[] for _ in range(n)]
    i = 0
    while i < n:
        if has_sub[i]:
            j = i
            while j < n and has_sub[j]:
                j += 1
            if (j - i) >= min_persist:
                merged_box = (
                    min(b for f in range(i, j) for b in raw_sub_boxes[f]),
                    min(b for f in range(i, j) for b in raw_sub_boxes[f]),
                    max(b for f in range(i, j) for b in raw_sub_boxes[f]),
                    max(b for f in range(i, j) for b in raw_sub_boxes[f]),
                )
                for k in range(max(0, i - 2), min(n, j + 2)):
                    persist_boxes[k].append(merged_box)
            i = j
        else:
            i += 1

    # 2. Static corner watermark boxes (frozen high-gradient pixels across >= 8 sampled frames)
    wm_boxes = []
    if n >= 15:
        step = max(1, n // 12)
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in orig_bgr_list[::step][:12] if f is not None]
        if len(grays) >= 6:
            stack = np.stack(grays, axis=0)
            t_std = np.std(stack, axis=0)
            mean_img = np.mean(stack, axis=0)
            gx_m = np.abs(cv2.Sobel(mean_img, cv2.CV_32F, 1, 0, ksize=3))
            gy_m = np.abs(cv2.Sobel(mean_img, cv2.CV_32F, 0, 1, ksize=3))
            static_edge = ((t_std < 3.5) & ((gx_m + gy_m) > 55.0)).astype(np.uint8)
            # Restrict to outer 18% border bands where watermarks/logos sit
            border = np.zeros_like(static_edge)
            bh_m, bw_m = int(0.18 * h), int(0.18 * w)
            border[:bh_m, :] = 1
            border[h - bh_m:, :] = 1
            border[:, :bw_m] = 1
            border[:, w - bw_m:] = 1
            wm_mask = cv2.morphologyEx(static_edge & border, cv2.MORPH_CLOSE, k_line)
            num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(wm_mask)
            for lbl in range(1, num_lbl):
                x, y, bw, bh, area = stats[lbl]
                if bw >= 35 and bh >= 12 and area >= 220:
                    wm_boxes.append((max(0, x - 6), max(0, y - 6), min(w, x + bw + 6), min(h, y + bh + 6)))

    return [persist_boxes[idx] + wm_boxes for idx in range(n)]

def _copy_split_clip(src_dir, dst_dir, frame_names, s_i, e_i, second_subdir, second_ext):
    """Copy a clean frame range [s_i, e_i) into dst_dir, renumbering from 00001."""
    os.makedirs(os.path.join(dst_dir, "origin_image"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, second_subdir), exist_ok=True)

    for out_idx, src_idx in enumerate(range(s_i, e_i), start=1):
        src_jpg = frame_names[src_idx]
        stem = os.path.splitext(src_jpg)[0]
        dst_jpg = f"{out_idx:05d}.jpg"
        dst_sec = f"{out_idx:05d}{second_ext}"

        shutil.copy2(
            os.path.join(src_dir, "origin_image", src_jpg),
            os.path.join(dst_dir, "origin_image", dst_jpg),
        )
        src_sec_path = os.path.join(src_dir, second_subdir, stem + second_ext)
        if os.path.exists(src_sec_path):
            shutil.copy2(src_sec_path, os.path.join(dst_dir, second_subdir, dst_sec))


def _scan_one_real_pair(item, dry_run):
    bucket, bdir, clip, cdir, orig_dir, mos_dir = item
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    n_total = len(names)
    if n_total < MIN_FRAMES:
        return None

    raw_flags = []
    for fname in names:
        o_img = cv2.imread(os.path.join(orig_dir, fname))
        m_img = cv2.imread(os.path.join(mos_dir, fname))
        if o_img is None or m_img is None:
            raw_flags.append(True)
        else:
            raw_flags.append(has_subtitle_in_pair(o_img, m_img))

    sub_flags = [False] * n_total
    idx = 0
    while idx < n_total:
        if raw_flags[idx]:
            j = idx
            while j < n_total and raw_flags[j]:
                j += 1
            if (j - idx) >= 5:
                for k in range(max(0, idx - 2), min(n_total, j + 2)):
                    sub_flags[k] = True
            idx = j
        else:
            idx += 1

    n_bad = sum(sub_flags)
    if n_bad == 0:
        return (n_total, 1, 0, n_total, None)

    bad_pct = 100.0 * n_bad / n_total
    runs = split_clean_runs(sub_flags, MIN_FRAMES)
    kept_in_scene = sum(e_i - s_i for s_i, e_i in runs)
    kept_pct = 100.0 * kept_in_scene / n_total
    msg = (
        f"[real_pair split] {bucket}/{clip}: {n_bad}/{n_total} ({bad_pct:.1f}%) subtitled -> "
        f"{len(runs)} sub-clip(s) >= {MIN_FRAMES}f ({kept_in_scene}/{n_total} frames, {kept_pct:.1f}% kept)"
    )

    if not dry_run:
        for r_idx, (s_i, e_i) in enumerate(runs):
            new_dir = os.path.join(bdir, f"{clip}_sub{r_idx}")
            _copy_split_clip(cdir, new_dir, names, s_i, e_i, "mosaic_image", ".jpg")
        shutil.rmtree(cdir, ignore_errors=True)

    return (n_total, len(runs), 1 if not runs else 0, kept_in_scene, msg)

def process_real_pairs_folder(target_dir, prefix_filter="", dry_run=False, workers=8):
    """Scan bucket_* subfolders under target_dir in parallel and split around subtitles."""
    if not os.path.isdir(target_dir):
        print(f"[skip] real_pair dir not found: {target_dir}")
        return

    prefixes = tuple(p.strip() for p in prefix_filter.split(",") if p.strip())
    candidates = []
    for bucket in sorted(os.listdir(target_dir)):
        bdir = os.path.join(target_dir, bucket)
        if not os.path.isdir(bdir) or not bucket.startswith("bucket_"):
            continue
        for clip in sorted(os.listdir(bdir)):
            if "_dup" in clip or "_sub" in clip:
                continue
            if prefixes and not clip.startswith(prefixes):
                continue
            cdir = os.path.join(bdir, clip)
            orig_dir = os.path.join(cdir, "origin_image")
            mos_dir = os.path.join(cdir, "mosaic_image")
            if os.path.isdir(orig_dir) and os.path.isdir(mos_dir):
                candidates.append((bucket, bdir, clip, cdir, orig_dir, mos_dir))

    total_checked = total_kept_clips = total_removed_clips = 0
    total_frames_scanned = total_frames_kept = split_hits = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_one_real_pair, item, dry_run) for item in candidates]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Scanning real_pair", unit="scene")
        for fut in pbar:
            res = fut.result()
            if res is None:
                continue
            n_tot, n_kept, n_rem, f_kept, msg = res
            total_checked += 1
            total_frames_scanned += n_tot
            total_kept_clips += n_kept
            total_removed_clips += n_rem
            total_frames_kept += f_kept
            if msg:
                split_hits += 1
                tqdm.write(msg)
                pbar.set_postfix(splits=split_hits, refresh=False)

    pct = (100.0 * total_frames_kept / total_frames_scanned) if total_frames_scanned else 0.0
    print(
        f"[real_pair Summary] checked {total_checked} scene(s) ({total_frames_scanned:,} frames) -> "
        f"{total_kept_clips} clean clip(s) kept ({total_frames_kept:,} frames, {pct:.1f}% survival), "
        f"{total_removed_clips} scene(s) fully discarded.\n"
    )

def _scan_one_synthetic(item, pool_dir, dry_run):
    clip, cdir, orig_dir = item
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))
    n_total = len(names)
    if n_total < MIN_FRAMES:
        return None

    frames = [cv2.imread(os.path.join(orig_dir, fname)) for fname in names]
    sub_flags = detect_single_image_subtitles(frames)

    n_bad = sum(sub_flags)
    if n_bad == 0:
        return (n_total, 1, 0, n_total, None)

    bad_pct = 100.0 * n_bad / n_total
    runs = split_clean_runs(sub_flags, MIN_FRAMES)
    kept_in_clip = sum(e_i - s_i for s_i, e_i in runs)
    kept_pct = 100.0 * kept_in_clip / n_total
    msg = (
        f"[synthetic split] {clip}: {n_bad}/{n_total} ({bad_pct:.1f}%) subtitled -> "
        f"{len(runs)} sub-clip(s) >= {MIN_FRAMES}f ({kept_in_clip}/{n_total} frames, {kept_pct:.1f}% kept)"
    )

    if not dry_run:
        for r_idx, (s_i, e_i) in enumerate(runs):
            new_dir = os.path.join(pool_dir, f"{clip}_sub{r_idx}")
            _copy_split_clip(cdir, new_dir, names, s_i, e_i, "mask", ".png")
        shutil.rmtree(cdir, ignore_errors=True)

    return (n_total, len(runs), 1 if not runs else 0, kept_in_clip, msg)

def process_synthetic_pool_folder(pool_dir, prefix_filter="", dry_run=False, workers=8):
    """Scan clip folders under synthetic_pool in parallel and split around subtitles."""
    if not os.path.isdir(pool_dir):
        print(f"[skip] synthetic_pool dir not found: {pool_dir}")
        return

    prefixes = tuple(p.strip() for p in prefix_filter.split(",") if p.strip())
    candidates = []
    for clip in sorted(os.listdir(pool_dir)):
        if clip.startswith(".") or "_sub" in clip:
            continue
        if prefixes and not clip.startswith(prefixes):
            continue
        cdir = os.path.join(pool_dir, clip)
        orig_dir = os.path.join(cdir, "origin_image")
        mask_dir = os.path.join(cdir, "mask")
        if os.path.isdir(orig_dir) and os.path.isdir(mask_dir):
            candidates.append((clip, cdir, orig_dir))

    total_checked = total_kept_clips = total_removed_clips = 0
    total_frames_scanned = total_frames_kept = split_hits = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_scan_one_synthetic, item, pool_dir, dry_run) for item in candidates]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Scanning synthetic_pool", unit="clip")
        for fut in pbar:
            res = fut.result()
            if res is None:
                continue
            n_tot, n_kept, n_rem, f_kept, msg = res
            total_checked += 1
            total_frames_scanned += n_tot
            total_kept_clips += n_kept
            total_removed_clips += n_rem
            total_frames_kept += f_kept
            if msg:
                split_hits += 1
                tqdm.write(msg)
                pbar.set_postfix(splits=split_hits, refresh=False)

    pct = (100.0 * total_frames_kept / total_frames_scanned) if total_frames_scanned else 0.0
    print(
        f"[synthetic_pool Summary] checked {total_checked} clip(s) ({total_frames_scanned:,} frames) -> "
        f"{total_kept_clips} clean clip(s) kept ({total_frames_kept:,} frames, {pct:.1f}% survival), "
        f"{total_removed_clips} clip(s) fully discarded.\n"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split real_pair and synthetic_pool clips around burned-in subtitles.")
    parser.add_argument("--real_pairs_dir", default=os.path.join("datasets", "real_pair"), help="Path to real_pair folder")
    parser.add_argument("--synthetic_dir", default=os.path.join("datasets", "synthetic_pool"), help="Path to synthetic_pool folder")
    parser.add_argument("--only", choices=["all", "real", "synthetic"], default="all", help="Which dataset(s) to scan")
    parser.add_argument("--prefix", default="", help="Optional folder prefix filter")
    parser.add_argument("--dry_run", action="store_true", help="Preview splits without modifying folders")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads to use")
    args = parser.parse_args()

    if args.only in ("all", "real"):
        process_real_pairs_folder(args.real_pairs_dir, prefix_filter=args.prefix, dry_run=args.dry_run, workers=args.workers)
    if args.only in ("all", "synthetic"):
        process_synthetic_pool_folder(args.synthetic_dir, prefix_filter=args.prefix, dry_run=args.dry_run, workers=args.workers)
