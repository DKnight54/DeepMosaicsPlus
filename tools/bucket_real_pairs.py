"""
Auto-bucket real censored/uncensored video pairs by detected mosaic size.

Given two folders — one of censored clips, one of matching uncensored clips,
each pair sharing the SAME filename across the two folders — this:
  1. Extracts frames from both videos into origin_image/ (uncensored) and
     mosaic_image/ (censored) — the exact layout util/dataloader.py's
     real-pair training path expects (see util.dataloader.VideoLoader).
  2. Runs the same block-size detector used by tools/test_mosaic_size.py on
     the censored frames to estimate the mosaic's size as % of frame short
     side.
  3. Copies the resulting pair folder into <output_dir>/bucket_<N>/<name>/,
     where N is the nearest bucket in your --model_manifest.

Usage:
    python tools/bucket_real_pairs.py \
        --censored_dir datasets/raw/censored \
        --uncensored_dir datasets/raw/uncensored \
        --output_dir datasets/real_pairs \
        --mosaic_position_model_path mosaic_position.pth \
        --model_manifest pretrained_models/mosaic/size_manifest.json
"""
import os
import sys
import glob
import time
import shutil
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import cv2
from tqdm import tqdm
from cores import Options
from models import loadmodel, runmodel
from util import mosaic as mosaic_util
from util import ffmpeg as ffmpeg_util
from util import util as util_
from util import image_processing as impro
from models.yolo_mosaic_detector import YoloMosaicDetector
from tools.filter_subtitled_scenes import (
    has_subtitle_in_pair,
    split_clean_runs,
    find_fullframe_overlay_boxes,
    crop_overlaps_boxes,
    MIN_FRAMES as SUB_MIN_FRAMES,
)
from util.video_sync import resolve_pair_cfr, align_pair_folders_dtw, discover_groups, match_group



def find_pairs(censored_dir, uncensored_dir):
    censored_files = {f for f in os.listdir(censored_dir) if util_.is_video(os.path.join(censored_dir, f))}
    uncensored_files = {f for f in os.listdir(uncensored_dir) if util_.is_video(os.path.join(uncensored_dir, f))}
    common = sorted(censored_files & uncensored_files)
    pairs = []
    for f in common:
        name, _ext = os.path.splitext(f)
        pairs.append((name, os.path.join(censored_dir, f), os.path.join(uncensored_dir, f)))

    only_censored = censored_files - uncensored_files
    only_uncensored = uncensored_files - censored_files
    if only_censored:
        print(f"Note: {len(only_censored)} file(s) in censored/ have no matching filename in "
              f"uncensored/, skipped: {sorted(only_censored)}")
    if only_uncensored:
        print(f"Note: {len(only_uncensored)} file(s) in uncensored/ have no matching filename in "
              f"censored/, skipped: {sorted(only_uncensored)}")
    return pairs


def _edge_median(values, window):
    """Sliding median with edge replication."""
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode='edge')
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))], dtype=values.dtype)

def _match_scene_color_to_origin(cropped_pairs, dirty_flags, masks_small):
    """Apply one constant scene-wide per-channel mean/std color transform to mosaic crops
    using ONLY unmasked outer-border pixels (mask == 0), matching origin color space cleanly."""
    o_px, m_px = [], []
    for i, pair in enumerate(cropped_pairs):
        if dirty_flags[i] or pair is None:
            continue
        orig_bgr, mos_bgr = pair
        bg = masks_small[i] == 0
        if int(np.sum(bg)) < 256:
            continue
        o_px.append(orig_bgr[bg].astype(np.float32))
        m_px.append(mos_bgr[bg].astype(np.float32))
    if not o_px:
        return cropped_pairs

    o_all = np.concatenate(o_px, axis=0)
    m_all = np.concatenate(m_px, axis=0)
    o_mean, o_std = o_all.mean(axis=0), o_all.std(axis=0)
    m_mean, m_std = m_all.mean(axis=0), m_all.std(axis=0)
    gain = np.where(m_std > 1.0, o_std / m_std, 1.0)
    gain = np.clip(gain, 0.75, 1.35)  # guard against extreme outlier shifts

    out_pairs = []
    for i, pair in enumerate(cropped_pairs):
        if dirty_flags[i] or pair is None:
            out_pairs.append(pair)
            continue
        orig_bgr, mos_bgr = pair
        adj = (mos_bgr.astype(np.float32) - m_mean) * gain + o_mean
        out_pairs.append((orig_bgr, np.clip(adj, 0, 255).astype(np.uint8)))
    return out_pairs

def crop_pair_to_mosaic(netM, opt, origin_dir, mosaic_dir, outsize,
                        ex_range=(1.2, 1.7), min_frames=100,
                        gpu_lock=None, show_pbar=True):
    """Crop every frame of a pair to the mosaic region and resize to outsize x outsize."""
    names = sorted(f for f in os.listdir(mosaic_dir) if f.lower().endswith('.jpg'))
    n_orig = len([f for f in os.listdir(origin_dir) if f.lower().endswith('.jpg')])
    names = names[:min(len(names), n_orig)]
    n = len(names)
    if n < min_frames or n == 0:
         return None, f"too few frames ({n} < {min_frames})"

    min_size = getattr(opt, 'min_mosaic_size', 100)
    pos = np.zeros((n, 3), dtype=np.int64)

    use_yolo_batch = isinstance(netM, YoloMosaicDetector) and hasattr(netM, 'detect_batch')
    batch_size = 16

    pbar_loc = tqdm(total=n, desc='  locating mosaic', leave=False, disable=not show_pbar)
    raw_masks = [None] * n
    for start_i in range(0, n, batch_size):
        chunk_names = names[start_i:start_i + batch_size]
        chunk_imgs = []
        valid_indices = []

        for offset, name in enumerate(chunk_names):
            frame = impro.imread(os.path.join(mosaic_dir, name))
            if frame is not None:
                chunk_imgs.append(frame)
                valid_indices.append(start_i + offset)

        if chunk_imgs:
            if gpu_lock is not None:
                with gpu_lock:
                    if use_yolo_batch:
                        dets = netM.detect_batch(chunk_imgs, opt)
                    else:
                        dets = [runmodel.get_mosaic_position(im, netM, opt) for im in chunk_imgs]
            else:
                if use_yolo_batch:
                    dets = netM.detect_batch(chunk_imgs, opt)
                else:
                    dets = [runmodel.get_mosaic_position(im, netM, opt) for im in chunk_imgs]

            for idx, (mask, x, y, size) in zip(valid_indices, dets):
                if mask is not None and size > min_size:
                    pos[idx] = (x, y, size)
                    raw_masks[idx] = mask

        pbar_loc.update(len(chunk_names))
    pbar_loc.close()

    valid = pos[:, 2] > 0
    if not valid.any():
        return None, "no valid mosaic detections in extracted frames"

    idx = np.arange(n)
    prev_i = np.maximum.accumulate(np.where(valid, idx, -1))
    next_i = np.minimum.accumulate(np.where(valid, idx, n)[::-1])[::-1]
    for i in np.where(~valid)[0]:
        cands = [j for j in (prev_i[i], next_i[i]) if 0 <= j < n]
        pos[i] = pos[min(cands, key=lambda j: abs(j - i))]

    win = getattr(opt, 'medfilt_num', 5)
    win = win if win % 2 == 1 else win + 1
    for c in range(3):
        pos[:, c] = _edge_median(pos[:, c], win)

    ex_mul = random.uniform(*ex_range)
    orig_paths = [os.path.join(origin_dir, name) for name in names]
    mos_paths = [os.path.join(mosaic_dir, name) for name in names]
    overlay_boxes = find_fullframe_overlay_boxes(orig_paths, mos_paths)

    dirty_flags = []
    cropped_pairs = []
    masks_small = []
    sides, size_mismatch = [], False

    for i, name in enumerate(tqdm(names, desc='  Cropping Mosaic area', leave=False, disable=not show_pbar)):
        mosaic = impro.imread(mos_paths[i])
        origin = impro.imread(orig_paths[i])
        if mosaic is None or origin is None:
            dirty_flags.append(True)
            cropped_pairs.append(None)
            masks_small.append(None)
            continue
        h, w = mosaic.shape[:2]
        if origin.shape[:2] != (h, w):
            origin = cv2.resize(origin, (w, h), interpolation=cv2.INTER_AREA)
            size_mismatch = True
        half = max(8, min(int(pos[i, 2] * ex_mul), h // 2, w // 2))
        cx = int(np.clip(pos[i, 0], half, w - half))
        cy = int(np.clip(pos[i, 1], half, h - half))
        crop_rect = (cx - half, cy - half, cx + half, cy + half)
        if i < len(overlay_boxes) and crop_overlaps_boxes(crop_rect, overlay_boxes[i]):
            dirty_flags.append(True)
            cropped_pairs.append(None)
            masks_small.append(None)
            continue

        dirty_flags.append(False)
        box = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        interp = cv2.INTER_AREA if 2 * half > outsize else cv2.INTER_CUBIC
        cropped_pairs.append((
            cv2.resize(origin[box], (outsize, outsize), interpolation=interp),
            cv2.resize(mosaic[box], (outsize, outsize), interpolation=interp),
        ))
        mask_crop = raw_masks[i][box] if i < len(raw_masks) and raw_masks[i] is not None else np.zeros((2 * half, 2 * half), dtype=np.uint8)
        masks_small.append(cv2.resize(mask_crop, (outsize, outsize), interpolation=cv2.INTER_NEAREST))
        sides.append(2 * half)

    # Clear raw full-size frames and keep only clean runs >= min_frames
    for d in (origin_dir, mosaic_dir):
        for f in os.listdir(d):
            if f.lower().endswith('.jpg'):
                os.remove(os.path.join(d, f))

    runs = split_clean_runs(dirty_flags, min_frames) if any(dirty_flags) else (
        [(0, len(cropped_pairs))] if len(cropped_pairs) >= min_frames else []
    )
    if not runs:
        n_ovl = sum(dirty_flags)
        return None, f"overlay overlap dropped {n_ovl}/{len(dirty_flags)} frames (0 clean runs >= {min_frames}f)"

    parent_dir = os.path.dirname(origin_dir)

    cropped_pairs = _match_scene_color_to_origin(cropped_pairs, dirty_flags, masks_small)
    for r_idx, (s_i, e_i) in enumerate(runs):
        if len(runs) == 1 and r_idx == 0:
            cur_orig, cur_mos = origin_dir, mosaic_dir
        else:
            sub_dir = f"{parent_dir}_sub{r_idx}"
            cur_orig = os.path.join(sub_dir, 'origin_image')
            cur_mos = os.path.join(sub_dir, 'mosaic_image')
            os.makedirs(cur_orig, exist_ok=True)
            os.makedirs(cur_mos, exist_ok=True)
        for out_i, src_i in enumerate(range(s_i, e_i), start=1):
            impro.imwrite(os.path.join(cur_orig, f"{out_i:05d}.jpg"), cropped_pairs[src_i][0])
            impro.imwrite(os.path.join(cur_mos, f"{out_i:05d}.jpg"), cropped_pairs[src_i][1])

    if len(runs) > 1:
        shutil.rmtree(parent_dir, ignore_errors=True)

    kept_frames = sum(e_i - s_i for s_i, e_i in runs)
    return {
        'frames': kept_frames,
        'median_side': int(np.median(sides)),
        'ex_mul': ex_mul,
        'enlarged_pct': 100.0 * float(np.mean(np.array(sides) < outsize)),
        'resized_origin': size_mismatch,
        'sub_clips': len(runs),
    }, None

def _measure_postcrop_scene_pct(clip_dir, min_pct, max_pct, fallback_pct):
    """Measure mosaic block % on already-cropped origin_image/mosaic_image pairs."""
    orig_dir = os.path.join(clip_dir, 'origin_image')
    mos_dir = os.path.join(clip_dir, 'mosaic_image')
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.jpg'))
    if not names:
        return fallback_pct
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    pairs = []
    for fn in names[::max(1, len(names) // 8)][:8]:
        oi = cv2.imread(os.path.join(orig_dir, fn))
        mi = cv2.imread(os.path.join(mos_dir, fn))
        if oi is None or mi is None or oi.shape != mi.shape:
            continue
        diff = (np.abs(oi.astype(np.int16) - mi.astype(np.int16)).mean(axis=2) > 12).astype(np.uint8) * 255
        mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k_close)
        pairs.append((mi, mask))
    res = mosaic_util.estimate_mosaic_block_pct_multi(
        pairs, min_confidence=0.12, min_pct=max(2.0, min_pct), max_pct=max(max_pct, 12.0)
    )
    return res['pct'] if res else fallback_pct

def _split_subtitles_in_place(bucket_dir, min_frames=SUB_MIN_FRAMES):
    """Split a newly cropped pair folder into _sub0, _sub1... if burned-in subtitles are detected."""
    orig_dir = os.path.join(bucket_dir, 'origin_image')
    mos_dir = os.path.join(bucket_dir, 'mosaic_image')
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.jpg'))
    n = len(names)
    if n < min_frames:
        return 0

    raw = []
    for fn in names:
        oi = cv2.imread(os.path.join(orig_dir, fn))
        mi = cv2.imread(os.path.join(mos_dir, fn))
        raw.append(True if (oi is None or mi is None) else has_subtitle_in_pair(oi, mi))

    flags = [False] * n
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            if (j - i) >= 5:
                for k in range(max(0, i - 2), min(n, j + 2)):
                    flags[k] = True
            i = j
        else:
            i += 1

    if sum(flags) == 0:
        return 1

    runs = split_clean_runs(flags, min_frames)
    for r_idx, (s_i, e_i) in enumerate(runs):
        sub_dir = f"{bucket_dir}_sub{r_idx}"
        os.makedirs(os.path.join(sub_dir, 'origin_image'), exist_ok=True)
        os.makedirs(os.path.join(sub_dir, 'mosaic_image'), exist_ok=True)
        for out_i, src_i in enumerate(range(s_i, e_i), start=1):
            shutil.copy2(os.path.join(orig_dir, names[src_i]), os.path.join(sub_dir, 'origin_image', f"{out_i:05d}.jpg"))
            shutil.copy2(os.path.join(mos_dir, names[src_i]), os.path.join(sub_dir, 'mosaic_image', f"{out_i:05d}.jpg"))
    shutil.rmtree(bucket_dir, ignore_errors=True)
    return len(runs)

def _fmt_stamp_ms(sec):
    s = max(0.0, float(sec))
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{s % 60:06.3f}"


def extract_frames(videopath, out_dir, fps, start_sec=None, duration_sec=None):
    """Extract frames from `videopath` into `out_dir` (%05d.jpg) with sub-second precision."""
    util_.makedirs(out_dir)
    out_pattern = os.path.join(out_dir, '%05d.jpg')
    if start_sec is not None and duration_sec is not None:
        ffmpeg_util.video2image(
            videopath,
            out_pattern,
            fps=fps,
            start_time=_fmt_stamp_ms(start_sec),
            last_time=_fmt_stamp_ms(max(0.1, float(duration_sec))),
        )
    else:
        ffmpeg_util.video2image(videopath, out_pattern, fps=fps)


def detect_pct(opt, netM, censored_video, n_samples, min_pct, max_pct,
               scan_stride_sec=2.0, max_gap_sec=5.0, min_scene_sec=3.0,
               gpu_lock=None, show_pbar=True, progress_cb=None):
    """Single-pass sequential scan using cap.grab().
    Discovers continuous censored scenes and estimates mosaic block % per scene.
    Returns a list of scene dicts:
      [{'start': float, 'duration': float, 'pct': float,
        'confidence': float, 'n_nonsquare': int, 'n_samples': int}, ...]
    """
    cap = cv2.VideoCapture(censored_video)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, int(round(fps * scan_stride_sec)))
    min_size = getattr(opt, 'min_mosaic_size', 100)
    min_conf = getattr(opt, 'auto_model_min_confidence', 0.15)
    max_samples_per_scene = max(n_samples, 12)
    use_yolo = isinstance(netM, YoloMosaicDetector)

    hits = []  # list of (timestamp_sec, frame, mask)
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc='  scanning video', unit='frame', leave=False, disable=not show_pbar)
    while True:
        if not cap.grab():
            break

        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                orig_h, orig_w = frame.shape[:2]
                max_side = max(orig_h, orig_w)

                if use_yolo and max_side > 640:
                    scale = 640.0 / max_side
                    scan_frame = cv2.resize(
                        frame,
                        (int(round(orig_w * scale)), int(round(orig_h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask_small, x, y, size = netM.detect(scan_frame, opt)
                    else:
                        mask_small, x, y, size = netM.detect(scan_frame, opt)
                    size = int(round(size / scale))
                    if mask_small is not None and size > min_size:
                        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask = None
                elif use_yolo:
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask, x, y, size = netM.detect(frame, opt)
                    else:
                        mask, x, y, size = netM.detect(frame, opt)
                else:
                    if gpu_lock is not None:
                        with gpu_lock:
                            mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)
                    else:
                        mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)

                if mask is not None and size > min_size:
                    hits.append((frame_idx / fps, frame, mask))
                    if show_pbar:
                        pbar.set_postfix(hits=len(hits), refresh=False)

        frame_idx += 1
        pbar.update(1)
        if progress_cb is not None and (frame_idx % (step * 5) == 0):
            progress_cb(int(100 * frame_idx / max(1, total_frames)))

    pbar.close()
    cap.release()

    if not hits:
        return []

    # Group adjacent hits into continuous scenes
    raw_scenes = []
    cur = [hits[0]]
    for item in hits[1:]:
        if item[0] - cur[-1][0] <= max_gap_sec:
            cur.append(item)
        else:
            raw_scenes.append(cur)
            cur = [item]
    raw_scenes.append(cur)

    # Estimate mosaic block % individually for each scene
    scenes = []
    video_dur = total_frames / fps
    for group in raw_scenes:
        start_sec = max(0.0, group[0][0] - (scan_stride_sec * 0.5))
        end_sec = min(video_dur, group[-1][0] + (scan_stride_sec * 0.5))
        duration_sec = end_sec - start_sec

        if duration_sec < min_scene_sec:
            continue

        if len(group) > max_samples_per_scene:
            idxs = np.linspace(0, len(group) - 1, max_samples_per_scene, dtype=int)
            sampled = [group[i] for i in idxs]
        else:
            sampled = group

        pairs = [(f, m) for (_ts, f, m) in sampled]
        res = mosaic_util.estimate_mosaic_block_pct_multi(
            pairs,
            min_confidence=min_conf,
            min_pct=1.2,
            max_pct=max(max_pct, 12.0),
        )
        if res is None:
            continue

        scenes.append({
            'start': start_sec,
            'duration': duration_sec,
            'pct': res['pct'],
            'confidence': res['confidence'],
            'n_nonsquare': res.get('n_nonsquare', 0),
            'n_samples': len(pairs),
        })

    return scenes


def maybe_reuse_uncensored(pairs, mode, synthetic_source_dir):
    if mode == 'no':
        return
    if mode == 'ask':
        answer = input(f"Copy uncensored clips into {synthetic_source_dir} for synthetic generation? [y/N]: ").strip().lower()
        if answer not in ('y', 'yes'):
            return
    os.makedirs(synthetic_source_dir, exist_ok=True)
    copied = 0
    for _name, _censored_path, uncensored_path in pairs:
        dst = os.path.join(synthetic_source_dir, os.path.basename(uncensored_path))
        if os.path.exists(dst):
            continue
        shutil.copy2(uncensored_path, dst)
        copied += 1
    if copied:
        print(f"Copied {copied} new uncensored clip(s) into {synthetic_source_dir}.")


def _cleanup_partial_pair(output_dir, pair_name):
    """Remove any leftover scene folders for a pair that did not finish cleanly."""
    for sub in ('bucket_*', '_staging'):
        for d in glob.glob(os.path.join(output_dir, sub, f"{pair_name}_scene*")):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


def _find_existing_pair_scenes(output_dir, pair_name):
    """Find already-extracted scene folders on disk for pairs built before .done markers existed."""
    found = []
    pattern = os.path.join(output_dir, 'bucket_*', f"{pair_name}_scene*")
    for d in sorted(glob.glob(pattern)):
        orig_d = os.path.join(d, 'origin_image')
        mos_d = os.path.join(d, 'mosaic_image')
        if os.path.isdir(orig_d) and os.path.isdir(mos_d):
            n_o = len([f for f in os.listdir(orig_d) if f.lower().endswith('.jpg')])
            n_m = len([f for f in os.listdir(mos_d) if f.lower().endswith('.jpg')])
            if n_o >= 100 and n_o == n_m:
                bucket_folder = os.path.basename(os.path.dirname(d))
                try:
                    b_val = float(bucket_folder.replace('bucket_', ''))
                except ValueError:
                    b_val = 0.0
                found.append((os.path.basename(d), 0.0, b_val))
    return found


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--censored_dir', required=True, help='folder of real censored clips or subfolders')
    parser.add_argument('--uncensored_dir', required=True, help='folder of matching uncensored clips or subfolders')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fps', type=int, default=0, help='target FPS (0 = auto matched native CFR, >0 = manual override)')
    parser.add_argument('--min_frames', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--outsize', type=int, default=572)
    parser.add_argument('--no_crop', action='store_true')
    parser.add_argument('--reuse_uncensored_for_synthetic', choices=['ask', 'yes', 'no'], default='ask')
    parser.add_argument('--yolo_conf', type=float, default=0.35)
    parser.add_argument('--synthetic_source_dir', default=os.path.join('datasets', 'raw', 'synthetic_source'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--gui_progress', action='store_true')
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if '--model_path' not in sys.argv and '--mosaic_position_model_path' in sys.argv:
        pos_idx = sys.argv.index('--mosaic_position_model_path')
        sys.argv += ['--model_path', sys.argv[pos_idx + 1], '--mode', 'clean', '--netG', 'unet_128']

    opt = Options().getparse(test_flag=True)
    opt.yolo_conf = known.yolo_conf

    if not os.path.isdir(known.censored_dir) or not os.path.isdir(known.uncensored_dir):
        print("Error: --censored_dir or --uncensored_dir not found.")
        sys.exit(1)

    if opt.mosaic_position_model_path.lower().endswith(('.pt', '.onnx')):
        netM = YoloMosaicDetector(opt.mosaic_position_model_path, conf_thresh=known.yolo_conf)
    else:
        netM = loadmodel.bisenet(opt, 'mosaic')

    manifest_path = opt.model_manifest
    manifest = loadmodel.load_size_manifest(manifest_path)
    min_pct, max_pct = min(manifest.keys()), max(manifest.keys())
    print(f"Bucket range from manifest: {min_pct:g}%-{max_pct:g}%")

    groups = discover_groups(known.censored_dir, known.uncensored_dir)
    if not groups:
        print("No matching subfolders or root video filenames found.")
        sys.exit(1)

    # Link/copy raw uncensored files cleanly if requested (never modifies raw source)
    raw_pairs = [(lbl, c_list[0], u_list[0]) for lbl, c_list, u_list in groups if len(c_list) == 1 and len(u_list) == 1]
    maybe_reuse_uncensored(raw_pairs, known.reuse_uncensored_for_synthetic, known.synthetic_source_dir)

    os.makedirs(known.output_dir, exist_ok=True)
    done_dir = os.path.join(known.output_dir, '.done')
    os.makedirs(done_dir, exist_ok=True)

    todo_groups = []
    skipped = 0
    for item in groups:
        lbl = item[0]
        sentinel = os.path.join(done_dir, f"{lbl}.done")
        if known.no_resume and os.path.exists(sentinel):
            os.remove(sentinel)
        if not known.no_resume and os.path.exists(sentinel):
            skipped += 1
        else:
            _cleanup_partial_pair(known.output_dir, lbl)
            todo_groups.append(item)

    workers = max(1, min(int(known.workers), 8))
    use_gui = bool(known.gui_progress)
    show_inner_pbar = (workers == 1) and (not use_gui)
    print(f"Found {len(groups)} group(s) — {skipped} completed, {len(todo_groups)} to process (workers={workers}).")
    if not todo_groups:
        return

    gpu_lock = threading.Lock()
    state_lock = threading.Lock()
    slot_pool = list(range(workers))
    worker_slots = {}
    completed = 0
    total_todo = len(todo_groups)
    results = []

    def emit_gui(msg=""):
        if use_gui:
            with state_lock:
                w_str = ";".join(worker_slots[k] for k in sorted(worker_slots))
                dn = completed
            print(f"@@PROGRESS|done={dn}|total={total_todo}|workers={w_str}|msg={msg}", flush=True)

    def set_slot(slot, name, stage, pct=None):
        short = name if len(name) <= 22 else name[:19] + "..."
        with state_lock:
            worker_slots[slot] = f"W{slot + 1}: {short} [{stage}]@{pct if pct is not None else ''}"
        emit_gui()

    def process_one_group(g_item):
        nonlocal completed
        label, cen_paths, unc_paths = g_item
        t0 = time.time()
        with state_lock:
            slot = slot_pool.pop(0)
        set_slot(slot, label, "sync", 0)
        logs, g_res = [], []
        try:
            qc_dir = os.path.join(known.output_dir, '_sync_qc')
            scenes = match_group(label, cen_paths, unc_paths, qc_dir, export_dir=None)
            if not scenes:
                logs.append(f"  {label}: 0 matched scenes.")
            mos_by_file = {
                os.path.basename(cp): detect_pct(opt, netM, cp, known.n_samples, min_pct, max_pct, gpu_lock=gpu_lock, show_pbar=False)
                for cp in cen_paths
            }
            sub_scenes = []
            for sc in scenes:
                m_list = mos_by_file.get(sc['cen_source'], [])
                c0, c1 = sc['cen_start'], sc['cen_start'] + sc['duration']
                off = sc['unc_start'] - sc['cen_start']
                for msc in m_list:
                    ov0 = max(c0, msc['start'])
                    ov1 = min(c1, msc['start'] + msc['duration'])
                    if ov1 - ov0 < 4.5:
                        continue
                    cur_t = ov0
                    while cur_t < ov1 - 4.0:
                        seg_dur = min(45.0, ov1 - cur_t)
                        sub_scenes.append({
                            'clip_name': f"{os.path.splitext(sc['clip_name'])[0]}_m{len(sub_scenes):03d}.mp4",
                            'cen_source': sc['cen_source'],
                            'unc_source': sc['unc_source'],
                            'cen_start': round(cur_t, 4),
                            'unc_start': round(cur_t + off, 4),
                            'duration': round(seg_dur, 4),
                            'pct': msc['pct'],
                        })
                        cur_t += seg_dur

            for s_idx, sc in enumerate(sub_scenes):
                set_slot(slot, label, f"scene {s_idx + 1}/{len(sub_scenes)}", int(100 * s_idx / max(1, len(sub_scenes))))
                cen_src = next(p for p in cen_paths if os.path.basename(p) == sc['cen_source'])
                unc_src = next(p for p in unc_paths if os.path.basename(p) == sc['unc_source'])
                scene_fps = float(known.fps) if known.fps > 0 else resolve_pair_cfr(cen_src, unc_src)
                if sc['duration'] * scene_fps < known.min_frames:
                    continue

                clip_tag = os.path.splitext(sc['clip_name'])[0]
                stage_dir = os.path.join(known.output_dir, '_staging', clip_tag)
                origin_dir = os.path.join(stage_dir, 'origin_image')
                mosaic_dir = os.path.join(stage_dir, 'mosaic_image')
                extract_frames(unc_src, origin_dir, scene_fps, start_sec=sc['unc_start'], duration_sec=sc['duration'])
                extract_frames(cen_src, mosaic_dir, scene_fps, start_sec=sc['cen_start'], duration_sec=sc['duration'])

                ok_sync, _, sync_info = align_pair_folders_dtw(origin_dir, mosaic_dir, max_band=3, min_mean_corr=0.80)
                if not ok_sync:
                    shutil.rmtree(stage_dir, ignore_errors=True)
                    logs.append(f"    {clip_tag}: skipped ({sync_info})")
                    continue

                stats, skip_reason = crop_pair_to_mosaic(
                    netM, opt, origin_dir, mosaic_dir, known.outsize,
                    min_frames=known.min_frames, gpu_lock=gpu_lock, show_pbar=show_inner_pbar,
                )
                if stats is None:
                    shutil.rmtree(stage_dir, ignore_errors=True)
                    logs.append(f"    {clip_tag}: skipped ({skip_reason})")
                    continue

                for sd in sorted(glob.glob(f"{stage_dir}*")):
                    sub_name = os.path.basename(sd)
                    post_pct = _measure_postcrop_scene_pct(sd, min_pct, max_pct, sc['pct'])
                    _, bucket = loadmodel.select_model_by_pct(post_pct, manifest_path)
                    bucket_dir = os.path.join(known.output_dir, f'bucket_{bucket:g}', sub_name)
                    os.makedirs(os.path.dirname(bucket_dir), exist_ok=True)
                    shutil.move(sd, bucket_dir)
                    n_clean = _split_subtitles_in_place(bucket_dir, min_frames=known.min_frames)
                    if n_clean > 0:
                        g_res.append((sub_name, post_pct, bucket))
                        logs.append(f"    {sub_name}: fps={scene_fps:.3f} {sync_info} pct={post_pct:.2f}% -> bucket {bucket:g}% (+{n_clean})")

            with open(os.path.join(done_dir, f"{label}.done"), 'w', encoding='utf-8') as f:
                f.write(f"ok clips={len(g_res)}\n")
        except Exception as e:
            logs.append(f"  [ERROR] {label}: {e}")
        finally:
            with state_lock:
                worker_slots.pop(slot, None)
                slot_pool.append(slot)
                slot_pool.sort()
                completed += 1
                idx_now = completed
                results.extend(g_res)
            print("\n".join([f"[{idx_now}/{total_todo}] {label} ({time.time() - t0:.1f}s) -> {len(g_res)} clip(s)"] + logs), flush=True)
            emit_gui(f"last: {label} ({len(g_res)} clip(s))")

    emit_gui("starting workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(process_one_group, g) for g in todo_groups]):
            fut.result()

    print(f"\nDone. {len(results)} clean scene(s) from {total_todo} group(s) bucketed under {known.output_dir}/bucket_<N>/")


if __name__ == '__main__':
    main()