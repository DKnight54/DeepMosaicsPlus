import os
import sys
import shutil
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_here, '..'))
from cores import Options
opt = Options()

import random
import datetime
import time
from tqdm import tqdm

import numpy as np
import cv2
import torch

from models import runmodel, loadmodel
from models.yolo_mosaic_detector import YoloMosaicDetector
import util.image_processing as impro
from util import filt, util, mosaic, data, ffmpeg
from tools.filter_subtitled_scenes import detect_single_image_subtitles, split_clean_runs, MIN_FRAMES as SUB_MIN_FRAMES


opt.parser.add_argument('--datadir', type=str, default='your video dir', help='')
opt.parser.add_argument('--savedir', type=str, default='../datasets/video/face', help='')
opt.parser.add_argument('--interval', type=int, default=30, help='interval of split video ')
opt.parser.add_argument('--time', type=int, default=5, help='split video time')
opt.parser.add_argument('--minmaskarea', type=int, default=2000, help='')
opt.parser.add_argument('--quality', type=int, default=45, help='minimal quality')
opt.parser.add_argument('--scene_accept_ratio', type=float, default=1.0)
opt.parser.add_argument('--outsize', type=int, default=572)
opt.parser.add_argument('--startcnt', type=int, default=0)
opt.parser.add_argument('--minsize', type=int, default=96, help='minimal roi size')
opt.parser.add_argument('--no_sclectscene', action='store_true', help='')
opt.parser.add_argument('--yolo_conf', type=float, default=0.35)
opt.parser.add_argument('--workers', type=int, default=2,
                        help='parallel worker threads for FFmpeg extraction & cropping (default: 2)')
opt.parser.add_argument('--no_resume', action='store_true',
                        help='reprocess all source videos from scratch instead of skipping .done videos')
opt.parser.add_argument('--gui_progress', action='store_true',
                        help='emit machine-readable @@PROGRESS lines and suppress tqdm bars')

opt = opt.getparse()

util.makedirs(opt.savedir)
util.writelog(os.path.join(opt.savedir, 'opt.txt'),
              str(time.asctime(time.localtime(time.time()))) + '\n' + util.opt2str(opt))

videopaths = util.Traversal(opt.datadir)
videopaths = util.is_videos(videopaths)
if not videopaths:
    print(f"Error: no video files found in --datadir '{opt.datadir}'.")
    sys.exit(1)
random.shuffle(videopaths)

if opt.model_path.lower().endswith(('.pt', '.onnx')):
    net = YoloMosaicDetector(opt.model_path, conf_thresh=opt.yolo_conf)
else:
    net = loadmodel.bisenet(opt, 'roi')


def _next_available_startcnt(savedir, requested_start):
    max_id = requested_start - 1
    try:
        for entry in os.listdir(savedir):
            if entry.isdigit() and os.path.isdir(os.path.join(savedir, entry)):
                max_id = max(max_id, int(entry))
    except OSError:
        pass
    return max_id + 1


def _video_done_key(videopath):
    base = os.path.splitext(os.path.basename(videopath))[0]
    digest = hashlib.sha1(os.path.abspath(videopath).encode('utf-8', errors='replace')).hexdigest()[:8]
    return f"{base}_{digest}.done"


done_dir = os.path.join(opt.savedir, '.done')
os.makedirs(done_dir, exist_ok=True)

if opt.no_resume:
    for vp in videopaths:
        p = os.path.join(done_dir, _video_done_key(vp))
        if os.path.exists(p):
            os.remove(p)
    todo_videos = videopaths
    skipped_videos = 0
else:
    todo_videos = []
    skipped_videos = 0
    for vp in videopaths:
        if os.path.exists(os.path.join(done_dir, _video_done_key(vp))):
            skipped_videos += 1
        else:
            todo_videos.append(vp)

start_cnt = _next_available_startcnt(opt.savedir, opt.startcnt)
workers = max(1, min(int(opt.workers), 8))
use_gui = bool(opt.gui_progress)
show_inner_pbar = (workers == 1) and (not use_gui)

print(f"Found {len(videopaths)} video(s) — {skipped_videos} already completed, "
      f"{len(todo_videos)} to process (workers={workers}, next_clip={start_cnt:05d}).")

native_sides = []
gpu_lock = threading.Lock()
state_lock = threading.Lock()
cnt_lock = threading.Lock()

next_clip_id = start_cnt
slot_pool = list(range(workers))
worker_slots = {}  # slot_idx -> "W1: name [stage]@pct"
completed_videos = 0
total_todo = len(todo_videos)
starttime = datetime.datetime.now()


def _alloc_clip_id():
    global next_clip_id
    with cnt_lock:
        cid = next_clip_id
        next_clip_id += 1
        return cid


def _clean_worker_temp(worker_tmp):
    if os.path.isdir(worker_tmp):
        shutil.rmtree(worker_tmp, ignore_errors=True)
    os.makedirs(worker_tmp, exist_ok=True)


def _emit_gui_progress(msg=""):
    if not use_gui:
        return
    with state_lock:
        w_str = ";".join(worker_slots[k] for k in sorted(worker_slots))
        done_n = completed_videos
    print(f"@@PROGRESS|done={done_n}|total={total_todo}|workers={w_str}|msg={msg}", flush=True)


def _set_worker_status(slot, name, stage, pct=None):
    short = name if len(name) <= 22 else name[:19] + "..."
    pct_part = f"@{pct}" if pct is not None else "@"
    with state_lock:
        worker_slots[slot] = f"W{slot + 1}: {short} [{stage}]{pct_part}"
    _emit_gui_progress()


def _process_one_video(videopath):
    global completed_videos
    vname = os.path.basename(videopath)
    t0 = time.time()
    worker_tmp = os.path.join(opt.temp_dir, f"worker_{threading.get_ident()}")
    local_sides = []
    clips_created = 0

    with state_lock:
        slot = slot_pool.pop(0)
    _set_worker_status(slot, vname, "scan", 0)

    try:
        if opt.no_sclectscene:
            timestamps = ['00:00:00']
        else:
            timestamps = []
            fps, endtime, height, width = ffmpeg.get_video_infos(videopath)
            short_name = (vname[:28] + '...') if len(vname) > 31 else vname
            max_cut = int((endtime - opt.time) / opt.interval)
            cut_points = range(0, max(1, max_cut)) if endtime >= opt.time else range(0)

            pbar = tqdm(
                cut_points,
                desc=f"Scanning {short_name}",
                unit="clip",
                leave=False,
                disable=not show_inner_pbar,
            )
            for cut_point in pbar:
                stamp = util.second2stamp(cut_point * opt.interval)
                if len(cut_points) > 0:
                    _set_worker_status(slot, vname, "scan", int(100 * cut_point / len(cut_points)))
                if show_inner_pbar:
                    pbar.set_postfix_str(f"t={stamp} hits={len(timestamps)}", refresh=False)
                _clean_worker_temp(worker_tmp)
                ffmpeg.video2image(
                    videopath,
                    os.path.join(worker_tmp, '%05d.' + opt.tempimage_type),
                    fps=1,
                    start_time=stamp,
                    last_time=util.second2stamp(opt.time),
                )
                imagepaths = sorted(util.Traversal(worker_tmp))
                cnt = 0
                for i in range(min(opt.time, len(imagepaths))):
                    img = impro.imread(imagepaths[i])
                    if img is None:
                        continue
                    with gpu_lock:
                        if isinstance(net, YoloMosaicDetector):
                            mask, _, _, _ = net.detect(img, opt)
                            if mask is None:
                                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        else:
                            mask = runmodel.get_ROI_position(img, net, opt, keepsize=True)[0]
                    if not opt.all_mosaic_area:
                        mask = impro.find_mostlikely_ROI(mask)
                    x, y, size, area = impro.boundingSquare(mask, Ex_mul=1)
                    if area > opt.minmaskarea and size > opt.minsize and impro.Q_lapulase(img) > opt.quality:
                        cnt += 1
                if cnt >= opt.time * opt.scene_accept_ratio:
                    timestamps.append(stamp)
            pbar.close()

        util.writelog(os.path.join(opt.savedir, 'opt.txt'), videopath + '\n' + str(timestamps))

        for ts_idx, timestamp in enumerate(tqdm(timestamps, desc="Extracting", unit="scene", leave=False, disable=not show_inner_pbar)):
            _set_worker_status(slot, vname, f"extract {ts_idx + 1}/{len(timestamps)}", int(100 * ts_idx / max(1, len(timestamps))))
            cid = _alloc_clip_id()
            savecnt = '%05d' % cid
            origindir = os.path.join(opt.savedir, savecnt, 'origin_image')
            maskdir = os.path.join(opt.savedir, savecnt, 'mask')
            util.makedirs(origindir)
            util.makedirs(maskdir)

            _clean_worker_temp(worker_tmp)
            ffmpeg.video2image(
                videopath,
                os.path.join(worker_tmp, '%05d.' + opt.tempimage_type),
                fps=opt.fps,
                start_time=timestamp,
                last_time=util.second2stamp(opt.time),
            )

            imagepaths = sorted(util.Traversal(worker_tmp))
            imgs, masks = [], []
            ex_mul = random.uniform(1.35, 1.85)
            positions = []
            for imagepath in imagepaths:
                img = impro.imread(imagepath)
                if img is None:
                    continue
                with gpu_lock:
                    if isinstance(net, YoloMosaicDetector):
                        mask, _, _, _ = net.detect(img, opt)
                        if mask is None:
                            mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    else:
                        mask = runmodel.get_ROI_position(img, net, opt, keepsize=True)[0]
                imgs.append(img)
                masks.append(mask)
                x, y, size, area = impro.boundingSquare(mask, Ex_mul=ex_mul)
                positions.append([x, y, size])

            if not positions:
                shutil.rmtree(os.path.join(opt.savedir, savecnt), ignore_errors=True)
                continue

            positions = np.array(positions)
            for c in range(3):
                positions[:, c] = filt.medfilt(positions[:, c], opt.medfilt_num)

            cropped_pairs = []
            k_pad = random.randrange(15, 37, 2)
            pad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_pad, k_pad))
            mh = max(16, int(0.10 * opt.outsize))
            for i in range(len(imgs)):
                x, y, size = positions[i][0], positions[i][1], positions[i][2]
                tmp_cnt = i
                fallback_steps = 0
                while size < opt.minsize // 2 and fallback_steps < len(positions):
                    tmp_cnt -= 1
                    x, y, size = positions[tmp_cnt][0], positions[tmp_cnt][1], positions[tmp_cnt][2]
                    fallback_steps += 1
                if size < opt.minsize // 2:
                    continue

                h_img, w_img = imgs[i].shape[:2]
                y0, y1 = max(0, y - size), min(h_img, y + size)
                x0, x1 = max(0, x - size), min(w_img, x + size)
                if y1 <= y0 or x1 <= x0:
                    continue

                local_sides.append(min(y1 - y0, x1 - x0))
                img_out = impro.resize(imgs[i][y0:y1, x0:x1], opt.outsize, interpolation=cv2.INTER_CUBIC)
                mask_out = impro.resize(masks[i][y0:y1, x0:x1], opt.outsize, interpolation=cv2.INTER_CUBIC)
                dilated = cv2.dilate(mask_out, pad_kernel, iterations=1)
                safe = np.zeros_like(dilated)
                safe[mh:opt.outsize - mh, mh:opt.outsize - mh] = dilated[mh:opt.outsize - mh, mh:opt.outsize - mh]
                mask_out = np.maximum(mask_out, safe)
                cropped_pairs.append((img_out, mask_out))

            min_keep = min(SUB_MIN_FRAMES, max(1, int(opt.time * opt.fps * 0.8)))
            if len(cropped_pairs) < min_keep:
                shutil.rmtree(os.path.join(opt.savedir, savecnt), ignore_errors=True)
                continue

            sub_flags = detect_single_image_subtitles([p[0] for p in cropped_pairs])
            runs = split_clean_runs(sub_flags, min_keep) if any(sub_flags) else [(0, len(cropped_pairs))]
            if not runs:
                shutil.rmtree(os.path.join(opt.savedir, savecnt), ignore_errors=True)
                continue

            for r_idx, (s_i, e_i) in enumerate(runs):
                if r_idx == 0:
                    cur_orig, cur_mask = origindir, maskdir
                else:
                    sub_cid = _alloc_clip_id()
                    sub_cnt = '%05d' % sub_cid
                    cur_orig = os.path.join(opt.savedir, sub_cnt, 'origin_image')
                    cur_mask = os.path.join(opt.savedir, sub_cnt, 'mask')
                    util.makedirs(cur_orig)
                    util.makedirs(cur_mask)
                for out_i, src_i in enumerate(range(s_i, e_i), start=1):
                    impro.imwrite(os.path.join(cur_orig, f"{out_i:05d}.jpg"), cropped_pairs[src_i][0])
                    impro.imwrite(os.path.join(cur_mask, f"{out_i:05d}.png"), cropped_pairs[src_i][1])
                clips_created += 1

        with open(os.path.join(done_dir, _video_done_key(videopath)), 'w', encoding='utf-8') as f:
            f.write(f"ok clips={clips_created}\n")

    except Exception as e:
        print(f"Error processing '{vname}': {e}", flush=True)
        util.writelog(os.path.join(opt.savedir, 'opt.txt'), videopath + '\n' + str(e))
    finally:
        if os.path.isdir(worker_tmp):
            shutil.rmtree(worker_tmp, ignore_errors=True)
        dt = time.time() - t0
        with state_lock:
            worker_slots.pop(slot, None)
            slot_pool.append(slot)
            slot_pool.sort()
            completed_videos += 1
            idx_now = completed_videos
            native_sides.extend(local_sides)
        print(f"[{idx_now}/{total_todo}] {vname} ({dt:.1f}s) -> +{clips_created} clip(s)", flush=True)
        _emit_gui_progress(f"last: {vname} (+{clips_created})")


if todo_videos:
    _emit_gui_progress("starting workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one_video, vp) for vp in todo_videos]
        for fut in as_completed(futures):
            fut.result()

if opt.gpu_id != '-1':
    torch.cuda.empty_cache()

if native_sides:
    ns = np.array(native_sides)
    below = 100.0 * (ns < opt.outsize).mean()
    print(f"\nCrop sizes in your source footage: median {int(np.median(ns))}px "
          f"(25th-75th percentile {int(np.percentile(ns, 25))}-{int(np.percentile(ns, 75))}px). "
          f"{below:.0f}% of crops are smaller than --outsize {opt.outsize} and were ENLARGED to reach it.")

n_created = next_clip_id - start_cnt
print(f"\nDone: created {n_created} new clip(s) from {total_todo} video(s) under {opt.savedir}.")