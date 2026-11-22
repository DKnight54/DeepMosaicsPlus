import os
import sys
import json
import random
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cores import Options
from models import loadmodel, runmodel
from models.yolo_mosaic_detector import YoloMosaicDetector


def _load_detector(path, opt, kind, conf):
    if path.lower().endswith(('.pt', '.onnx')):
        return YoloMosaicDetector(path, conf_thresh=conf)
    opt.mosaic_position_model_path = path
    return loadmodel.bisenet(opt, kind)


def _detect_mask(det, img, opt):
    if isinstance(det, YoloMosaicDetector):
        m, _, _, _ = det.detect(img, opt)
    else:
        m = runmodel.get_ROI_position(img, det, opt, keepsize=True)[0]
    if m is None:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return (m > 127).astype(np.uint8)


def _mask_props(m):
    ys, xs = np.where(m > 0)
    if len(xs) < 16:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    area = float(len(xs))
    rect_fill = area / float(bw * bh)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return {'area': area, 'w': float(bw), 'h': float(bh), 'cx': cx, 'cy': cy, 'rect_fill': rect_fill}
def _render_panels(mos_bgr, orig_bgr, mos_m, roi_m, label_txt):
    h, w = mos_bgr.shape[:2]
    p1 = mos_bgr.copy()
    c_mos, _ = cv2.findContours(mos_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, c_mos, -1, (0, 255, 0), 2)

    p2 = orig_bgr.copy()
    c_roi, _ = cv2.findContours(roi_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, c_roi, -1, (255, 220, 0), 2)

    # Pure mask-vs-mask overlay: Red=Mosaic, Green=ROI, Yellow=Intersection
    overlay_only = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_only[mos_m > 0, 2] = 255
    overlay_only[roi_m > 0, 1] = 255

    p4 = orig_bgr.copy()
    cv2.drawContours(p4, c_mos, -1, (0, 0, 255), 2)
    cv2.drawContours(p4, c_roi, -1, (0, 255, 0), 2)

    strip = np.hstack([p1, p2, overlay_only, p4])
    cv2.rectangle(strip, (0, 0), (strip.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(strip, label_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return strip, overlay_only


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--real_pairs_dir', default=os.path.join('datasets', 'real_pair'))
    parser.add_argument('--mosaic_model', required=True, help='YOLO mosaic detector (.pt/.onnx)')
    parser.add_argument('--roi_model', required=True, help='YOLO uncensored ROI detector (.pt/.onnx)')
    parser.add_argument('--out_dir', default=os.path.join('tmp', 'mask_delta_audit'))
    parser.add_argument('--samples_per_clip', type=int, default=6)
    parser.add_argument('--n_vis', type=int, default=100)
    parser.add_argument('--yolo_conf', type=float, default=0.35)
    known, rem = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rem + ['--model_path', known.mosaic_model, '--mode', 'clean', '--netG', 'unet_128']
    opt = Options().getparse(test_flag=True)
    opt.yolo_conf = known.yolo_conf

    det_mos = _load_detector(known.mosaic_model, opt, 'mosaic', known.yolo_conf)
    det_roi = _load_detector(known.roi_model, opt, 'roi', known.yolo_conf)

    clips = []
    for b in sorted(os.listdir(known.real_pairs_dir)):
        bdir = os.path.join(known.real_pairs_dir, b)
        if os.path.isdir(bdir) and b.startswith('bucket_'):
            for c in sorted(os.listdir(bdir)):
                cdir = os.path.join(bdir, c)
                if os.path.isdir(os.path.join(cdir, 'origin_image')):
                    clips.append((b, c, cdir))

    os.makedirs(os.path.join(known.out_dir, 'strips'), exist_ok=True)
    os.makedirs(os.path.join(known.out_dir, 'overlay_only'), exist_ok=True)

    rows, vis_pool, scene_stds = [], [], []
    for bucket, clip, cdir in tqdm(clips, desc='Analyzing real_pair', unit='clip'):
        orig_dir = os.path.join(cdir, 'origin_image')
        mos_dir = os.path.join(cdir, 'mosaic_image')
        names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.jpg'))
        if not names:
            continue
        step = max(1, len(names) // known.samples_per_clip)
        clip_w_rats = []
        prefix = clip.split('_')[0]
        for fn in names[::step][:known.samples_per_clip]:
            oi = cv2.imread(os.path.join(orig_dir, fn))
            mi = cv2.imread(os.path.join(mos_dir, fn))
            if oi is None or mi is None or oi.shape != mi.shape:
                continue
            h, w = oi.shape[:2]
            mm = _detect_mask(det_mos, mi, opt)
            rm = _detect_mask(det_roi, oi, opt)
            mp, rp = _mask_props(mm), _mask_props(rm)
            if mp is None or rp is None:
                continue
            inter = float(np.logical_and(mm > 0, rm > 0).sum())
            union = float(np.logical_or(mm > 0, rm > 0).sum())
            iou = inter / max(1.0, union)
            roi_in_mos = inter / max(1.0, rp['area'])
            w_rat, h_rat = mp['w'] / rp['w'], mp['h'] / rp['h']
            a_rat = mp['area'] / rp['area']
            dx_rat = (mp['cx'] - rp['cx']) / rp['w']
            dy_rat = (mp['cy'] - rp['cy']) / rp['h']
            clip_w_rats.append(w_rat)
            rec = (prefix, bucket, clip, fn, 100 * mp['area'] / (h * w), 100 * rp['area'] / (h * w),
                   a_rat, w_rat, h_rat, iou, roi_in_mos, mp['rect_fill'], rp['rect_fill'], dx_rat, dy_rat)
            rows.append(rec)
            vis_pool.append((mi, oi, mm, rm, rec))
        if len(clip_w_rats) >= 3:
            scene_stds.append(float(np.std(clip_w_rats)))

    # Save 100 evenly-sampled visual comparison strips & standalone overlay-only masks
    random.seed(42)
    chosen = random.sample(vis_pool, min(known.n_vis, len(vis_pool)))
    for idx, (mi, oi, mm, rm, rec) in enumerate(chosen):
        lbl = f"{rec[1]}/{rec[2]}/{rec[3]} | area={rec[6]:.2f}x w={rec[7]:.2f}x h={rec[8]:.2f}x IoU={rec[9]:.2f} cov={rec[10]*100:.0f}%"
        strip, ov = _render_panels(mi, oi, mm, rm, lbl)
        cv2.imwrite(os.path.join(known.out_dir, 'strips', f"strip_{idx:03d}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(known.out_dir, 'overlay_only', f"overlay_{idx:03d}.png"), ov)

    # Write human-readable statistical report
    arr = np.array([r[4:] for r in rows], dtype=np.float32)
    report_path = os.path.join(known.out_dir, 'mask_delta_report.txt')
    names_col = ['mos_crop_pct', 'roi_crop_pct', 'area_ratio(mos/roi)', 'width_ratio', 'height_ratio',
                 'IoU', 'roi_covered_by_mos', 'mos_rect_fill', 'roi_rect_fill', 'dx_norm', 'dy_norm']
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Real-Pair Mosaic vs. YOLO ROI Audit ({len(rows)} paired detections across {len(clips)} clips)\n")
        f.write("=" * 84 + "\n")
        f.write(f"{'Metric':<22} {'p10':>8} {'p25':>8} {'Median':>8} {'p75':>8} {'p90':>8} {'Mean':>8} {'Std':>8}\n")
        for ci, cname in enumerate(names_col):
            col = arr[:, ci]
            p10, p25, p50, p75, p90 = np.percentile(col, [10, 25, 50, 75, 90])
            f.write(f"{cname:<22} {p10:8.3f} {p25:8.3f} {p50:8.3f} {p75:8.3f} {p90:8.3f} {col.mean():8.3f} {col.std():8.3f}\n")
        if scene_stds:
            f.write(f"\nWithin-scene width_ratio std dev: median={np.median(scene_stds):.3f}, p90={np.percentile(scene_stds, 90):.3f}\n")
        f.write("\nPer-Video Prefix Breakdown (Median values):\n")
        f.write(f"{'Prefix':<16} {'N':>5} {'Mos%':>6} {'ROI%':>6} {'AreaX':>6} {'WidthX':>7} {'HeightX':>7} {'IoU':>6} {'Cov%':>6} {'MosRect':>8}\n")
        for pfx in sorted({r[0] for r in rows}):
            sub = np.array([r[4:] for r in rows if r[0] == pfx], dtype=np.float32)
            med = np.median(sub, axis=0)
            f.write(f"{pfx:<16} {len(sub):5d} {med[0]:6.1f} {med[1]:6.1f} {med[2]:6.2f} {med[3]:7.2f} {med[4]:7.2f} {med[5]:6.2f} {med[6]*100:6.1f} {med[7]:8.2f}\n")
    print(f"Saved report -> {report_path}\nSaved {len(chosen)} strips + overlay_only images -> {known.out_dir}")


if __name__ == '__main__':
    main()