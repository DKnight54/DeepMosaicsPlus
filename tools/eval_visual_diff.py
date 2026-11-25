import os
import sys
import json
import glob
import argparse
import subprocess
import cv2
import numpy as np
import torch


_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from models import loadmodel
from models.yolo_mosaic_detector import YoloMosaicDetector
from util import data


def _find_latest_ckpt(ckpt_dir):
    pths = sorted(
        glob.glob(os.path.join(ckpt_dir, "*_G.pth")),
        key=lambda p: int(os.path.basename(p).split("_")[0]) if os.path.basename(p).split("_")[0].isdigit() else -1,
    )
    return pths[-1] if pths else None


def _load_netG(ckpt_path, gpu_id="0"):
    finesize = 512
    meta_json = os.path.splitext(ckpt_path)[0] + ".json"
    if os.path.isfile(meta_json):
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                finesize = int(json.load(f).get("finesize", 512))
        except Exception:
            pass

    class _Opt:
        pass

    opt = _Opt()
    opt.model_path = ckpt_path
    opt.gpu_id = str(gpu_id)
    opt.finesize = finesize
    opt.loadsize = int(round(finesize * 286 / 256))
    opt.netG = "video"
    opt.norm = "instance"
    opt.no_dropout = True
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.verbose = False
    opt.isTrain = False
    return loadmodel.video(opt), finesize


def _center_crop(img, size):
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return img[y0:y0 + size, x0:x0 + size]


def _color_match_border(src_bgr, ref_bgr, mask_bin):
    """Match src_bgr per-channel mean/std to ref_bgr using ONLY unmasked border pixels (mask_bin == 0)."""
    bg = mask_bin == 0
    if int(np.sum(bg)) < 256:
        return src_bgr
    out = src_bgr.astype(np.float32).copy()
    ref_f = ref_bgr.astype(np.float32)
    for c in range(3):
        s_vals = out[:, :, c][bg]
        r_vals = ref_f[:, :, c][bg]
        s_std = float(np.std(s_vals))
        r_std = float(np.std(r_vals))
        if s_std > 1.0:
            out[:, :, c] = (out[:, :, c] - float(np.mean(s_vals))) * (r_std / s_std) + float(np.mean(r_vals))
        else:
            out[:, :, c] = out[:, :, c] - float(np.mean(s_vals)) + float(np.mean(r_vals))
    return np.clip(out, 0, 255).astype(np.uint8)


def _compute_ssim_map(a_gray, b_gray):
    c1, c2 = 6.5025, 58.5225
    a = a_gray.astype(np.float32)
    b = b_gray.astype(np.float32)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sig_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sig_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sig_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2)
    return num / np.maximum(den, 1e-6)


def _make_diag_tiles(orig, mos, pred, mask_bin, prev_err_signed):
    """Return (err_heatmap, sharp_deficit_map, flicker_map, cur_err_signed, metrics_dict)."""
    o_g = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m_bool = mask_bin > 0
    if not np.any(m_bool):
        m_bool = np.ones_like(mask_bin, dtype=bool)

    # Ring around mask boundary (5px inside + 5px outside) to catch seam halos
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    seam_ring = (cv2.dilate(mask_bin, k_ring) > 0) ^ (cv2.erode(mask_bin, k_ring) > 0)

    # 1. Absolute RGB error heatmap (0..40 luma scale -> INFERNO)
    # Inside mask: |pred - orig| (true restoration error)
    # Outside mask: |pred - mos| (pure netG background drift, excluding source codec noise)
    bg_bool = ~m_bool
    abs_err = np.mean(np.abs(pred.astype(np.float32) - orig.astype(np.float32)), axis=2)
    abs_pred_vs_mos = np.mean(np.abs(pred.astype(np.float32) - mos.astype(np.float32)), axis=2)
    abs_mos_vs_orig = np.mean(np.abs(mos.astype(np.float32) - orig.astype(np.float32)), axis=2)
    bg_drift_mae = float(np.mean(abs_pred_vs_mos[bg_bool])) if np.any(bg_bool) else 0.0
    codec_noise_mae = float(np.mean(abs_mos_vs_orig[bg_bool])) if np.any(bg_bool) else 0.0

    combined_err = np.where(m_bool, abs_err, abs_pred_vs_mos)
    err_vis = np.clip(combined_err * (255.0 / 40.0), 0, 255).astype(np.uint8)
    err_heat = cv2.applyColorMap(err_vis, cv2.COLORMAP_INFERNO)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(err_heat, contours, -1, (0, 255, 180), 1)

    # 2. Texture / sharpness deficit: red = over-smoothed plastic, blue = over-sharpened
    gx_o = cv2.Sobel(o_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_o = cv2.Sobel(o_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_o = np.hypot(gx_o, gy_o)
    gx_p = cv2.Sobel(p_g, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_p = np.hypot(gx_p, gy_p)
    deficit = grad_o - grad_p  # >0 means lost texture (red), <0 means extra edges (blue)
    sharp_map = np.full_like(orig, 128, dtype=np.uint8)
    sharp_map[:, :, 2] = np.clip(128.0 + deficit * 1.6, 0, 255).astype(np.uint8)  # Red = blur deficit
    sharp_map[:, :, 0] = np.clip(128.0 - deficit * 1.6, 0, 255).astype(np.uint8)  # Blue = over-sharp
    cv2.drawContours(sharp_map, contours, -1, (220, 220, 220), 1)

    # 3. Temporal residual flicker: |(pred_t - orig_t) - (pred_{t-1} - orig_{t-1})|
    cur_err_signed = p_g - o_g
    if prev_err_signed is None:
        flicker = np.zeros_like(o_g)
    else:
        flicker = np.abs(cur_err_signed - prev_err_signed)
    flick_vis = np.clip(flicker * (255.0 / 25.0), 0, 255).astype(np.uint8)
    flick_heat = cv2.applyColorMap(flick_vis, cv2.COLORMAP_VIRIDIS)
    cv2.drawContours(flick_heat, contours, -1, (255, 255, 255), 1)

    mse = float(np.mean((pred.astype(np.float32)[m_bool] - orig.astype(np.float32)[m_bool]) ** 2))
    psnr = 20.0 * np.log10(255.0 / max(np.sqrt(mse), 1e-4))
    ssim_map = _compute_ssim_map(o_g, p_g)
    ssim_val = float(np.mean(ssim_map[m_bool]))
    sharp_ret = 100.0 * float(np.mean(grad_p[m_bool])) / max(float(np.mean(grad_o[m_bool])), 1e-4)
    seam_mae = float(np.mean(abs_err[seam_ring])) if np.any(seam_ring) else float(np.mean(abs_err[m_bool]))
    flick_mae = float(np.mean(flicker[m_bool]))

    metrics = {
        "psnr": psnr,
        "ssim": ssim_val,
        "sharp_ret": sharp_ret,
        "seam_mae": seam_mae,
        "flick_mae": flick_mae,
        "abs_mae": float(np.mean(abs_err[m_bool])),
        "bg_drift_mae": bg_drift_mae,
        "codec_noise_mae": codec_noise_mae,
    }
    return err_heat, sharp_map, flick_heat, cur_err_signed, metrics

def _label_tile(tile, text, color=(240, 240, 240)):
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 9 + 16, 26), (15, 18, 22), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out


def evaluate_clip(clip_dir, ckpt_path, yolo_path, out_dir, gpu_id="0", max_frames=240, fps=24):
    orig_dir = os.path.join(clip_dir, "origin_image")
    mos_dir = os.path.join(clip_dir, "mosaic_image")
    names = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith(".jpg"))[:max_frames]
    if len(names) < 10:
        raise RuntimeError(f"Too few frames in {clip_dir}: {len(names)}")

    netG, finesize = _load_netG(ckpt_path, gpu_id=gpu_id)
    yolo = YoloMosaicDetector(yolo_path, conf_thresh=0.25)

    class _YOpt:
        pass

    yopt = _YOpt()
    yopt.gpu_id = str(gpu_id)

    orig_frames, mos_frames, mask_frames = [], [], []
    for fn in names:
        o = _center_crop(cv2.imread(os.path.join(orig_dir, fn)), finesize)
        m = _center_crop(cv2.imread(os.path.join(mos_dir, fn)), finesize)
        det_mask, _, _, _ = yolo.detect(m, yopt)
        if det_mask is None:
            det_mask = np.zeros((finesize, finesize), dtype=np.uint8)
        else:
            det_mask = (det_mask > 127).astype(np.uint8) * 255
        m_matched = _color_match_border(m, o, det_mask)
        orig_frames.append(o)
        mos_frames.append(m_matched)
        mask_frames.append(det_mask)

    # Run temporal netG (T=5 sliding window, matching DeepMosaicsPlus video inference)
    n = len(mos_frames)
    device = next(netG.parameters()).device
    pred_frames = []
    prev_out = None
    with data.inference_context(device):
        for i in range(n):
            seq_crops = [mos_frames[min(max(i + dt, 0), n - 1)] for dt in (-2, -1, 0, 1, 2)]
            inp = data.crops_to_stream(seq_crops, finesize, device)
            if prev_out is None:
                prev_out = inp[:, :, 2].clone()
            out_t = netG(inp, prev_out)
            prev_out = out_t
            pred_bgr = data.tensor2im(out_t)
            pred_bgr = _color_match_border(pred_bgr, orig_frames[i], mask_frames[i])
            pred_frames.append(pred_bgr)

    os.makedirs(out_dir, exist_ok=True)
    clip_tag = os.path.basename(os.path.normpath(clip_dir))
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    prefix = os.path.join(out_dir, f"{clip_tag}__{ckpt_tag}")

    grid_frames = []
    safe_triplets = []
    all_metrics = []
    prev_err = None

    for i in range(n):
        err_h, sharp_m, flick_h, prev_err, met = _make_diag_tiles(
            orig_frames[i], mos_frames[i], pred_frames[i], mask_frames[i], prev_err
        )
        all_metrics.append(met)
        safe_triplets.append((err_h, sharp_m, flick_h, i, met))

        t1 = _label_tile(orig_frames[i], f"Original  [f={i:04d}/{n}]")
        t2 = _label_tile(mos_frames[i], "Mosaic Input (Border Color-Matched)")
        t3 = _label_tile(pred_frames[i], f"Restored ({ckpt_tag})")
        t4 = _label_tile(
            err_h,
            f"Err (In=|P-O|, Out=|P-M|) | PSNR={met['psnr']:.1f} BgDrift={met['bg_drift_mae']:.1f} Codec={met['codec_noise_mae']:.1f}",
            (120, 240, 255),
        )
        t5 = _label_tile(sharp_m, f"Texture Deficit (Red=Blur) | SharpRet={met['sharp_ret']:.0f}%", (180, 210, 255))
        t6 = _label_tile(flick_h, f"Temporal Flicker | SeamMAE={met['seam_mae']:.1f} Flick={met['flick_mae']:.1f}", (180, 255, 190))

        frame_3x2 = np.vstack([np.hstack([t1, t2, t3]), np.hstack([t4, t5, t6])])
        grid_frames.append(frame_3x2)

# 1. Export 1536x1024 MP4 (100% native 512x512 tiles, fits 1080p monitor)
    mp4_path = f"{prefix}_3x2_preview.mp4"
    gh, gw = grid_frames[0].shape[:2]
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{gw}x{gh}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", mp4_path,
        ],
        stdin=subprocess.PIPE,
    )
    for gf in grid_frames:
        ff.stdin.write(gf.tobytes())
    ff.stdin.close()
    ff.wait()

    # 2. Export human 3x2 contact sheet (4 evenly spaced timestamps)
    idxs_4 = np.linspace(1, n - 1, min(4, n - 1), dtype=int)
    cv2.imwrite(f"{prefix}_human_sheet.png", np.vstack([grid_frames[idx] for idx in idxs_4]))

    # 3. Export chat-safe 6-timestamp diagnostic-only contact sheet (no RGB imagery)
    idxs_6 = np.linspace(1, n - 1, min(6, n - 1), dtype=int)
    safe_cols = []
    for idx in idxs_6:
        eh, sm, fh, f_i, met = safe_triplets[idx]
        eh_s = _label_tile(cv2.resize(eh, (256, 256)), f"f={f_i} PSNR={met['psnr']:.1f} SSIM={met['ssim']:.3f}")
        sm_s = _label_tile(cv2.resize(sm, (256, 256)), f"SharpRet={met['sharp_ret']:.0f}% Seam={met['seam_mae']:.1f}")
        fh_s = _label_tile(cv2.resize(fh, (256, 256)), f"Flicker={met['flick_mae']:.2f}")
        safe_cols.append(np.vstack([eh_s, sm_s, fh_s]))

    summary = {k: float(np.mean([m[k] for m in all_metrics[1:]])) for k in all_metrics[0]}
    banner = np.full((44, 256 * len(safe_cols), 3), 18, dtype=np.uint8)
    hdr = (
        f"SAFE DIFF AUDIT | {clip_tag} | {ckpt_tag} | "
        f"PSNR={summary['psnr']:.2f}dB | SSIM={summary['ssim']:.4f} | "
        f"SharpRet={summary['sharp_ret']:.1f}% | InMAE={summary['abs_mae']:.2f} | "
        f"BgDrift={summary['bg_drift_mae']:.2f} | Codec={summary['codec_noise_mae']:.2f} | Flick={summary['flick_mae']:.2f}"
    )
    cv2.putText(banner, hdr, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 230, 160), 1, cv2.LINE_AA)
    safe_sheet_path = f"{prefix}_safe_diff_sheet.png"
    cv2.imwrite(safe_sheet_path, np.vstack([banner, np.hstack(safe_cols)]))

    print(f"Saved MP4 preview     -> {mp4_path}")
    print(f"Saved Human 3x2 sheet -> {prefix}_human_sheet.png")
    print(f"Saved Chat-Safe sheet -> {safe_sheet_path}")
    print("Summary Metrics:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate 1536x1024 3x2 visual diff MP4 + chat-safe diagnostic contact sheet.")
    p.add_argument("--clip_dir", required=True, help="Path to a real_pair clip folder containing origin_image/ and mosaic_image/")
    p.add_argument("--ckpt", default="", help="Path to *_G.pth (if omitted, uses latest in --ckpt_dir)")
    p.add_argument("--ckpt_dir", default="checkpoints/clean_5pct", help="Checkpoint folder used when --ckpt is omitted")
    p.add_argument("--yolo_model", default="pretrained_models/mosaic/lada_mosaic_detection_model_v4_accurate.pt")
    p.add_argument("--out_dir", default="datasets/Holdout/eval_diffs")
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--max_frames", type=int, default=240, help="Max frames to render (240f = 10s at 24fps)")
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    ckpt = args.ckpt or _find_latest_ckpt(args.ckpt_dir)
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"No *_G.pth found (checked --ckpt='{args.ckpt}' and --ckpt_dir='{args.ckpt_dir}').")

    evaluate_clip(
        clip_dir=args.clip_dir,
        ckpt_path=ckpt,
        yolo_path=args.yolo_model,
        out_dir=args.out_dir,
        gpu_id=args.gpu_id,
        max_frames=args.max_frames,
        fps=args.fps,
    )