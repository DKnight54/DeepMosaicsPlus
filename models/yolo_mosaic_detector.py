import cv2
import numpy as np
from ultralytics import YOLO
import util.image_processing as impro


class YoloMosaicDetector:
    """Drop-in replacement for BiSeNet mosaic_position.pth using Lada V4 YOLO-Seg."""

    def __init__(self, model_path, conf_thresh=0.35, device=None, gpu_id="0"):
        self.model = YOLO(model_path, task="segment")
        self.conf_thresh = conf_thresh
        if device is not None:
            self.device = device
        elif str(gpu_id) == "-1":
            self.device = "cpu"
        elif model_path.lower().endswith(".onnx"):
            self.device = "dml"
        else:
            try:
                self.device = int(str(gpu_id).split(",")[0])
            except ValueError:
                self.device = 0
        
    def _postprocess_result(self, res, h, w, opt):
        if res.masks is None or len(res.masks) == 0:
            return None, 0, 0, 0

        masks_data = res.masks.data.cpu().numpy()
        combined = np.any(masks_data > 0.5, axis=0).astype(np.uint8) * 255

        if combined.shape[:2] != (h, w):
            combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

        # Expand mask outward (default ~1.5% of shorter side, or override via --yolo_pad)
        pad_px = getattr(opt, "yolo_pad", None) if opt is not None else None
        if pad_px is None:
            pad_px = max(3, int(min(h, w) * 0.015))
        if pad_px > 0:
            ksize = pad_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            combined = cv2.dilate(combined, kernel, iterations=1)

        # Filter tiny stray blobs (matching mask_to_position's min_mosaic_area)
        min_area = getattr(opt, "min_mosaic_area", 300) if opt is not None else 300
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    combined[labels == i] = 0

        if opt is not None and not getattr(opt, "all_mosaic_area", True):
            combined = impro.find_mostlikely_ROI(combined)

        ys, xs = np.where(combined > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, 0, 0, 0

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_size = max(x2 - x1, y2 - y1) // 2 + 1
        return combined, cx, cy, half_size

    def _get_conf(self, opt):
        if opt is not None and hasattr(opt, "yolo_conf"):
            return float(opt.yolo_conf)
        if opt is not None and hasattr(opt, "mask_threshold"):
            return float(np.clip(opt.mask_threshold / 137.0, 0.10, 0.90))
        return self.conf_thresh

    def detect(self, frame, opt=None):
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self._get_conf(opt),
            iou=0.45,
            max_det=20,
            imgsz=640,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None, 0, 0, 0
        return self._postprocess_result(results[0], h, w, opt)

    def detect_batch(self, frames, opt=None):
        results = self.model.predict(
            source=frames,
            conf=self._get_conf(opt),
            iou=0.45,
            max_det=20,
            imgsz=640,
            batch=len(frames),
            device=self.device,
            verbose=False,
        )
        out = []
        for frame, res in zip(frames, results):
            h, w = frame.shape[:2]
            out.append(self._postprocess_result(res, h, w, opt))
        return out