import torch
import os
import json
from . import model_util
from .pix2pix_model import define_G as pix2pix_G
from .pix2pixHD_model import define_G as pix2pixHD_G
# from .video_model import MosaicNet
# from .videoHD_model import MosaicNet as MosaicNet_HD
from .BiSeNet_model import BiSeNet
from .BVDNet import define_G as video_G

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')

def pix2pix(opt):
    # print(opt.model_path,opt.netG)
    if opt.netG == 'HD':
        netG = pix2pixHD_G(3, 3, 64, 'global' ,4)
    else:
        netG = pix2pix_G(3, 3, 64, opt.netG, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG


def style(opt):
    if opt.edges:
        netG = pix2pix_G(1, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=True, init_type='normal', gpu_ids=[])
    else:
        netG = pix2pix_G(3, 3, 64, 'resnet_9blocks', norm='instance',use_dropout=False, init_type='normal', gpu_ids=[])

    #in other to load old pretrain model
    #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/models/base_model.py
    if isinstance(netG, torch.nn.DataParallel):
        netG = netG.module
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(opt.model_path, map_location='cpu')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        model_util.patch_instance_norm_state_dict(state_dict, netG, key.split('.'))
    netG.load_state_dict(state_dict)

    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG

def video(opt):
    netG = video_G(N=2,n_blocks=4,gpu_id=opt.gpu_id)
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(opt.model_path))
    netG = model_util.todevice(netG,opt.gpu_id)
    netG = model_util.to_inference_device(netG, opt)
    netG.eval()
    return netG

def bisenet(opt, type='roi'):
    '''
    type: roi or mosaic
    '''
    target_path = opt.model_path if type == 'roi' else opt.mosaic_position_model_path

    # If 'auto', prefer a .pt/.onnx Lada detector in pretrained_models/mosaic/ if present
    if type == 'mosaic' and (not target_path or target_path == 'auto'):
        mosaic_dir = os.path.join('pretrained_models', 'mosaic')
        if os.path.isdir(mosaic_dir):
            for fname in sorted(os.listdir(mosaic_dir)):
                if fname.lower().endswith(('.pt', '.onnx')) and 'mosaic' in fname.lower():
                    target_path = os.path.join(mosaic_dir, fname)
                    break

    if target_path and target_path.lower().endswith(('.pt', '.onnx')):
        from .yolo_mosaic_detector import YoloMosaicDetector
        return YoloMosaicDetector(
            target_path,
            conf_thresh=0.35,
            gpu_id=getattr(opt, 'gpu_id', '0'),
        )

    net = BiSeNet(num_classes=1, context_path='resnet18', train_flag=False)
    show_paramsnumber(net, 'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net = model_util.todevice(net, opt.gpu_id)
    net = model_util.to_inference_device(net, opt)
    net.eval()
    return net


# ── Size-based model auto-selection (--auto_model) ──────────────────────────
# Maps a detected mosaic block size (see util.mosaic.estimate_mosaic_block_pct,
# expressed as % of the frame's shorter side) to a specific pretrained clean
# model checkpoint. Manifest format is a flat JSON object of
# {"<bucket_pct>": "<path to .pth, relative to the manifest file's folder>"}.
# Selection is nearest-bucket, clamped to the manifest's own min/max buckets.

def load_size_manifest_details(manifest_path):
    """Parse size_manifest.json supporting both legacy {"4": "model.pth"} and rich
    {"4": {"model": "model.pth", "min_pct": 2.2, "max_pct": 4.5}} entries.
    Missing min_pct/max_pct bounds are auto-inferred via Voronoi midpoints.
    Returns dict[float, dict]: {bucket: {"model": rel_path, "min_pct": float, "max_pct": float}}.
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not raw:
        raise ValueError(f"Model manifest '{manifest_path}' is empty")

    parsed = {}
    for k, val in raw.items():
        b = float(k)
        if isinstance(val, dict):
            parsed[b] = {
                "model": str(val.get("model", f"clean_youknow_video_{b:g}pct.pth")),
                "min_pct": float(val["min_pct"]) if val.get("min_pct") is not None else None,
                "max_pct": float(val["max_pct"]) if val.get("max_pct") is not None else None,
            }
        else:
            parsed[b] = {"model": str(val), "min_pct": None, "max_pct": None}

    buckets = sorted(parsed.keys())
    for idx, b in enumerate(buckets):
        if parsed[b]["min_pct"] is None:
            parsed[b]["min_pct"] = round(max(2.0, b - 1.8), 2) if idx == 0 else round(0.5 * (buckets[idx - 1] + b), 2)
        if parsed[b]["max_pct"] is None:
            parsed[b]["max_pct"] = round(max(9.5, b + 2.5), 2) if idx == len(buckets) - 1 else round(0.5 * (b + buckets[idx + 1]), 2)
    return parsed


def load_size_manifest(manifest_path):
    """Backwards-compatible flat map {bucket_float: relative_model_path}."""
    details = load_size_manifest_details(manifest_path)
    return {b: info["model"] for b, info in details.items()}


def select_model_by_pct(pct, manifest_path):
    """Route `pct` to [min_pct, max_pct] first (breaking ties by closest bucket center),
    falling back to nearest bucket center if outside/in a gap."""
    details = load_size_manifest_details(manifest_path)
    buckets = sorted(details.keys())
    in_range = [b for b in buckets if details[b]["min_pct"] <= pct <= details[b]["max_pct"]]
    matched = min(in_range, key=lambda b: abs(b - pct)) if in_range else min(buckets, key=lambda b: abs(b - pct))
    rel_path = details[matched]["model"]
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    abs_path = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(base_dir, rel_path))
    return abs_path, matched