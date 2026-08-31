"""批量 GPU 合成引擎（单进程 torch；替代多进程 multi_rs 的卡死面）。

- 像素级（背景裁剪/增强、目标 warp/粘贴、噪声）批量张量化；
- 随机采样、放置拒绝、标注组装保留 CPU（随机语义与 realscene.py 一致）；
- 兼容 torch>=2.4 交集 API（服务器 torch 2.4.1+CUDA12.8；本机 2.13+cu130 亦可）；
- 所有算子同时支持 CPU 后端（device='cpu'）便于调试与 API 预检。

用法（入口见 gpu_synth.py）：
    python realscene/gpu_synth.py --backend gpu --epochs 200 --output ./seven/train_output
"""
import os
import sys
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import load_config

IMG_EXT = (".png", ".jpg", ".jpeg")


# ============================================================================
# 配置提取（与 realscene.py 顶部同义）
# ============================================================================
def extract_synth_cfg(cfg):
    _S = cfg.synth
    out = {
        "BASE_SIZE": tuple(_S.base_size),            # (W, H)
        "NUM_ROUNDS": _S.num_rounds,
        "EPOCHS": _S.epochs,
        "NUM_TARGETS_PER_IMAGE": _S.num_targets_per_image,
        "BACKGROUND_RATIO": _S.get("background_ratio", 0.0),
        "TARGET_REPEAT": max(1, int(_S.get("target_repeat", 1))),
        "MIN_CROP_RATIO": _S.min_crop_ratio,
        "MAX_CROP_RATIO": _S.max_crop_ratio,
        "MIN_TARGET_RATIO": _S.min_target_ratio,
        "MAX_TARGET_RATIO": _S.max_target_ratio,
        "MAX_TARGET_FAILURE": _S.max_target_failure,
        "MAX_OVERLAP_ATTEMPTS": _S.max_overlap_attempts,
        "TO_BORDER": float(_S.to_border),
        "JPEG_QUALITY": int(_S.get("jpeg_quality", 95)),
        "AUG_MENUS": dict(_S.get("aug", {})),
        "TARGET_TYPES": dict(_S.target_types),
        "BACKEND": _S.get("backend", "cpu"),
        "GPU_BATCH": int(_S.get("gpu_batch", 32)),
        "JPEG_WORKERS": int(_S.get("gpu_jpeg_workers", 4)),
        # torch<2.5 的 grid_sample 存在边界裁剪残缺：true=warp 走 cv2(CPU, 稳定/版本无关)
        "GPU_WARP_CV": bool(_S.get("gpu_warp_cv", True)),
    }
    tt = out["TARGET_TYPES"]
    out["TARGET_ROI"] = {t: (list(s["roi"]) if s.get("roi") else None) for t, s in tt.items()}
    out["TARGET_SCALE"] = {t: tuple(s.get("scale", [0.5, 1.0])) for t, s in tt.items()}
    out["TARGET_ROTATE"] = {t: tuple(s.get("rotate", [-45, 45])) for t, s in tt.items()}
    out["TARGET_PERSPECTIVE"] = {t: (tuple(s["perspective"]) if s.get("perspective") else None)
                                 for t, s in tt.items()}
    out["TARGET_FEATHER"] = {t: int(s.get("feather", 0)) for t, s in tt.items()}
    out["TARGET_CROP_TRANSPARENT"] = {t: bool(s.get("crop_transparent", True)) for t, s in tt.items()}
    out["CLASSES"] = list(cfg.classes)
    out["REAL_IMAGES_DIR"] = cfg.paths.backgrounds
    out["OUTPUT_DIR"] = cfg.paths.synth_output
    return out


def pick_device(backend, verbose=True):
    """按 backend 选择设备：gpu→cuda:0（不可用报错）；auto→有则 gpu 否则 cpu；cpu→cpu。"""
    if backend == "cpu":
        if verbose:
            print("GPU 引擎：backend=cpu（tensor 在 CPU 上，仅调试用）")
        return "cpu"
    if torch.cuda.is_available():
        if verbose:
            print(f"GPU 引擎：cuda:0 {torch.cuda.get_device_name(0)} "
                  f"(torch {torch.__version__}, cuda_build {torch.version.cuda})")
        return "cuda:0"
    if backend == "gpu":
        print("❌ GPU 引擎：--backend gpu 但 torch 无可用 CUDA。改用 --backend cpu 或回退 realscene.py。")
        raise SystemExit(1)
    if verbose:
        print("GPU 引擎：无 CUDA，回退 CPU（--backend auto）")
    return "cpu"


# ============================================================================
# 目标 / 背景加载（与 realscene.py 同语义）
# ============================================================================

def _crop_transparent(img, room=0):
    """裁掉 alpha=0 的透明边距（与 realscene.py 一致）。"""
    if img.shape[2] != 4:
        return img
    a = img[:, :, 3]
    if a.all() or not a.any():
        return img
    ys, xs = np.where(a > 0)
    y0, y1 = max(0, ys.min() - room), min(a.shape[0], ys.max() + 1 + room)
    x0, x1 = max(0, xs.min() - room), min(a.shape[1], xs.max() + 1 + room)
    return img[y0:y1, x0:x1]


def load_target_images(cfg, crop_map):
    """加载所有类型目标（支持 文件夹类/单图类；类名=顶层名字）→ list[(img HxWx4, class_id, idx, ttype)]。"""
    target_images = []
    dir_type_map = []
    for ttype, spec in cfg["TARGET_TYPES"].items():
        d = spec.get("dir")
        if d and os.path.isdir(d):
            dir_type_map.append((d, ttype))
        else:
            print(f"跳过 target_type '{ttype}'：目录不存在 {d}")

    for target_dir, ttype in dir_type_map:
        dir_classes, file_classes = [], []
        for entry in sorted(os.listdir(target_dir)):
            full = os.path.join(target_dir, entry)
            if os.path.isdir(full):
                dir_classes.append(entry)
            elif entry.lower().endswith(IMG_EXT):
                file_classes.append(os.path.splitext(entry)[0])

        conflicts = [dc for dc in dir_classes
                     if any(os.path.exists(os.path.join(target_dir, dc + e)) for e in IMG_EXT)]
        if conflicts:
            print(f"❌ '{target_dir}' 下类名冲突（既是文件夹又是单图）：{conflicts}")
            print("   请只保留一种布局：要么删除文件夹，要么删除同名图片文件。")
            sys.exit(1)

        def add(path, category):
            try:
                img = np.array(Image.open(path).convert("RGBA"))
                if crop_map.get(ttype, True):
                    img = _crop_transparent(img)
                target_images.append((img, cfg["CLASSES"].index(category),
                                      len(target_images), ttype))
            except Exception as e:
                print(f"加载失败：{path}，原因：{e}")

        for dc in dir_classes:
            class_dir = os.path.join(target_dir, dc)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(IMG_EXT):
                        add(os.path.join(root, file), dc)
        for fc in file_classes:
            if fc not in cfg["CLASSES"]:      # 目录里的无关文件(如 *_overlay.jpg)跳过而非报错
                print(f"跳过目标文件(类名不在 classes)：{fc}")
                continue
            for e in IMG_EXT:
                p = os.path.join(target_dir, fc + e)
                if os.path.exists(p):
                    add(p, fc)
    return target_images


def load_background_arrays(bg_dir, verbose=True):
    """预解码所有背景图 → list[uint8 HxWx3]。无图返回 []。"""
    if not os.path.isdir(bg_dir):
        return []
    arrs = []
    for f in sorted(os.listdir(bg_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            p = cv2.imread(os.path.join(bg_dir, f))
            if p is None:
                continue
            arrs.append(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
    if verbose:
        print(f"GPU 引擎：预解码背景 {len(arrs)} 张")
    return arrs


# ============================================================================
# torch 批量增强算子（输入输出 float32 [B,N,C,H,W] 或 [N,C,H,W]）
# ============================================================================
def t_brightness(img, factors):
    return img * factors.view(-1, 1, 1, 1)


def t_contrast(img, factors):
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    return (img - mean) * factors.view(-1, 1, 1, 1) + mean


def t_color(img, factors):
    # cv2 BT.601 灰色近似（uint8 量化差异在 P4 对拍允许范围）
    rgb_w = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype).to(img).view(1, 3, 1, 1)
    gray = (img * rgb_w).sum(dim=1, keepdim=True)
    f = factors.view(-1, 1, 1, 1)
    return img * f + gray * (1.0 - f)


def t_hsv(img, deltas):
    """HSV 扰动（float 归一化空间；H 0-1 环、S/V 0-1）。deltas: [N,3] (±幅度)。"""
    x = img.clamp(0, 255) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = x.max(dim=1).values
    mn = x.min(dim=1).values
    d = (mx - mn).clamp(min=1e-8)
    h = torch.zeros_like(mx)
    h = torch.where((mx == r) & (d > 0), (((g - b) / d) % 6.0), h)
    h = torch.where((mx == g) & (d > 0), ((b - r) / d + 2.0), h)
    h = torch.where((mx == b) & (d > 0), ((r - g) / d + 4.0), h)
    h = h * 60.0 / 360.0                       # 0-1
    s = torch.where(d > 0, d / mx.clamp(min=1e-8), torch.zeros_like(mx))
    v = mx
    dh, ds, dv = deltas[:, 0], deltas[:, 1], deltas[:, 2]
    h = (h + dh.view(-1, 1, 1)) % 1.0
    s = (s + ds.view(-1, 1, 1)).clamp(0, 1)
    v = (v + dv.view(-1, 1, 1)).clamp(0, 1)
    hi = (h * 6.0).floor() % 6
    f = h * 6.0 - (h * 6.0).floor()
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    idx = hi.long().unsqueeze(1)
    rr = torch.stack([v, t, p, p, q, v], dim=1).gather(1, idx)[:, 0]
    gg = torch.stack([q, v, v, p, p, t], dim=1).gather(1, idx)[:, 0]
    bb = torch.stack([p, p, t, v, v, q], dim=1).gather(1, idx)[:, 0]
    return torch.stack([rr, gg, bb], dim=1) * 255.0


def t_cutout(img, rects, noise, rng=None):
    """逐 item 矩形填随机噪声（numpy rng 生成，跨进程确定）。"""
    out = img
    for (i, (y0, y1, x0, x1)), nz in zip(enumerate(rects), noise):
        n = rng.normal(0, 255.0 * float(nz), out[i, :, y0:y1, x0:x1].shape)
        out[i, :, y0:y1, x0:x1] = torch.from_numpy(np.clip(n, 0, 255)).to(out.device)
    return out


def t_flip(img):
    return img.flip(-1)


def t_gaussian(img, sigmas, rng=None):
    """批量高斯噪声（numpy rng 生成，跨进程确定）。"""
    sigmas_np = sigmas.cpu().numpy()
    noise = rng.normal(0, 1.0, img.shape).astype(np.float32) * sigmas_np.reshape(-1, 1, 1, 1)
    return torch.clamp(img + torch.from_numpy(noise).to(img.device), 0, 255)


def t_salt_pepper(img, amounts, rng=None):
    out = img
    for i, a in enumerate(amounts):
        n = int(np.ceil(float(a) * out[i].numel() * 0.5))
        C, H, W = out[i].shape
        for c in range(3):
            coords0 = (rng.randint(0, H, n), rng.randint(0, W, n))
            out[i, c][coords0[0], coords0[1]] = 0.0
            coords1 = (rng.randint(0, H, n), rng.randint(0, W, n))
            out[i, c][coords1[0], coords1[1]] = 255.0
    return out


def t_poisson(img, intensities, rng=None):
    batch = img.clamp(0, 255) * intensities.view(-1, 1, 1, 1)
    noise = rng.poisson(batch.clamp(0, 255).cpu().numpy()).astype(np.float32)
    return torch.clamp(img + torch.from_numpy(noise).to(img.device), 0, 255)


_GAUSS7 = None


def _gauss7_kernel(device, dtype):
    """cv2 GaussianBlur(σ=1.0) 等价的 7x7 核（与 CPU aug_sharpness 对齐）。"""
    global _GAUSS7
    if _GAUSS7 is None:
        k = cv2.getGaussianKernel(7, 1.0)
        _GAUSS7 = np.outer(k, k).astype(np.float32)
    return torch.tensor(_GAUSS7, device=device, dtype=dtype)


def t_sharpness(img, amount, rng=None):
    """锐化（unsharp mask）：对齐 CPU aug_sharpness —— (1+a)*img - a*GaussianBlur(σ=1)。"""
    k = _gauss7_kernel(img.device, img.dtype).view(1, 1, 7, 7)
    k = k.expand(img.shape[1], 1, 7, 7)
    blurred = F.conv2d(img, k, padding=3, groups=img.shape[1])
    a = amount.view(-1, 1, 1, 1)
    return torch.clamp((1.0 + a) * img - a * blurred, 0, 255)


def t_geometric(images, distortions, rng=None):
    """批量背景透视扰动：对齐 CPU aug_geometric（四角**独立**向内收缩、CUBIC、BORDER_REPLICATE）。
    作用在 base 层（贴目标前），不影响标注；输出尺寸不变。"""
    B, C, H, W = images.shape
    if images.device.type == "cpu":
        out = images.clone()
        for i, d in enumerate(distortions):
            hm = _persp_shrink_hm(W, H, d, rng)
            img8 = images[i].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            warped = cv2.warpPerspective(img8, hm, (W, H), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            out[i] = torch.from_numpy(warped).permute(2, 0, 1).float()
        return out
    # GPU：批量一次 grid_sample（bicubic 要求 align_corners=True；border 复刻 BORDER_REPLICATE）
    hms_inv = np.stack([np.linalg.inv(_persp_shrink_hm(W, H, d, rng)) for d in distortions])
    m = torch.tensor(hms_inv, dtype=torch.float32, device=images.device)
    px = torch.linspace(0, W - 1, W, device=images.device)
    py = torch.linspace(0, H - 1, H, device=images.device)
    gx, gy = torch.meshgrid(px, py, indexing="xy")
    grid = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).view(1, -1, 3)
    src_pts = grid.expand(B, -1, 3) @ m.transpose(1, 2)        # [B,HW,3]
    wx = src_pts[..., 2].clamp(min=1e-8)
    sx = (src_pts[..., :2] / wx.unsqueeze(-1)).view(B, H, W, 2).contiguous()
    sx[..., 0] = sx[..., 0] * 2.0 / (W - 1) - 1.0
    sx[..., 1] = sx[..., 1] * 2.0 / (H - 1) - 1.0
    return torch.clamp(F.grid_sample(images, sx, mode="bicubic", padding_mode="border",
                                     align_corners=True), 0, 255)


def _persp_shrink_hm(W, H, d, rng):
    """四角向内收缩的透视矩阵（src→dst；四角偏移各自独立随机，与 CPU 版逐角调用一致）。
    rng 为 numpy RandomState（random.randint 闭区间 vs rng.randint 半开：+1 对齐）。"""
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([
        [rng.randint(0, int(W * d) + 1), rng.randint(0, int(H * d) + 1)],
        [W - rng.randint(0, int(W * d) + 1), rng.randint(0, int(H * d) + 1)],
        [W - rng.randint(0, int(W * d) + 1), H - rng.randint(0, int(H * d) + 1)],
        [rng.randint(0, int(W * d) + 1), H - rng.randint(0, int(H * d) + 1)],
    ])
    return cv2.getPerspectiveTransform(src, dst)


def t_ink_reflection(images, rng=None):
    """批量油墨反光：随机椭圆高光斑（对齐 CPU aug_ink_reflection；无参数项）。"""
    B, C, H, W = images.shape
    out = images.clone()
    for i in range(B):
        cx = rng.randint(int(W * 0.2), int(W * 0.8) + 1)
        cy = rng.randint(int(H * 0.2), int(H * 0.8) + 1)
        rx, ry = rng.randint(10, 41), rng.randint(10, 41)
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), rng.randint(0, 361), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = np.clip(mask * rng.uniform(0.5, 1.5), 0, 1)
        out[i] = torch.clamp(images[i] + torch.from_numpy(mask[None, ...] * 255.0).to(images.device),
                             0, 255)
    return out


# ---- GPU 几何：矩阵/网格（与 cv2 语义对齐） ----
def _rotate_expand_dims(w, h, angle):
    """复刻 _rotate_rgba 的 expand 尺寸与 cos/sin 浮点钳位。"""
    cos_a, sin_a = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    if cos_a < 1e-10: cos_a = 0.0
    if sin_a < 1e-10: sin_a = 0.0
    if abs(cos_a - 1.0) < 1e-10: cos_a = 1.0
    if abs(sin_a - 1.0) < 1e-10: sin_a = 1.0
    return int(np.ceil(w * cos_a + h * sin_a)), int(np.ceil(w * sin_a + h * cos_a))


def _rotation_matrix(w, h, angle, max_w, max_h):
    """与 cv2 warpAffine 语义一致的 3x3（含 expand 平移）。"""
    m2 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m2[0, 2] += (max_w - w) / 2.0
    m2[1, 2] += (max_h - h) / 2.0
    m3 = np.eye(3)
    m3[:2] = m2
    return m3


def _warp_rgba_tensor(rgb, alpha, hm, out_h, out_w):
    """cv2.warpPerspective 等价：hm 为**正向（src→dst）**3x3，内部用 inv(H) 采样
    （实测方向修正：直接用 H 会在所有 device 上产生反向/放大的错误贴图）。
    CPU 后端走 **cv2 路径**（与 realscene.py 同款实现，兼容 torch 2.4.x 的
    grid_sample 差异——服务器 CPU 版本曾出现目标残缺）；GPU 用 grid_sample。
    rgb [1,3,H,W]、alpha [1,1,H,W] float [0,255]。"""
    if rgb.device.type == "cpu":
        return _warp_rgba_cv(rgb, alpha, hm, out_h, out_w)
    # 输出网格像素坐标 → 采样源坐标（正向 hm，与 cv2.warpPerspective 的 M 语义一致）
    hm_inv = np.linalg.inv(hm)                        # 采样矩阵 = H⁻¹（方向修正）
    m = torch.tensor(hm_inv, dtype=torch.float32, device=rgb.device)
    px = (torch.linspace(0, rgb.shape[-1] - 1, out_w, device=rgb.device)
          * (out_w - 1) / max(1, rgb.shape[-1] - 1))
    py = (torch.linspace(0, rgb.shape[-2] - 1, out_h, device=rgb.device)
          * (out_h - 1) / max(1, rgb.shape[-2] - 1))
    gx, gy = torch.meshgrid(px, py, indexing="xy")          # (H,W)
    grid = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).reshape(-1, 3)
    src_pts = grid @ m.T
    wx = src_pts[:, 2].clamp(min=1e-8)
    sx = (src_pts[:, :2] / wx.unsqueeze(1)).reshape(1, out_h, out_w, 2).contiguous()
    # 归一化到 grid_sample 帧（源尺寸 >1 时 2x-1）。
    # 严禁 clamp 到 [-1,1]: 越界坐标被 grid_sample 按 padding=zeros 采 0（=透明），
    # 若 clamp 回 1.0 会"拽回边界采样",把本应透明的区域变成边界像素的拉丝/拖尾
    # （diag_warp --gpu 实测: 多出 15204 px, bbox 撑满画布）。
    sx_n = sx.clone()
    sx_n[..., 0] = sx[..., 0] * 2.0 / (rgb.shape[-1] - 1) - 1.0
    sx_n[..., 1] = sx[..., 1] * 2.0 / (rgb.shape[-2] - 1) - 1.0
    rgb_o = F.grid_sample(rgb, sx_n, mode="bilinear", padding_mode="zeros", align_corners=False)
    a_o = F.grid_sample(alpha, sx_n, mode="nearest", padding_mode="zeros", align_corners=False)
    return rgb_o, a_o


def _warp_rgba_cv(rgb, alpha, hm, out_h, out_w):
    """CPU 版 warp：cv2.warpPerspective（H 为正向 src→dst；RGB 双线性、alpha 最近邻）。"""
    rgb8 = rgb[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    a8 = alpha[0, 0].clamp(0, 255).byte().cpu().numpy()
    rgb_o = cv2.warpPerspective(rgb8, hm, (out_w, out_h),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    a_o = cv2.warpPerspective(a8, hm, (out_w, out_h),
                              flags=cv2.INTER_NEAREST, borderValue=0)
    rgb_t = torch.from_numpy(rgb_o).permute(2, 0, 1).float().unsqueeze(0)
    a_t = torch.from_numpy(a_o).float().unsqueeze(0).unsqueeze(0)
    return rgb_t.to(rgb.device), a_t.to(rgb.device)


def _alpha_bbox_gpu(alpha):
    """alpha>0 包围盒: (x1,y1,x2,y2) 或 None（回传 CPU）。输入可为 (H,W)/(1,H,W)。"""
    a = alpha if alpha.dim() == 2 else alpha[0]
    ys, xs = torch.nonzero(a > 0, as_tuple=True)
    if xs.numel() == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _alpha_bboxes_batch(alpha_batch):
    """alpha_batch [N,H,W] 或 [N,1,H,W] → list[bbox|None]。全 GPU 向量化，仅一次 .cpu() 同步。"""
    a = alpha_batch[:, 0] if alpha_batch.dim() == 4 else alpha_batch
    n, h, w = a.shape
    m = a > 0
    xany = m.any(dim=1)                       # [N,W]
    yany = m.any(dim=2)                       # [N,H]
    xx = torch.arange(w, device=a.device).float()
    yy = torch.arange(h, device=a.device).float()
    xmin = torch.where(xany, xx[None, :], torch.tensor(1e9, device=a.device)).min(dim=1)[0]
    xmax = torch.where(xany, xx[None, :], torch.tensor(-1e9, device=a.device)).max(dim=1)[0]
    ymin = torch.where(yany, yy[None, :], torch.tensor(1e9, device=a.device)).min(dim=1)[0]
    ymax = torch.where(yany, yy[None, :], torch.tensor(-1e9, device=a.device)).max(dim=1)[0]
    has = m.any(dim=(1, 2)).cpu().numpy()
    xmin_n = xmin.cpu().numpy(); xmax_n = xmax.cpu().numpy()
    ymin_n = ymin.cpu().numpy(); ymax_n = ymax.cpu().numpy()
    out = []
    for i in range(n):
        if not has[i]:
            out.append(None)
        else:
            out.append((int(xmin_n[i]), int(ymin_n[i]), int(xmax_n[i]) + 1, int(ymax_n[i]) + 1))
    return out


def _feather_alpha_tensor(h, w, radius, device):
    """复刻 _build_edge_feather_alpha（radius<=0 → 全 255）。"""
    if radius <= 0:
        return torch.ones((1, 1, h, w), device=device) * 255.0
    yy = torch.arange(h, device=device).float()
    xx = torch.arange(w, device=device).float()
    dist = torch.minimum(
        torch.minimum(xx[None, :], (w - 1.0) - xx[None, :]),
        torch.minimum(yy[:, None], (h - 1.0) - yy[:, None]))    # (h,w)
    alpha = torch.clamp((dist + 1.0) / float(radius), 0.0, 1.0) * 255.0
    return alpha.unsqueeze(0).unsqueeze(0)


# ============================================================================
# 批量合成器
# ============================================================================
class BatchSynthesizer:
    """合成编排：采样（CPU 线程）→ 批量张量合成（GPU）→ 并行 JPEG 保存。"""

    def __init__(self, cfg_data=None, device="cpu", batch_size=None, jpeg_workers=None, seed=None):
        self.sc = cfg_data or extract_synth_cfg(load_config())
        self.device = device
        self.batch_size = batch_size or self.sc["GPU_BATCH"]
        self.jpeg_workers = jpeg_workers or self.sc["JPEG_WORKERS"]
        self._rng = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            # 噪声类增强统一走 numpy RandomState（同 seed 跨进程确定）
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()   # 无 seed：随机初始化
        self._lock = threading.Lock()

        # 载荷（P0：加载并检查）
        self.bg_arrays = load_background_arrays(self.sc["REAL_IMAGES_DIR"])
        self.target_images_np = load_target_images(self.sc, self.sc["TARGET_CROP_TRANSPARENT"])
        self.target_dev_tensors = []      # list[torch.Tensor RGBA float32 on device]（P2 起使用）

    # ---- P0: 统计并入口占位（P1 起填充 run 流程）----
    def summary(self):
        return {
            "device": self.device,
            "base_size": self.sc["BASE_SIZE"],
            "batch_size": self.batch_size,
            "jpeg_workers": self.jpeg_workers,
            "classes": self.sc["CLASSES"],
            "targets": len(self.target_images_np),
            "backgrounds": len(self.bg_arrays),
            "backend_cfg": self.sc["BACKEND"],
        }

    # ---- P1: 批量背景构建/保存/主循环 ----
    def _resolve_menu(self, menu_def):
        """复刻 apply_aug_menu 的 num/extra_prob/prob 采样（range→range_），返回已落实 op 列表。"""
        items = menu_def.get("menu", [])
        chosen = list(items)
        if "num" in menu_def:
            n = random.randint(*menu_def["num"])
            chosen = random.sample(items, min(n, len(items)))
            if (random.random() < menu_def.get("extra_prob", 0.0)
                    and len(chosen) < len(items)):
                chosen.append(random.choice([i for i in items if i not in chosen]))
        ops = []
        for item in chosen:
            if random.random() < item.get("prob", 1.0):
                tp = item["type"]
                params = {k: v for k, v in item.items() if k not in ("type", "prob")}
                # 参数在采样时落实具体值（与原版 uniform 语义一致）
                val = None
                if "range_" in params or "range" in params:
                    r = params.pop("range_", params.pop("range", None))
                    val = random.uniform(*r)
                elif "amount" in params:
                    val = random.uniform(*params.pop("amount"))
                elif "var" in params:
                    val = random.uniform(*params.pop("var"))
                elif "intensity" in params:
                    val = float(params.pop("intensity"))
                elif "delta" in params:
                    val = float(params.pop("delta"))
                if val is not None:
                    params["value"] = val
                if tp == "cutout":
                    params["w"] = random.randint(*params.pop("mask_size", (10, 100)))
                    params["n"] = random.randint(*params.pop("num_masks", (1, 3)))
                ops.append({"type": tp, "params": params})
        return ops

    def _sample_backgrounds(self, epoch, n):
        """采样 n 个背景计划（P1：crop 参数 + base 菜单 op 序列；slots 暂空）。"""
        plans = []
        for i in range(n):
            bg = random.choice(self.bg_idx)
            ow, oh = self.bg_w[bg], self.bg_h[bg]
            if ow < 2 or oh < 2:                      # 同 random_crop 的兜底
                plans.append({"bg": bg, "crop": None, "menu": [], "slots": [],
                              "round": self._round_seq[0], "empty": True})
                continue
            cw = random.randint(max(1, int(ow * self.sc["MIN_CROP_RATIO"])),
                                max(1, int(ow * self.sc["MAX_CROP_RATIO"])))
            ch = int(cw * 3 / 4); ch = min(ch, oh)
            x = random.randint(0, max(1, ow - cw)); y = random.randint(0, max(1, oh - ch))
            plans.append({"bg": bg, "crop": (x, y, cw, ch),
                          "menu": self._resolve_menu(self.sc["AUG_MENUS"].get("base", {"menu": []})),
                          "slots": [], "round": self._round_seq[0], "empty": True})
            self._round_seq[0] += 1
        return plans

    def build_backgrounds(self, plans):
        """[CPU numpy crop] → (B,3,H,W) tensor on device；空镜/兜底同 random_crop。"""
        W, H = self.sc["BASE_SIZE"]
        imgs = []
        for pl in plans:
            if pl["crop"] is None:
                arr = np.full((H, W, 3), 128, np.uint8)
            else:
                x, y, cw, ch = pl["crop"]
                bg = self.bg_arrays[pl["bg"]][y:y + ch, x:x + cw]
                arr = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LANCZOS4)
            imgs.append(np.transpose(arr, (2, 0, 1)))
        batch = torch.from_numpy(np.stack(imgs)).float().to(self.device)
        return self._apply_menu_groups(batch, plans, "base")

    def _apply_menu_groups(self, images, plans, layer):
        """按 op 类型分组、保留每图 op 顺序（step 序号全局单调）对子集向量化应用。"""
        menu_def = self.sc["AUG_MENUS"].get(layer, {"menu": []})
        # 收集 (step, image_item, op) 并找类型
        steps = []
        for i, pl in enumerate(plans):
            for step, op in enumerate(pl.get("menu", [])):
                steps.append((step, i, op))
        types = {op["type"] for _, _, op in steps}
        # 分层执行：按"该类型最早出现 step"排序类型，保持逐图顺序近似（同一图跨类型顺序由 step 全局保证）
        order = sorted(types, key=lambda tp: min(st for st, _, op in steps if op["type"] == tp))
        for tp in order:
            gens = [(st, i, op) for st, i, op in steps if op["type"] == tp]
            gens.sort()
            idxs = [i for _, i, _ in gens]
            params = [op["params"] for _, _, op in gens]
            sub = images[idxs]
            if tp in ("brightness", "contrast", "color"):
                f = torch.tensor([p["value"] for p in params]).to(images)
                if tp == "brightness":
                    images[idxs] = t_brightness(sub, f)
                elif tp == "contrast":
                    images[idxs] = t_contrast(sub, f)
                else:
                    images[idxs] = t_color(sub, f)
            elif tp == "hsv":
                d3 = torch.tensor([[p["value"]] * 3 for p in params]).to(images)
                images[idxs] = t_hsv(sub, d3)
            elif tp == "sharpness":
                amt = torch.tensor([p.get("value", 1.0) for p in params]).to(images).view(-1, 1, 1, 1)
                images[idxs] = t_sharpness(sub, amt, self._rng)
            elif tp == "geometric":
                images[idxs] = t_geometric(sub, [p.get("distortion", 0.1) for p in params],
                                           self._rng)
            elif tp == "cutout":
                rects, noises = [], []
                for p in params:
                    size = p["w"]
                    hh, ww = images.shape[2], images.shape[3]
                    x0 = random.randint(0, max(0, ww - size)); y0 = random.randint(0, max(0, hh - size))
                    rects.append((y0, min(hh, y0 + size), x0, min(ww, x0 + size)))
                    noises.append(1.0)
                images = t_cutout(images, rects, torch.tensor(noises).to(images), self._rng)
            else:
                print(f"⚠️ 未知增强类型：{tp}，跳过")
        return images

    # ---- P2: 目标 slot（采样/warp/放置/标签） ----
    def _sample_target_slots(self, plan, used_targets):
        """复刻 process_round 的 slot 采样：可用池(used<REPEAT) → 随机选 → 参数落实。
        注意：二次缩放失败判定需 warp 后尺寸，故放入 _build_slot 流程。"""
        slots = []
        for _ in range(self.sc["NUM_TARGETS_PER_IMAGE"]):
            avail = [i for i in range(len(self.target_images_np))
                     if used_targets[i] < self.sc["TARGET_REPEAT"]]
            if not avail:
                break
            t = random.choice(avail)
            img, cid, idx, ttype = self.target_images_np[t]
            lo, hi = self.sc["TARGET_SCALE"].get(ttype, (0.5, 1.0))
            s_init = random.uniform(lo, hi)
            menu = self._resolve_menu(self.sc["AUG_MENUS"].get("target", {"menu": []}))
            perv = self.sc["TARGET_PERSPECTIVE"].get(ttype)
            hm_persp = None
            if perv and random.random() < perv[0]:
                # 以 s_init 缩放后的 (nw,nh) 为基准（与 _warp_slot 中图像 resize 后一致），
                # 偏移幅度也在新尺度上取，避免透视作用于错误尺度导致强度/裁剪异常。
                # 注意 img 此处为 numpy (H,W,4)，用 shape[:2]（_warp_slot 内是 tensor 才用 [2]/[3]）。
                h0, w0 = img.shape[:2]
                nw0, nh0 = max(1, int(w0 * s_init)), max(1, int(h0 * s_init))
                d = perv[1] * min(nw0, nh0)
                r = lambda: random.uniform(-d, d)
                src = np.float32([[0, 0], [nw0, 0], [nw0, nh0], [0, nh0]])
                dst = np.float32([[r(), r()], [nw0 + r(), r()], [nw0 + r(), nh0 + r()], [r(), nh0 + r()]])
                hm_persp = cv2.getPerspectiveTransform(src, dst)
            rl, rh = self.sc["TARGET_ROTATE"].get(ttype, (-45, 45))
            angle = round(random.uniform(rl, rh), 1)
            slots.append({"t": t, "cid": cid, "ttype": ttype, "s_init": s_init,
                          "menu": menu, "persp": hm_persp, "angle": angle})
            plan["slots"] = slots
        return slots

    def _warp_slot(self, slot):
        """单 slot: 目标菜单 → 透视+旋转+比例缩放折叠为**单次 grid_sample** → (tensor, tight, tw, th)。
        比例缩放尺度先按 expand 尺寸计算（与 CPU 版语义一致，失败则返回 None）。"""
        W, H = self.sc["BASE_SIZE"]
        img = self.target_dev_tensors[slot["t"]]
        # 目标菜单（RGB 算子只作用于前 3 通道；flip 全通道一起）
        for op in slot["menu"]:
            if op["type"] == "flip":
                img = img.flip(-1)
            elif op["type"] in ("brightness", "contrast", "color"):
                f = torch.tensor([op["params"]["value"]], device=img.device)
                rgb_new = {"brightness": t_brightness, "contrast": t_contrast,
                           "color": t_color}[op["type"]](img[:, :3].contiguous(), f)
                img = torch.cat([rgb_new, img[:, 3:4]], dim=1)
            elif op["type"] == "sharpness":
                img = torch.cat([t_sharpness(img[:, :3].contiguous(),
                                             torch.tensor([op["params"]["value"]],
                                                          device=img.device),
                                             self._rng),
                                 img[:, 3:4]], dim=1)
            else:
                print(f"⚠️ target 菜单不支持的 op：{op['type']}")
        nw, nh = max(1, int(img.shape[3] * slot["s_init"])), max(1, int(img.shape[2] * slot["s_init"]))
        # 必须先把 s_init 缩放落实到图像（CPU apply_augmentation: resize → 菜单 → 旋转）。
        # 否则矩阵定义于 (nw,nh) 而输入仍是 (w,h)，采样错位 → 目标被画布直角切边。
        if (nw, nh) != (img.shape[3], img.shape[2]):
            img = F.interpolate(img, size=(nh, nw), mode="bilinear")
        # 透视（3x3）+ 旋转 expand（3x3）
        hm = slot["persp"] if slot["persp"] is not None else np.eye(3)
        max_w, max_h = nw, nh
        if abs(slot["angle"]) >= 0.01:
            max_w, max_h = _rotate_expand_dims(nw, nh, slot["angle"])
            hm = _rotation_matrix(nw, nh, slot["angle"], max_w, max_h) @ hm
        # 比例缩放尺度（提前算：失败则不消耗）
        min_dim = min(max_w, max_h)
        min_scale = (self.sc["MIN_TARGET_RATIO"] * min(W, H)) / min_dim
        max_scale = min((self.sc["MAX_TARGET_RATIO"] * min(W, H)) / min_dim,
                        min(W / max_w, H / max_h))
        if max_scale <= 0 or min_scale > max_scale:
            return None
        scale = random.uniform(min_scale, max_scale)
        fw, fh = max(1, int(max_w * scale)), max(1, int(max_h * scale))
        # 折叠：S ∘ (旋转∘透视)，一次采样到最终尺寸
        s_mat = np.diag([fw / max_w, fh / max_h, 1.0])
        hm_full = s_mat @ hm
        if self.sc.get("GPU_WARP_CV", True):
            rgb_o, al_o = _warp_rgba_cv(img[:, :3].contiguous(), img[:, 3:4].contiguous(),
                                        hm_full, fh, fw)
        else:
            rgb_o, al_o = _warp_rgba_tensor(img[:, :3].contiguous(), img[:, 3:4].contiguous(),
                                            hm_full, fh, fw)
        # tight 延后到整图尾部批计算（消除逐 slot CPU 同步）
        return rgb_o, al_o, None, fw, fh

    def _place_paste(self, img_tensor, rgb_o, al_o, ttype, placed_bboxes, w=None, h=None):
        """放置拒绝 + GPU paste（warp 已含比例缩放）。返回 (rec or None, 是否成功)。
        rec = {"x","y","w","h","cid","ttype"} —— 标签延后到整图尾部批 bbox 后组装。"""
        W, H = self.sc["BASE_SIZE"]
        tw, th = w, h
        if tw > W or th > H:
            return None, False
        for _ in range(self.sc["MAX_OVERLAP_ATTEMPTS"] * 2):
            x = random.randint(0, W - tw)
            y = random.randint(0, H - th)
            ok = True
            safe = max(tw, th) * 0.1
            for b in placed_bboxes:
                dx = max(0, abs((x + tw / 2) - (b["x"] + b["width"] / 2)) - (tw + b["width"]) / 2)
                dy = max(0, abs((y + th / 2) - (b["y"] + b["height"] / 2)) - (th + b["height"]) / 2)
                if dx < safe and dy < safe:
                    ok = False
                    break
            if ok:
                break
        else:
            return None, False
        # GPU paste（feather）
        feather = self.sc["TARGET_FEATHER"].get(ttype, 0)
        a = al_o[:, 0]
        if feather > 0:
            a = a * _feather_alpha_tensor(th, tw, feather, a.device).squeeze(0)
        a = (a / 255.0).clamp(0, 1).unsqueeze(0)
        src = rgb_o[0]
        reg = img_tensor[0, :, y:y + th, x:x + tw]
        blended = a * src + (1.0 - a) * reg
        img_tensor[0, :, y:y + th, x:x + tw] = blended
        placed_bboxes.append({"x": x, "y": y, "width": tw, "height": th})
        return {"x": x, "y": y, "w": tw, "h": th}, True

    def _make_label(self, rec, tight, ttype, cid):
        """从 rec + tight（warp 画布内 alpha bbox）组装 YOLO 行（ROI/归一化/TO_BORDER 同 CPU 版）。"""
        W, H = self.sc["BASE_SIZE"]
        if tight:
            bx1, by1, bx2, by2 = tight
            bx, by, bw, bh = rec["x"] + bx1, rec["y"] + by1, bx2 - bx1, by2 - by1
        else:
            bx, by, bw, bh = rec["x"], rec["y"], rec["w"], rec["h"]
        roi = self.sc["TARGET_ROI"].get(ttype)
        if roi:
            rx, ry, rw, rh = roi
            abs_x, abs_y = bx + rx * bw, by + ry * bh
            abs_w, abs_h = rw * bw, rh * bh
        else:
            abs_x, abs_y, abs_w, abs_h = bx, by, bw, bh
        xc = (abs_x + abs_w / 2) / W
        yc = (abs_y + abs_h / 2) / H
        wn = abs_w / W; hn = abs_h / H
        tb = self.sc["TO_BORDER"]
        x_min = max(0.0 + tb, xc - wn / 2); x_max = min(1.0 - tb, xc + wn / 2)
        y_min = max(0.0 + tb, yc - hn / 2); y_max = min(1.0 - tb, yc + hn / 2)
        return f"{cid} {(x_min + x_max) / 2:.6f} {(y_min + y_max) / 2:.6f} " \
               f"{x_max - x_min:.6f} {y_max - y_min:.6f}"

    def compose_plan(self, bg_tensor, plan, used_targets, epoch):
        """单图完整合成：返回 (image_tensor[1,3,H,W], labels[list[str]])。"""
        labels = []
        if plan.get("empty"):
            return bg_tensor, labels
        self._sample_target_slots(plan, used_targets)
        placed = []
        img = bg_tensor          # 约定输入 (1,3,H,W)
        retained = []            # (al_o, rec, cid, ttype)
        for slot in plan["slots"]:
            out = self._warp_slot(slot)
            if out is None:
                plan["failed"] = plan.get("failed", 0) + 1
                if plan["failed"] >= self.sc["MAX_TARGET_FAILURE"]:
                    break
                continue
            rgb_o, al_o, tight, w, h = out
            rec, ok = self._place_paste(img, rgb_o, al_o, slot["ttype"], placed, w, h)
            if ok:
                retained.append((al_o, rec, slot["cid"], slot["ttype"]))
                used_targets[slot["t"]] += 1
                plan["failed"] = 0
            else:
                plan["failed"] = plan.get("failed", 0) + 1
                if plan["failed"] >= self.sc["MAX_TARGET_FAILURE"]:
                    break
        # 整图尾部：一次批 bbox（单次 GPU→CPU 同步）→ 组装标签
        if retained:
            maxh = max(int(a.shape[2]) for a, *_ in retained)
            maxw = max(int(a.shape[3]) for a, *_ in retained)
            stack = torch.cat([F.pad(a[0], (0, maxw - a.shape[3], 0, maxh - a.shape[2]), value=0)
                               for a, *_ in retained])
            bboxes = _alpha_bboxes_batch(stack)
            for (a_o, rec, cid, ttype), bbs in zip(retained, bboxes):
                # a 在 pad 后的坐标系：bbs 减 pad 偏移无影响（right/bottom pad：min 不变）
                labels.append(self._make_label(rec, bbs, ttype, cid))
        return img, labels

    def _apply_final_menu(self, images, plans):
        """final 菜单批量（gaussian/salt_pepper/poisson；逐 item 概率在采样时已落实）。"""
        menu_def = self.sc["AUG_MENUS"].get("final", {"menu": []})
        steps = []
        for i, pl in enumerate(plans):
            for step, op in enumerate(pl.get("final_menu", [])):
                steps.append((step, i, op))
        order = sorted({op["type"] for _, _, op in steps},
                       key=lambda tp: min(st for st, _, op in steps if op["type"] == tp))
        for tp in order:
            gens = sorted([(st, i, op) for st, i, op in steps if op["type"] == tp])
            idxs = [i for _, i, _ in gens]
            params = [op["params"] for _, _, op in gens]
            sub = images[idxs]
            if tp == "gaussian":
                s = torch.tensor([p["value"] ** 0.5 for p in params]).to(images)
                images[idxs] = t_gaussian(sub, s, self._rng)
            elif tp == "salt_pepper":
                images = t_salt_pepper(images, [p["value"] for p in params], self._rng)
            elif tp == "poisson":
                it = torch.tensor([p["value"] for p in params]).to(images)
                images[idxs] = t_poisson(sub, it, self._rng)
            elif tp == "ink_reflection":
                images[idxs] = t_ink_reflection(sub, self._rng)
            else:
                print(f"⚠️ final 未适配类型：{tp}")
        return images

    def save_batch(self, images_np, plans, labels_list, epoch):
        """并行 JPEG+txt 保存（cv2 imwrite，q=JPEG_QUALITY）。返回 futures 列表。"""
        futs = []
        for i, pl in enumerate(plans):
            name = f"E{epoch}_R{pl['round']}_{self.bg_names[pl['bg']]}"
            jp = os.path.join(self.sc["OUTPUT_DIR"], name + ".jpg")
            tp = jp.replace(".jpg", ".txt")
            with open(tp, "w") as f:
                f.write("\n".join(labels_list[i]) + ("\n" if labels_list[i] else ""))
            bgr_i = np.ascontiguousarray(images_np[i, :, :, ::-1])   # 拷贝，避免视图共享
            futs.append(self._pool.submit(cv2.imwrite, jp, bgr_i,
                                          [cv2.IMWRITE_JPEG_QUALITY, self.sc["JPEG_QUALITY"]]))
        return futs

    def run_epoch(self, epoch, rounds=None):
        """until 耗尽 or rounds 上限；空镜率照 background_ratio。"""
        os.makedirs(self.sc["OUTPUT_DIR"], exist_ok=True)
        self._round_seq = [0]
        n_rounds = rounds or self.sc["NUM_ROUNDS"]
        made = n_empty = 0
        t0 = time.perf_counter()
        used_targets = [0] * len(self.target_images_np)
        pbar = tqdm(total=n_rounds, desc=f"Epoch {epoch + 1}", unit="img",
                    dynamic_ncols=True, miniters=1)
        while made < n_rounds and not all(u >= self.sc["TARGET_REPEAT"] for u in used_targets):
            n = min(self.batch_size, n_rounds - made)
            plans, labels_list, cpu_list = [], [], []
            for _ in range(n):
                if all(u >= self.sc["TARGET_REPEAT"] for u in used_targets):
                    break
                rnd = self._round_seq[0]; self._round_seq[0] += 1
                bg = random.choice(self.bg_idx)
                ow, oh = self.bg_w[bg], self.bg_h[bg]
                if ow < 2 or oh < 2:
                    plan = {"bg": bg, "crop": None, "menu": [], "final_menu": [], "round": rnd}
                    continue
                cw = random.randint(max(1, int(ow * self.sc["MIN_CROP_RATIO"])),
                                    max(1, int(ow * self.sc["MAX_CROP_RATIO"])))
                ch = int(cw * 3 / 4); ch = min(ch, oh)
                x = random.randint(0, max(1, ow - cw)); y = random.randint(0, max(1, oh - ch))
                plan = {"bg": bg, "crop": (x, y, cw, ch), "round": rnd, "slots": [], "failed": 0,
                        "menu": self._resolve_menu(self.sc["AUG_MENUS"].get("base", {"menu": []})),
                        "final_menu": self._resolve_menu(self.sc["AUG_MENUS"].get("final", {"menu": []}))}
                if random.random() < self.sc["BACKGROUND_RATIO"]:
                    plan["slots"] = []          # 空镜
                    plan["empty"] = True
                    n_empty += 1
                plans.append(plan)
            if not plans:
                break
            images = self.build_backgrounds(plans)
            futures = []
            for i, pl in enumerate(plans):
                if all(u >= self.sc["TARGET_REPEAT"] for u in used_targets):
                    plans, images = plans[:i], images[:i]   # 用完即截断（round 级早退）
                    break
                img_i, labels = self.compose_plan(images[i:i + 1], pl, used_targets, epoch)
                images[i:i + 1] = img_i
                labels_list.append(labels)
                made += 1
                pbar.update(1)
            if not plans:
                break
            images = self._apply_final_menu(images, plans)
            cpu = images.clamp(0, 255).to("cpu", torch.uint8).permute(0, 2, 3, 1).numpy()
            self._pending += self.save_batch(cpu, plans, labels_list, epoch)
            if len(self._pending) >= 2 * self.batch_size:   # 2 批背压
                done, self._pending = self._pending[:self.batch_size], self._pending[self.batch_size:]
                for f in done:
                    f.result()
        for f in self._pending:
            f.result()
        pbar.close()
        dt = time.perf_counter() - t0
        rate = made / dt if dt > 0 else 0.0
        print(f"Epoch {epoch + 1} 完成：产出 {made} 张，用时 {dt:.1f}s （{rate:.1f} img/s）")
        return made

    def run(self, epochs=None):
        devices = self.bg_arrays
        self.bg_idx = list(range(len(devices)))
        self.bg_w = [a.shape[1] for a in devices]
        self.bg_h = [a.shape[0] for a in devices]
        self.bg_names = [os.path.splitext(os.path.basename(f))[0]
                         for f in sorted(os.listdir(self.sc["REAL_IMAGES_DIR"]))
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        self._pool = ThreadPoolExecutor(max_workers=self.jpeg_workers)
        self._pending = []
        # 目标预载设备（RGBA float [1,4,H,W]）；大源先降模板（贴入上限≈0.6*640，1600 边足有余量）
        template_max = int(getattr(self.sc, "TEMPLATE_MAX", 1600))
        self.target_dev_tensors = []
        for img, _, _, ttype in self.target_images_np:
            h, w = img.shape[:2]
            if max(h, w) > template_max:
                s = template_max / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).contiguous().to(self.device)
            self.target_dev_tensors.append(t)
            flag = "*" if max(h, w) > template_max else ""
            if flag:
                self._tmpl_note = getattr(self, "_tmpl_note", 0) + 1
        if getattr(self, "_tmpl_note", 0):
            print(f"GPU 引擎：{self._tmpl_note} 张目标已降模板至 {template_max}px")
        made_total = 0
        t_all = time.perf_counter()
        with torch.inference_mode():
            n_ep = epochs or self.sc["EPOCHS"]
            for ep in range(n_ep):
                print(f"Epoch {ep + 1}/{n_ep}")
                made = self.run_epoch(ep)
                made_total += made
        dt_all = time.perf_counter() - t_all
        rate = made_total / dt_all if dt_all > 0 else 0.0
        print(f"── 全部完成：{n_ep} epochs，共 {made_total} 张，总用时 {dt_all:.1f}s "
              f"（{rate:.1f} img/s）")
        return made


if __name__ == "__main__":
    print("gpu_engine 为库模块；入口见 gpu_synth.py")
