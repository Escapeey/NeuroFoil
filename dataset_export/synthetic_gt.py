"""合成全息光栅卡 GT 渲染器（PC 穿刺阶段）。

用纯 NumPy 离线渲染「宝可梦光栅卡」的四层组合效果：
  1. 多层视差   —— 主体 / 背景 / 线稿 / 文字处于不同深度，随视角相对位移
  2. 镭射彩虹反光 —— 虹彩色相随视角连续变化
  3. Voronoi 星光 —— 星光亮度随视角闪烁
  4. 金色卡边    —— 卡片边缘金属金框 + 镜面高光

这是「用其他方法生成训练集 / GT」的自研实现（不依赖 Blender /
holo-card-studio），目标是先验证「小网络能否拟合视角→外观映射」这一核心风险，
真实 Blender 管线留待后续对齐。

角度参数化约定（与需求说明书 §3.3 对齐）：
  - theta (俯仰 pitch)：绕 X 轴旋转，单位度，范围 ±45°
  - phi   (旋转 yaw)  ：绕 Y 轴旋转，单位度，范围 ±45°
  - 相机绕卡片原点轨道运动、始终注视原点；卡片固定于 z=0 平面、法向 +Z。

渲染模型：透视相机（距离 D），逐像素投射射线，与各深度层平面 z=dz 求交，
得到卡片空间坐标 (u, v) ∈ [0,1]²（v=1 为卡片顶部），再按层材质解析求色，
自后向前 alpha 合成。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 几何常量
# ---------------------------------------------------------------------------
CAM_DIST = 3.0          # 相机到原点距离（卡片半高 = 1.0）
FOV_DEG = 60.0          # 垂直视场角（度），保证 ±45° 视角下卡片不裁切
MAX_ANGLE_DEG = 45.0    # 视角范围（与需求说明书一致）

# 各层深度（z 坐标，正方向朝向相机）
DEPTH_BG = -0.20
DEPTH_SUBJECT = 0.00
DEPTH_LINEART = 0.04
DEPTH_TEXT = 0.08
DEPTH_BORDER = 0.02

# 固定随机源（保证跨角度一致性，让网络学习的是确定性的 f(θ,φ,u,v)）
RNG_SEED = 20261125


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """向量化 HSV→RGB，h/s/v ∈ [0,1]，返回 (..., 3) ∈ [0,1]。"""
    h = np.mod(h, 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.where((i == 0) | (i == 5), v, np.where((i == 1) | (i == 4), q, p))
    g = np.where((i == 1) | (i == 2), v, np.where((i == 0) | (i == 3), p, t))
    b = np.where((i == 3) | (i == 4), v, np.where((i == 2) | (i == 5), t, p))
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# 相机与射线
# ---------------------------------------------------------------------------
def _camera(theta_deg: float, phi_deg: float):
    """返回相机位置 P、基向量 (forward, right, up)。"""
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    # P = R_y(φ) · R_x(θ) · [0, 0, D]
    P = np.array([
        CAM_DIST * np.cos(th) * np.sin(ph),
        -CAM_DIST * np.sin(th),
        CAM_DIST * np.cos(th) * np.cos(ph),
    ])
    forward = -P / np.linalg.norm(P)          # 指向原点
    world_up = np.array([0.0, 1.0, 0.0])      # ±45° 视角内 forward 不会与 up 平行
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return P, forward, right, up


def _ray_grid(res: int, forward, right, up) -> np.ndarray:
    """生成 (res, res, 3) 的单位射线方向（图像行 0 = 卡片顶部）。"""
    half = res / 2.0
    tan_f = np.tan(np.deg2rad(FOV_DEG) / 2.0)
    yy, xx = np.mgrid[0:res, 0:res]
    sx = (xx - half) / half                 # 左 -1 → 右 +1
    sy = -(yy - half) / half                # 顶 +1 → 底 -1（避免纵向翻转）
    d = forward[None, None, :] \
        + (sx * tan_f)[..., None] * right[None, None, :] \
        + (sy * tan_f)[..., None] * up[None, None, :]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 材质辅助
# ---------------------------------------------------------------------------
def _view_specular(view_dir: np.ndarray, normal, light, shininess: float) -> np.ndarray:
    """镜面高光，view_dir 形状 (res, res, 3)，返回 (res, res) ∈ [0,1]。"""
    ndv = np.sum(view_dir * normal, axis=-1, keepdims=True)
    reflect = 2.0 * ndv * normal - view_dir
    cos = np.sum(reflect * light, axis=-1)
    return np.clip(cos, 0.0, 1.0) ** shininess


# ---------------------------------------------------------------------------
# 各层材质：均为 (u, v) 的解析函数 + 视角相关量
# ---------------------------------------------------------------------------
def _foil_background(u, v, theta_n, phi_n, spec):
    """镭射彩虹反光背景：色相随视角 + 空间连续变化，叠加移动的镭射亮带。"""
    hue = 0.05 + 0.55 * (theta_n + phi_n) + 0.25 * u + 0.18 * v
    sat = np.clip(0.55 + 0.45 * spec, 0.0, 1.0)
    val = np.clip(0.28 + 0.55 * spec + 0.12 * (0.5 + 0.5 * np.sin(6.0 * np.pi * (u + v))),
                  0.0, 1.0)
    return hsv_to_rgb(hue, sat, val)


def _subject(u, v, spec):
    """主体：简化的可爱生物（身体 + 肚皮 + 耳朵 + 眼睛 + 脸颊 + 嘴）。"""
    def ellipse(cx, cy, rx, ry):
        return ((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2

    body = ellipse(0.50, 0.55, 0.30, 0.26)
    belly = ellipse(0.50, 0.64, 0.18, 0.12)
    ear_l = ellipse(0.32, 0.38, 0.09, 0.14)
    ear_r = ellipse(0.68, 0.38, 0.09, 0.14)
    eye_l = ellipse(0.42, 0.52, 0.05, 0.065)
    eye_r = ellipse(0.58, 0.52, 0.05, 0.065)
    pupil_l = ellipse(0.42, 0.52, 0.02, 0.032)
    pupil_r = ellipse(0.58, 0.52, 0.02, 0.032)
    cheek_l = ellipse(0.35, 0.63, 0.05, 0.035)
    cheek_r = ellipse(0.65, 0.63, 0.05, 0.035)
    mouth = ((u - 0.50) / 0.05) ** 2 + ((v - 0.60) / 0.012) ** 2

    base = np.array([0.96, 0.83, 0.30])[None, None, :] * (0.80 + 0.20 * spec[..., None])
    base = np.where(belly[..., None] < 1.0, np.array([1.0, 0.96, 0.80]), base)
    base = np.where((ear_l[..., None] < 1.0) | (ear_r[..., None] < 1.0),
                    np.array([0.96, 0.83, 0.30]), base)
    base = np.where((eye_l[..., None] < 1.0) | (eye_r[..., None] < 1.0),
                    np.array([1.0, 1.0, 1.0]), base)
    base = np.where((pupil_l[..., None] < 1.0) | (pupil_r[..., None] < 1.0),
                    np.array([0.08, 0.08, 0.10]), base)
    base = np.where((cheek_l[..., None] < 1.0) | (cheek_r[..., None] < 1.0),
                    np.array([0.95, 0.35, 0.30]), base)
    base = np.where(((mouth < 1.0) & (v > 0.60))[..., None],
                    np.array([0.20, 0.12, 0.10]), base)

    a = ((body < 1.0) | (ear_l < 1.0) | (ear_r < 1.0)).astype(np.float32)
    return base, a


def _lineart(u, v):
    """线稿：主体轮廓 + 五官描边（高频边缘，检验网络学习能力的关键）。"""
    def ellipse(cx, cy, rx, ry):
        return ((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2

    body = ellipse(0.50, 0.55, 0.30, 0.26)
    ear_l = ellipse(0.32, 0.38, 0.09, 0.14)
    ear_r = ellipse(0.68, 0.38, 0.09, 0.14)
    eye_l = ellipse(0.42, 0.52, 0.05, 0.065)
    eye_r = ellipse(0.58, 0.52, 0.05, 0.065)
    mouth = ((u - 0.50) / 0.05) ** 2 + ((v - 0.60) / 0.012) ** 2

    w = 0.03  # 描边带宽
    stroke = ((np.abs(body - 1.0) < w) | (np.abs(ear_l - 1.0) < w)
              | (np.abs(ear_r - 1.0) < w) | (np.abs(eye_l - 1.0) < w)
              | (np.abs(eye_r - 1.0) < w))
    stroke |= (mouth < 1.0) & (v > 0.60)
    color = np.broadcast_to(np.array([0.10, 0.08, 0.10])[None, None, :],
                            u.shape + (3,))
    return color, stroke.astype(np.float32)


_TEXT_RGB = None
_TEXT_A = None


def _text_layer(res: int):
    """文字层：用 Pillow 渲染标题与编号，得到卡片空间纹理 (res,res)；缓存避免重复。"""
    global _TEXT_RGB, _TEXT_A
    if _TEXT_RGB is not None and _TEXT_RGB.shape[0] == res:
        return _TEXT_RGB, _TEXT_A

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", int(res * 0.052))
        num_font = ImageFont.truetype("arial.ttf", int(res * 0.030))
    except OSError:
        title_font = num_font = ImageFont.load_default()

    title = "NEUROFOIL"
    tw = draw.textlength(title, font=title_font)
    draw.text(((res - tw) / 2, res * 0.86), title, font=title_font,
              fill=(255, 252, 245, 255))
    draw.text((res * 0.70, res * 0.06), "No.001", font=num_font,
              fill=(255, 252, 245, 255))

    arr = np.asarray(img).astype(np.float32) / 255.0
    _TEXT_RGB = arr[..., :3]
    _TEXT_A = arr[..., 3:4]
    return _TEXT_RGB, _TEXT_A


def _border(u, v, spec):
    """金色卡边：卡片边缘金属金框 + 镜面高光，含一条内描边定义边界。"""
    edge = np.minimum(np.minimum(u, v), np.minimum(1.0 - u, 1.0 - v))
    w = 0.035
    frame = (edge < w) & (edge > w * 0.22)          # 外框金
    line = (np.abs(edge - w) < 0.006)               # 边界内描边
    gold = np.array([0.83, 0.66, 0.20])[None, None, :] * (0.55 + 0.85 * spec[..., None])
    dark = np.array([0.30, 0.22, 0.10])
    rgb = np.where(frame[..., None], gold, np.zeros_like(gold))
    rgb = np.where(line[..., None], dark, rgb)
    a = (frame | line).astype(np.float32)
    return rgb, a


def _voronoi_sparkle(u, v, theta_n, phi_n):
    """Voronoi 星光：固定种子点的星光，亮度随视角闪烁（各自带相位）。"""
    rng = np.random.default_rng(RNG_SEED)
    n_seed = 30
    su = rng.random(n_seed).astype(np.float32)
    sv = rng.random(n_seed).astype(np.float32)
    a_i = rng.random(n_seed).astype(np.float32)
    b_i = rng.random(n_seed).astype(np.float32)
    phase = rng.random(n_seed).astype(np.float32)

    du = u[..., None].astype(np.float32) - su[None, None, :]
    dv = v[..., None].astype(np.float32) - sv[None, None, :]
    d2 = du * du + dv * dv
    sigma2 = 2.0 * (0.028 ** 2)
    star = np.exp(-d2 / sigma2)

    tw = 0.5 + 0.5 * np.cos(2.0 * np.pi * (a_i * theta_n + b_i * phi_n) + phase)
    bright = np.sum(star * tw[None, None, :], axis=-1)
    bright = np.clip((bright - 0.15) * 1.6, 0.0, 1.0)
    return np.array([1.0, 1.0, 0.98])[None, None, :] * bright[..., None], bright.astype(np.float32)


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render(theta_deg: float, phi_deg: float, res: int = 512,
           return_float: bool = False) -> np.ndarray:
    """渲染单帧：给定视角 (θ, φ) 度，返回 (res, res, 4) RGBA。"""
    theta_n = theta_deg / MAX_ANGLE_DEG
    phi_n = phi_deg / MAX_ANGLE_DEG

    P, forward, right, up = _camera(theta_deg, phi_deg)
    ray = _ray_grid(res, forward, right, up)           # (res,res,3)
    view_dir = -ray                                    # 表面 → 相机
    light = np.array([0.35, 0.55, 0.75])
    light = light / np.linalg.norm(light)
    normal = np.array([0.0, 0.0, 1.0])
    spec = _view_specular(view_dir, normal, light, shininess=14.0)  # (res,res)

    text_rgb, text_a = _text_layer(res)                # 离散文字纹理

    out_rgb = np.zeros((res, res, 3), dtype=np.float32)
    out_a = np.zeros((res, res), dtype=np.float32)

    def intersect(dz):
        denom = ray[..., 2]                                  # 相机朝 -Z，恒为负
        denom = np.where(np.abs(denom) < 1e-6, -1e-6, denom)  # 保号除法保护
        t = (dz - P[2]) / denom
        x = P[0] + t * ray[..., 0]
        y = P[1] + t * ray[..., 1]
        u = (x + 1.0) / 2.0
        v = (y + 1.0) / 2.0
        inside = ((x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)).astype(np.float32)
        return u, v, inside

    def composite(rgb, a):
        nonlocal out_rgb, out_a
        out_rgb = out_rgb * (1.0 - a[..., None]) + rgb * a[..., None]
        out_a = out_a * (1.0 - a) + a

    # 1) 背景（整个卡片正面）
    u, v, inside = intersect(DEPTH_BG)
    composite(_foil_background(u, v, theta_n, phi_n, spec), inside)
    # 2) 星光（同一平面，衬在主体之后）
    u, v, inside = intersect(DEPTH_BG)
    star_rgb, star_a = _voronoi_sparkle(u, v, theta_n, phi_n)
    composite(star_rgb, star_a * inside)
    # 3) 主体
    u, v, inside = intersect(DEPTH_SUBJECT)
    sub_rgb, sub_a = _subject(u, v, spec)
    composite(sub_rgb, sub_a * inside)
    # 4) 线稿
    u, v, inside = intersect(DEPTH_LINEART)
    lin_rgb, lin_a = _lineart(u, v)
    composite(lin_rgb, lin_a * inside)
    # 5) 文字
    u, v, inside = intersect(DEPTH_TEXT)
    ui = np.clip((u * (res - 1)).astype(np.int32), 0, res - 1)
    vi = np.clip((v * (res - 1)).astype(np.int32), 0, res - 1)
    composite(text_rgb[vi, ui], (text_a[vi, ui, 0] * inside))
    # 6) 金色卡边
    u, v, inside = intersect(DEPTH_BORDER)
    bd_rgb, bd_a = _border(u, v, spec)
    composite(bd_rgb, bd_a * inside)

    rgba = np.clip(np.concatenate([out_rgb, out_a[..., None]], axis=-1), 0.0, 1.0)
    if return_float:
        return rgba
    return (rgba * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 数据集生成
# ---------------------------------------------------------------------------
def generate_dataset(angles_deg: np.ndarray, res: int = 512) -> np.ndarray:
    """批量渲染：angles_deg 形状 (N, 2)，返回 (N, res, res, 4) uint8。"""
    out = np.empty((len(angles_deg), res, res, 4), dtype=np.uint8)
    for i, (th, ph) in enumerate(angles_deg):
        out[i] = render(float(th), float(ph), res=res)
    return out