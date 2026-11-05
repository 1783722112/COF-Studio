# -*- coding: utf-8 -*-
"""COF Studio · 粉末 XRD 模拟

从 CIF 直接算粉末衍射图谱：
    每个 (hkl) → d 间距 → 结构因子 F → |F|^2 × 多重度 × LP 因子
    → 按伪 Voigt 峰形铺到 2θ 轴上（宽度由晶粒尺寸决定）

只依赖 numpy + ase，不联网、不调外部程序。

命令行：
    python xrd.py --cif a.cif --wl 1.5406 --tth-min 2 --tth-max 40 --step 0.02
输出：stdout 打印一段 JSON（curve + peaks + 统计），可选写 csv/json 文件。

单位约定：波长 Å、2θ 度、晶粒尺寸 nm、B 因子 Å²。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 原子散射因子（Cromer-Mann 四高斯 + 常数项）
# f(s) = Σ a_i exp(-b_i s²) + c,  s = sinθ/λ
# 数据来自 International Tables Vol. C，覆盖 COF 里会出现的元素。
# --------------------------------------------------------------------------
CM = {
    "H": (0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519,
          0.049879, 2.20159, 0.001305),
    "B": (2.0545, 23.2185, 1.3326, 1.021, 1.0979, 60.3498,
          0.7068, 0.1403, -0.1932),
    "C": (2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687,
          0.865, 51.6512, 0.2156),
    "N": (12.2126, 0.0057, 3.1322, 9.8933, 2.0125, 28.9975,
          1.1663, 0.5826, -11.529),
    "O": (3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239,
          0.867, 32.9089, 0.2508),
    "F": (3.5392, 10.2825, 2.6412, 4.2944, 1.517, 0.2615,
          1.0243, 26.1476, 0.2776),
    "Si": (6.2915, 2.4386, 3.0353, 32.3337, 1.9891, 0.6785,
           1.541, 81.6937, 1.1407),
    "P": (6.4345, 1.9067, 4.1791, 27.157, 1.78, 0.526,
          1.4908, 68.1645, 1.1149),
    "S": (6.9053, 1.4679, 5.2034, 22.2151, 1.4379, 0.2536,
          1.5863, 56.172, 0.8669),
    "Cl": (11.4604, 0.0104, 7.1964, 1.1662, 6.2556, 18.5194,
           1.6455, 47.7749, -9.5574),
    "Br": (17.1789, 2.172, 5.2358, 16.5796, 5.6377, 0.2609,
           3.9851, 50.9121, 2.1723),
    "I": (20.2332, 2.3737, 17.9889, 0.2347, 8.0061, 15.6194,
          2.5318, 44.5938, 2.1179),
    "Fe": (11.7695, 4.7611, 7.3573, 0.3072, 3.5222, 15.3535,
           2.3045, 76.8805, 1.0363),
    "Co": (12.2841, 4.2791, 7.3409, 0.2784, 4.0034, 13.5359,
           2.3488, 71.1692, 1.0118),
    "Ni": (12.8376, 3.8785, 7.292, 0.2565, 4.4438, 12.1763,
           2.38, 66.3421, 1.0341),
    "Cu": (13.338, 3.5828, 7.1676, 0.247, 5.6158, 11.3966,
           1.6735, 64.8126, 1.191),
    "Zn": (14.0743, 3.2655, 7.0318, 0.2333, 5.1652, 10.3163,
           2.41, 58.7097, 1.3041),
}

# 波长预设（Kα 加权平均）
TARGETS = {
    "CuKa": ("Cu Kα", 1.54056),
    "CuKa1": ("Cu Kα1", 1.54059),
    "MoKa": ("Mo Kα", 0.71073),
    "CoKa": ("Co Kα", 1.78901),
    "CrKa": ("Cr Kα", 2.28970),
    "FeKa": ("Fe Kα", 1.93604),
}


def _f_atom(sym: str, s: np.ndarray) -> np.ndarray:
    """某元素在给定 s = sinθ/λ 下的散射因子。表里没有就退化到 6 电子近似。"""
    c = CM.get(sym)
    s2 = s * s
    if c is None:
        # 未收录元素：用指数衰减近似（低角接近 Z，高角迅速掉）
        z = _FAKE_Z.get(sym, 6.0)
        return z * np.exp(-18.0 * s2)
    a = (c[0], c[2], c[4], c[6])
    b = (c[1], c[3], c[5], c[7])
    out = np.full_like(s, c[8], dtype=float)
    for ai, bi in zip(a, b):
        out += ai * np.exp(-bi * s2)
    return out


_FAKE_Z = {"Li": 3, "Na": 11, "K": 19, "Mg": 12, "Ca": 20, "Al": 13, "Ti": 22}


def _pseudo_voigt(x, x0, fwhm, eta=0.5):
    """伪 Voigt：eta=1 纯洛伦兹（长尾），eta=0 纯高斯。"""
    g = 2.0 * math.sqrt(math.log(2.0) / math.pi) / fwhm
    gl = g * np.exp(-4.0 * math.log(2.0) * ((x - x0) / fwhm) ** 2)
    lz = (2.0 / math.pi) * (fwhm / 2.0) / ((x - x0) ** 2 + (fwhm / 2.0) ** 2)
    return eta * lz + (1.0 - eta) * gl


# --------------------------------------------------------------------------
# 晶胞自检
# --------------------------------------------------------------------------
def axis_geometry(atoms) -> list:
    """每个晶轴方向：轴长、最大空隙（真空）、层厚、次大空隙（都是 Å）。"""
    cell = np.array(atoms.cell, dtype=float)
    frac = np.array(atoms.get_scaled_positions(wrap=True), dtype=float)
    out = []
    for i in range(3):
        L = float(np.linalg.norm(cell[i]))
        s = np.sort(frac[:, i])
        if s.size < 2:
            out.append({"len": L, "void": L, "thick": 0.0, "second": 0.0})
            continue
        gaps = np.diff(np.concatenate([s, [s[0] + 1.0]]))
        k = int(np.argmax(gaps))
        second = float(np.sort(gaps)[-2]) * L
        out.append({"len": L, "void": float(gaps[k]) * L,
                    "thick": float((1.0 - gaps[k]) * L), "second": second})
    return out


def _split_layers(atoms, tol: float = 1.5) -> list:
    """按 z 方向把原子切成一层层（相邻 z 落差 > tol 就算换层）。"""
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    zs = z[order]
    out, start = [], 0
    for i in range(1, len(zs)):
        if zs[i] - zs[i - 1] > tol:
            out.append(order[start:i])
            start = i
    out.append(order[start:])
    return [np.asarray(idx, dtype=int) for idx in out]


def _same_layer(atoms, ia, ib, tol: float = 0.20) -> bool:
    """两层是不是「平移等价」：面内分数坐标能一一对上（同一晶格平移）。"""
    if len(ia) != len(ib):
        return False
    sym = np.array(atoms.get_chemical_symbols())
    if sorted(sym[ia]) != sorted(sym[ib]):
        return False
    frac = atoms.get_scaled_positions(wrap=True)
    cell = np.array(atoms.cell, dtype=float)
    a2 = np.array([cell[0][:2], cell[1][:2]])          # 面内基矢
    fa, fb = frac[ia][:, :2], frac[ib][:, :2]
    d = fb[:, None, :] - fa[None, :, :]
    d -= np.round(d)                                    # 最小镜像（分数坐标）
    dist = np.linalg.norm(d @ a2, axis=2)
    ok_a = np.zeros(len(ia), dtype=bool)                 # b 里每个原子都能在 a 里找到
    ok_a = dist.min(axis=1) < tol
    ok_b = dist.min(axis=0) < tol
    return bool(ok_a.all() and ok_b.all())


def _bonds_cross_c(atoms, cut: float = 1.9) -> bool:
    """沿 c 的周期性方向上有没有成键跨过胞壁。有 = 三维骨架，别乱压。"""
    from ase.neighborlist import neighbor_list
    try:
        _i, _j, S = neighbor_list("ijS", atoms, cut)
    except Exception:
        return True
    if len(S) == 0:
        return False
    return bool(np.any(S[:, 2] != 0))


def normalize_cell(atoms, d_stack: float = 3.60, min_void: float = 4.0):
    """把「超胞 / 带真空的板层」约化到真实的层间周期，专供 XRD 用。

    一个 COF 的模型文件常见三种非晶体学写法，直接拿去算衍射都会错：

    1. 单层 + 20~30 Å 真空（优化器给二维体系加真空后的默认输出）
       → c 被人为拉长，倒格矢 1/c 跟着变小，(00l) 挤到低角，
         真实的最强峰 (100) 反而被压下去。典型症状：最强峰 3.15°，
         而 d = 1.5406/(2 sin(1.575°)) = 28.0 Å —— 正好等于 c。
    2. N 层超胞（建模默认 layers=2）：c = N × 层间距
       → 出现 l 不是 N 的倍数的假 (00l)（例如 2 层的 c = 7.2 Å，
         12.3° 冒出一条 (001)，真实晶体里没有）。
    3. 1 和 2 叠加：2 层 + 真空，c = 2×3.6 + 真空。

    做法：先按 z 切层；各层等间距且互相平移等价时，取最小重复周期 p
    （AA 的 p = 1，AB 的 p = 2 —— AB 的两层不等价，c 必须留 2 层，这是对的）；
    只有一层时，若 c 明显大于「层厚 + d_stack」，按 AA 堆叠压回。

    判据里加了一条成键检查：c 方向有键跨过胞壁 = 三维骨架（ctn/dia 那类），
    一律不动，避免把真正的三维孔道压扁。
    """
    cell = np.array(atoms.cell, dtype=float)
    na, nb, nc = (float(np.linalg.norm(cell[j])) for j in range(3))
    # c 必须垂直于 a、b 且沿 z，否则"按 z 压"没有意义
    if abs(np.dot(cell[2], cell[0])) > 0.02 * na * nc:
        return None
    if abs(np.dot(cell[2], cell[1])) > 0.02 * nb * nc:
        return None
    if abs(cell[2][0]) > 0.05 or abs(cell[2][1]) > 0.05:
        return None
    if _bonds_cross_c(atoms):
        return None

    layers = _split_layers(atoms)
    c0 = float(np.linalg.norm(cell[2]))
    z = atoms.positions[:, 2]

    # ---- 情形 2/3：多层，且层间距均匀 ----
    if len(layers) >= 2:
        zc = np.array([float(z[idx].mean()) for idx in layers])
        sp = np.diff(zc)
        if sp.max() - sp.min() < 0.3 and 2.0 <= sp.mean() <= 8.0:
            period = None
            for p in range(1, len(layers)):
                if all(_same_layer(atoms, layers[k], layers[k + p])
                       for k in range(len(layers) - p)):
                    period = p
                    break
            if period is None:
                return None
            c_new = float(sp.mean()) * period
            if c_new > c0 - 0.5:            # 已经是真实周期，不用动
                return None
            keep = np.concatenate(layers[:period]) if period > 1 else layers[0]
            keep = np.sort(keep)
            new = atoms[keep]
            pos = new.get_positions()
            pos[:, 2] = pos[:, 2] - pos[:, 2].min()
            cell2 = cell.copy()
            cell2[2] = [0.0, 0.0, c_new]
            new.set_cell(cell2, scale_atoms=False)
            new.set_positions(pos)
            new.wrap()
            return new, {"void": None, "thickness": None,
                         "c_before": round(c0, 3), "c_after": round(c_new, 3),
                         "d_stack": float(d_stack), "layers": len(layers),
                         "period": period, "n_atoms_before": len(atoms)}

    # ---- 情形 1：单层 + 真空 ----
    info = axis_geometry(atoms)
    g = info[2]
    if g["void"] < min_void:
        return None
    thick = float(z.max() - z.min())
    c_new = thick + d_stack
    if c0 - c_new < 0.5:
        return None
    new = atoms.copy()
    pos = new.get_positions()
    pos[:, 2] = pos[:, 2] - pos[:, 2].min()
    cell2 = cell.copy()
    cell2[2] = [0.0, 0.0, c_new]
    new.set_cell(cell2, scale_atoms=False)
    new.set_positions(pos)
    new.wrap()
    return new, {"void": round(g["void"], 2), "thickness": round(thick, 2),
                 "c_before": round(c0, 3), "c_after": round(c_new, 3),
                 "d_stack": float(d_stack), "layers": 1, "period": 1,
                 "n_atoms_before": len(atoms)}


def fix_vacuum(atoms, d_stack: float = 3.60, min_void: float = 4.0):
    """兼容旧接口：等价于 normalize_cell。"""
    return normalize_cell(atoms, d_stack, min_void)


def lowest_reflection(cell, wl: float, rng: int = 3):
    """最低角的那条反射（即最大的 d）。用来判断 2θ 起点会不会把最强峰切掉。"""
    rec = np.linalg.inv(np.array(cell, dtype=float)).T
    best_hkl, best_d = None, 0.0
    for h in range(-rng, rng + 1):
        for k in range(-rng, rng + 1):
            for l in range(-rng, rng + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                d = 1.0 / float(np.linalg.norm(np.array([h, k, l], dtype=float) @ rec))
                if d > best_d:
                    best_hkl, best_d = (h, k, l), d
    if best_d <= 0:
        return None, None
    s = wl / (2.0 * best_d)
    if s >= 1.0:
        return None, best_d
    return 2.0 * math.degrees(math.asin(s)), best_d


# --------------------------------------------------------------------------
# 主计算
# --------------------------------------------------------------------------
def _canonical_hkl(members, fallback):
    """同一个晶面族里挑一个方便读的代表：优先 (h,0,l)，其次 h>0、k>0。

    否则六方的 (100) 会被标成 (0,-1,0)、立方 (311) 标成 (-3,-1,-1)，
    跟文献里的写法对不上。
    """
    if not members:
        return fallback

    def key(h):
        hh, kk, ll = h
        return (0 if (hh > 0 and kk == 0) else 1,
                0 if (hh > 0 and kk > 0) else 1,
                0 if hh >= 0 else 1,
                0 if kk >= 0 else 1,
                abs(ll), -hh)

    return min(members, key=key)


def _compute(atoms, wl, tth_min, tth_max, step, size_nm, b_iso, eta, inst_fwhm) -> dict:
    cell = np.array(atoms.cell, dtype=float)           # 行向量 a, b, c
    frac = np.array(atoms.get_scaled_positions(wrap=True), dtype=float)
    syms = atoms.get_chemical_symbols()
    if len(syms) == 0:
        raise RuntimeError("结构里没有原子")

    # 倒格矢（不做 2π，d = 1/|g| 用晶体学约定）
    rec = np.linalg.inv(cell).T                        # rec[i] 是第 i 个倒格矢

    tth = np.arange(tth_min, tth_max + step * 0.5, step)
    if tth.size < 2:
        raise RuntimeError("2θ 范围太窄")
    d_min = wl / (2.0 * math.sin(math.radians(tth_max) / 2.0))
    d_min = max(d_min, 0.55)

    # ---- hkl 搜索范围：|h| ≤ a / d_min ----
    lens = np.linalg.norm(cell, axis=1)
    hmax = int(math.ceil(lens[0] / d_min)) + 1
    kmax = int(math.ceil(lens[1] / d_min)) + 1
    lmax = int(math.ceil(lens[2] / d_min)) + 1
    capped = max(hmax, kmax, lmax) > 60
    hmax, kmax, lmax = min(hmax, 60), min(kmax, 60), min(lmax, 60)

    rng = [np.arange(-n, n + 1) for n in (hmax, kmax, lmax)]
    H, K, L = np.meshgrid(*rng, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1).astype(float)
    hkl = hkl[(hkl != 0).any(axis=1)]                  # 去掉 (000)

    G = hkl @ rec                                      # 笛卡尔倒格矢
    dsp = 1.0 / np.linalg.norm(G, axis=1)
    keep = dsp >= d_min
    hkl, G, dsp = hkl[keep], G[keep], dsp[keep]
    if hkl.size == 0:
        raise RuntimeError("这个 2θ 范围内没有可用衍射点")

    sth = wl / (2.0 * dsp)                             # sinθ
    tth_hkl = 2.0 * np.degrees(np.arcsin(np.clip(sth, 0, 1)))
    inside = (tth_hkl >= tth_min - 1e-6) & (tth_hkl <= tth_max + 1e-6)
    hkl, G, dsp, sth, tth_hkl = (hkl[inside], G[inside], dsp[inside],
                                 sth[inside], tth_hkl[inside])
    if hkl.size == 0:
        raise RuntimeError("这个 2θ 范围内没有可用衍射点")

    # ---- 结构因子 ----
    uniq, inv_idx = np.unique(syms, return_inverse=True)
    fs = {u: _f_atom(u, sth) for u in uniq}            # 每种元素 → (nhkl,)
    dw = np.exp(-b_iso * sth ** 2)                     # 温度因子

    phase = 2.0 * np.pi * (hkl @ frac.T)               # (nhkl, natom)
    F = np.zeros(len(hkl), dtype=complex)
    for i, u in enumerate(uniq):
        m = inv_idx == i
        F += fs[u] * np.exp(1j * phase[:, m]).sum(axis=1)
    F *= dw
    I = np.abs(F) ** 2

    # ---- 洛伦兹-偏振因子 ----
    th = np.radians(tth_hkl) / 2.0
    lp = (1.0 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th) + 1e-12)
    I = I * lp

    # ---- 峰形：Scherrer（晶粒）+ 仪器展宽 ----
    if size_nm and size_nm > 0:
        beta = 0.9 * wl / (size_nm * 10.0 * np.cos(th))   # rad
        fwhm_hkl = np.degrees(beta)
    else:
        fwhm_hkl = np.zeros_like(th)
    fwhm_hkl = np.sqrt(fwhm_hkl ** 2 + inst_fwhm ** 2)     # 高斯合成
    fwhm_hkl = np.clip(fwhm_hkl, 0.02, 3.0)

    # ---- 铺到 2θ 网格 ----
    y = np.zeros_like(tth)
    strong = I > (I.max() * 1e-6 if I.size else 0)
    for x0, ii, w in zip(tth_hkl[strong], I[strong], fwhm_hkl[strong]):
        lo = np.searchsorted(tth, x0 - 6 * w)
        hi = np.searchsorted(tth, x0 + 6 * w)
        if hi > lo:
            y[lo:hi] += ii * _pseudo_voigt(tth[lo:hi], x0, w, eta)
    if y.max() <= 0:
        raise RuntimeError("算出来全是零强度")
    y = y / y.max() * 100.0

    # ---- 峰表：按 d 归并简并反射，强度按整个晶面族求和 ----
    # （只取族里最强的一条会让 (220)/(311) 这类高多重度峰被严重低估）
    order = np.argsort(dsp)
    groups = []
    for i in order:
        d = float(dsp[i])
        tol = max(1e-3, d * 6e-4)
        if groups and abs(d - groups[-1]["d"]) <= tol:
            g = groups[-1]
            g["I"] += float(I[i])
            g["m"] += 1
            g["mem"].append([int(v) for v in hkl[i]])
            if I[i] > g["Ibest"]:
                g["Ibest"] = float(I[i])
                g["hkl"] = [int(v) for v in hkl[i]]
                g["tth"] = float(tth_hkl[i])
        else:
            groups.append({"d": d, "I": float(I[i]), "m": 1,
                           "Ibest": float(I[i]),
                           "hkl": [int(v) for v in hkl[i]],
                           "tth": float(tth_hkl[i]),
                           "mem": [[int(v) for v in hkl[i]]]})

    imax = max(g["I"] for g in groups)
    peaks = []
    for g in sorted(groups, key=lambda g: -g["I"]):
        rel = g["I"] / imax * 100.0
        if rel < 0.05:                 # 完全消光/极弱的反射不进峰表
            break
        peaks.append({"tth": round(g["tth"], 3), "d": round(g["d"], 4),
                      "hkl": _canonical_hkl(g["mem"], g["hkl"]),
                      "I": round(rel, 2), "m": g["m"]})
        if len(peaks) >= 60:
            break
    peaks.sort(key=lambda p: p["tth"])

    warn = []
    if capped:
        warn.append("晶胞较大，hkl 搜索被截断，高角部分可能缺反射")

    return {
        "ok": True,
        "wavelength": wl,
        "tth": [round(float(v), 3) for v in tth],
        "I": [round(float(v), 2) for v in y],
        "peaks": peaks,
        "n_hkl": int(len(hkl)),
        "n_atoms": int(len(syms)),
        "cell": [round(float(v), 4) for v in lens],
        "formula": atoms.get_chemical_formula(),
        "tth_min_used": float(tth[0]),
        "b_iso": b_iso,
        "warn": warn,
    }


def powder_pattern(cif: str | Path, wl: float = 1.54056,
                   tth_min: float = 2.0, tth_max: float = 40.0,
                   step: float = 0.02, size_nm: float = 20.0,
                   b_iso: float = 2.0, eta: float = 0.5,
                   inst_fwhm: float = 0.10, d_stack: float = 3.60,
                   fix_vac: bool = True, auto_extend: bool = True) -> dict:
    """算一张粉末 XRD 图谱。size_nm 指相干散射区尺寸（Scherrer），0 表示无限大。"""
    from ase.io import read

    atoms = read(str(cif))
    warn, vac = [], None

    if fix_vac:
        r = normalize_cell(atoms, d_stack)
        if r:
            atoms, vac = r
            if vac.get("layers", 1) > 1:
                warn.append("原胞是 %d 层超胞（c = %.2f Å），已按真实层间周期约化到 "
                            "c = %.2f Å（留 %d 层）计算"
                            % (vac["layers"], vac["c_before"], vac["c_after"],
                               vac["period"]))
            else:
                warn.append("原胞沿 c 有 %.1f Å 真空（单层模型），已按 AA 堆叠 "
                            "d = %.2f Å 压回 c = %.2f Å 计算"
                            % (vac["void"], vac["d_stack"], vac["c_after"]))

    res = _compute(atoms, wl, tth_min, tth_max, step, size_nm, b_iso, eta, inst_fwhm)

    low_tth, low_d = lowest_reflection(atoms.cell, wl)
    if low_tth is not None and low_tth < tth_min - 0.15:
        msg = "最低角反射在 2θ = %.2f°（d = %.2f Å），比起点 %.1f° 还小" % (
            low_tth, low_d, tth_min)
        if auto_extend:
            t0 = max(0.5, low_tth - 0.6)
            alt = _compute(atoms, wl, t0, tth_max, step, size_nm, b_iso, eta, inst_fwhm)
            xa, ya = np.array(alt["tth"]), np.array(alt["I"])
            low_part = ya[xa < tth_min - 0.01]
            in_part = ya[xa >= tth_min - 0.01]
            # 两者各自归一化到 100，所以要在同一张图里比：
            # 低角段是不是压过了起点之后的最高峰
            if (low_part.size and in_part.size
                    and low_part.max() > 1.05 * in_part.max()):
                alt["warn"].insert(0, msg + "，已自动把起点下探到 %.1f°" % t0)
                res = alt
            else:
                res["warn"].append(msg + "，但该反射很弱，不影响图谱")
        else:
            res["warn"].append(msg + "，低角信息被截断")

    res["warn"] = warn + res["warn"]
    if vac:
        res["vacuum"] = vac
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="COF Studio XRD 模拟")
    ap.add_argument("--cif", required=True)
    ap.add_argument("--wl", type=float, default=1.54056, help="波长 Å")
    ap.add_argument("--target", default="CuKa", choices=list(TARGETS),
                    help="靶材预设（给了 --wl 就忽略）")
    ap.add_argument("--tth-min", type=float, default=2.0)
    ap.add_argument("--tth-max", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--size", type=float, default=20.0, help="晶粒尺寸 nm，0=无限大")
    ap.add_argument("--b", type=float, default=2.0, help="各向同性温度因子 Å²")
    ap.add_argument("--eta", type=float, default=0.5, help="伪 Voigt 洛伦兹占比")
    ap.add_argument("--inst", type=float, default=0.10, help="仪器展宽 FWHM（度）")
    ap.add_argument("--d-stack", type=float, default=3.60,
                    help="真空层胞压回 AA 堆叠时的层间距 Å（默认 3.60，与建模一致）")
    ap.add_argument("--no-fix-vacuum", action="store_true",
                    help="保留原胞的真空层，不做 AA 压缩")
    ap.add_argument("--no-extend", action="store_true",
                    help="2θ 起点不够低时不要自动下探")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    wl = a.wl if a.wl else TARGETS[a.target][1]
    try:
        r = powder_pattern(a.cif, wl, a.tth_min, a.tth_max, a.step,
                           a.size, a.b, a.eta, a.inst,
                           d_stack=a.d_stack, fix_vac=not a.no_fix_vacuum,
                           auto_extend=not a.no_extend)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                         ensure_ascii=False))
        sys.exit(1)

    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("2theta_deg,intensity\n")
            for x, v in zip(r["tth"], r["I"]):
                f.write(f"{x:.3f},{v:.2f}\n")
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")

    # stdout 只给一段可解析的 JSON（日志走 stderr）
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
