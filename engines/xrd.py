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


def powder_pattern(cif: str | Path, wl: float = 1.54056,
                   tth_min: float = 2.0, tth_max: float = 40.0,
                   step: float = 0.02, size_nm: float = 20.0,
                   b_iso: float = 1.0, eta: float = 0.5,
                   inst_fwhm: float = 0.10) -> dict:
    """算一张粉末 XRD 图谱。size_nm 指相干散射区尺寸（Scherrer），inf 表示无限大。"""
    from ase.io import read

    atoms = read(str(cif))
    cell = np.array(atoms.cell, dtype=float)          # 行向量 a, b, c
    frac = np.array(atoms.get_scaled_positions(wrap=True), dtype=float)
    syms = atoms.get_chemical_symbols()
    if len(syms) == 0:
        raise RuntimeError("结构里没有原子")

    # 倒格矢（不做 2π，d = 1/|g| 用晶体学约定）
    inv = np.linalg.inv(cell)                          # 列向量基 → 倒数
    rec = inv.T                                        # rec[i] 是第 i 个倒格矢

    tth = np.arange(tth_min, tth_max + step * 0.5, step)
    if tth.size < 2:
        raise RuntimeError("2θ 范围太窄")
    th = np.radians(tth) / 2.0
    d_min = wl / (2.0 * math.sin(math.radians(tth_max) / 2.0))
    d_min = max(d_min, 0.55)

    # ---- hkl 搜索范围：|h| ≤ a_perp / d_min ----
    lens = np.linalg.norm(cell, axis=1)
    hmax = int(math.ceil(lens[0] / d_min)) + 1
    kmax = int(math.ceil(lens[1] / d_min)) + 1
    lmax = int(math.ceil(lens[2] / d_min)) + 1
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
    for u in uniq:
        m = inv_idx == list(uniq).index(u)
        F += fs[u] * (np.exp(1j * phase[:, m]).sum(axis=1))
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

    # ---- 峰表：按 d 归并等价反射 ----
    order = np.argsort(-I)
    peaks, used = [], []
    for i in order:
        d = dsp[i]
        if I[i] / I.max() * 100.0 < 0.05:      # 完全消光的反射不进峰表
            break
        if any(abs(d - u) < max(2e-3, d * 2e-3) for u in used):
            continue
        used.append(d)
        peaks.append({
            "tth": round(float(tth_hkl[i]), 3),
            "d": round(float(d), 4),
            "hkl": [int(v) for v in hkl[i]],
            "I": round(float(I[i] / I.max() * 100.0), 2),
        })
        if len(peaks) >= 60:
            break
    peaks.sort(key=lambda p: p["tth"])

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
    }


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
    ap.add_argument("--b", type=float, default=1.0, help="各向同性温度因子 Å²")
    ap.add_argument("--eta", type=float, default=0.5, help="伪 Voigt 洛伦兹占比")
    ap.add_argument("--inst", type=float, default=0.10, help="仪器展宽 FWHM（度）")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    wl = a.wl if a.wl else TARGETS[a.target][1]
    try:
        r = powder_pattern(a.cif, wl, a.tth_min, a.tth_max, a.step,
                           a.size, a.b, a.eta, a.inst)
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
