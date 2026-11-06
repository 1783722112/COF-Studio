# -*- coding: utf-8 -*-
"""修「一个氢同时挨着两个重原子」的结构（3D COF 建模的老毛病）。

只动氢，重原子坐标、晶胞、拓扑一个都不碰。

用法：
    python fix_bridge_h.py in.cif                  # 输出 in_fixed.cif
    python fix_bridge_h.py in.cif -o out.cif
    python fix_bridge_h.py work/*.cif --inplace     # 原地改（先备份 .bak）

判据：某个 H 到第二个重原子的距离 < 1.45 Å。正常 X–H 是 0.96–1.10 Å，
氢键 H···X 通常 ≥1.6 Å，所以落在 (1.10, 1.45) 这一段的基本都是
"缝合把氢推进了连接键里"。修法：绕主键原子转到两根重键的角平分线上，
再在平面内扫一圈避让其它原子（含周期像）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.io import read, write
except ImportError:
    sys.exit("需要 ase：pip install ase")


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def find_bridging(atoms, h_cut=1.45):
    """返回 [(h, a, b)]：h 同时贴着重原子 a（主键）和 b。"""
    S = atoms.get_chemical_symbols()
    P = atoms.get_positions()
    L = np.array(atoms.get_cell().lengths())

    def mic(dv):
        return dv - np.round(dv / L) * L

    out = []
    for i, s in enumerate(S):
        if s != "H":
            continue
        nb = sorted(((float(np.linalg.norm(mic(P[j] - P[i]))), j)
                     for j in range(len(S)) if j != i and S[j] != "H"),
                    key=lambda t: t[0])
        if len(nb) >= 2 and nb[1][0] < h_cut:
            out.append((i, nb[0][1], nb[1][1]))
    return out


def repair(atoms, blen=1.09, verbose=True):
    S = atoms.get_chemical_symbols()
    P = atoms.get_positions().copy()
    cell = np.asarray(atoms.get_cell(), float)
    Linv = np.linalg.inv(cell)
    L = np.array(atoms.get_cell().lengths())

    def mic(dv):
        return (dv @ Linv - np.round(dv @ Linv)) @ cell

    bad = find_bridging(atoms)
    if not bad:
        if verbose:
            print("  没有双连氢，不用改")
        return atoms, 0

    for h, a, b in bad:
        cp = P[a]
        v1 = mic(P[b] - cp)                      # 被误连的那根键
        v2 = [mic(P[j] - cp) for j in range(len(S))
              if j != a and j != h and S[j] != "H"
              and float(np.linalg.norm(mic(P[j] - cp))) < 1.75]
        v2.sort(key=lambda v: float(np.linalg.norm(v)))
        u2 = unit(v2[0]) if v2 else -unit(v1)
        u1 = unit(v1)
        nrm = np.cross(u1, u2)
        if float(np.linalg.norm(nrm)) < 0.15:    # 近共线，平面退化
            for ax in (np.array([0., 0., 1.]), np.array([0., 1., 0.]),
                       np.array([1., 0., 0.])):
                nrm = np.cross(u1, ax)
                if float(np.linalg.norm(nrm)) > 0.15:
                    break
        nrm = unit(nrm)
        e2 = unit(np.cross(nrm, u1))
        bis = -unit(u1 + u2)
        if float(np.linalg.norm(u1 + u2)) < 0.05:
            bis = e2

        keep = np.array([j for j in range(len(S)) if j not in (h, a)], int)
        best = None
        for dl in np.arange(-80.0, 80.01, 4.0):
            d = unit(np.cos(np.radians(dl)) * bis + np.sin(np.radians(dl)) * e2)
            q = cp + blen * d
            f = (P[keep] - q) @ Linv
            f -= np.round(f)
            dm = float(np.linalg.norm(f @ cell, axis=1).min())
            sc = dm - 0.0020 * abs(dl)
            if best is None or sc > best[0]:
                best = (sc, dm, d)
        P[h] = cp + blen * best[2]

    new = Atoms(symbols=list(S), positions=P, cell=atoms.cell, pbc=atoms.pbc)
    if verbose:
        print(f"  修好 {len(bad)} 个双连氢（只转氢的方位，重原子不动）")
    return new, len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--inplace", action="store_true", help="原地改，留 .bak")
    ap.add_argument("--suffix", default="_fixed")
    a = ap.parse_args()

    total = 0
    for f in a.files:
        p = Path(f)
        atoms = read(str(p))
        print(f"[{p.name}] {len(atoms)} 原子")
        new, n = repair(atoms)
        total += n
        if not n:
            continue
        if a.inplace:
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            write(str(p), new, format="cif")
            print(f"  → 已写回 {p}（备份 {p.name}.bak）")
        else:
            out = Path(a.out) if a.out else p.with_name(p.stem + a.suffix + ".cif")
            write(str(out), new, format="cif")
            print(f"  → {out}")
    print(f"合计修复 {total} 个氢")


if __name__ == "__main__":
    main()
