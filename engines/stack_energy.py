# -*- coding: utf-8 -*-
"""COF Studio · 层间堆叠能扫描（AA / AB / 自定义偏移）

为什么单独做一个引擎
--------------------
单层优化只能回答「骨架对不对」，回答不了「层怎么堆」。COF 的 AA / AB 之争
本质上是**层间结合能**的竞争：

    Eb(d) = [E_2L(d) - 2·E_1L] / N_layer

扫一遍层间距 d，取极小值就是这一种堆法的结合能（负值 = 稳定）。两种堆法
各自取极小后相减，才知道到底哪种更稳、差多少 meV/atom。

三个必须说清楚的点
------------------
1. **必须用带色散的势**。MACE-MP-0 + D3(BJ) 可以；MACE-OFF23 勉强；QCOF
   是短程势（r_max = 4 Å），没有长程吸引，层间会被算成零或排斥 —— 只能用
   来比层内，不能用来比堆叠。引擎会在日志里直接警告。
2. **d 的定义**：下层最高原子到上层最低原子的 z 间距（最近原子间距），
   不是质心距。3.4~3.6 Å 是化学上合理的 vdW 接触，跟实验报的层间距对得上。
3. **刚体扫描会低估结合能**（层不能自己褶皱 / 滑移）。所以要 --relax-min：
   在最优点放开原子再弛豫一轮，给出「弛豫后」的结合能。这是报给别人看的数。

命令行
------
    python stack_energy.py mono.cif -o out.json \
        -p mace-mp --model medium --device cpu \
        --modes aa,ab --gap-min 3.0 --gap-max 6.0 --gap-step 0.2 \
        --relax-mono --relax-min --csv out.csv

stdout 是给人看的进度（[i/N] 前缀，界面拿它算进度条），末行是 JSON。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 堆叠模式
# --------------------------------------------------------------------------
PRESET = {
    "aa": ("AA", (0.0, 0.0)),
    "ab": ("AB", (0.5, 0.5)),
    # 六方格子常见的第三次平移（石墨 ABC 型），COF 里偶尔用得上
    "abc": ("ABC", (1.0 / 3.0, 2.0 / 3.0)),
}


def parse_modes(spec: str) -> list[dict]:
    """把 "aa,ab" 或 "aa,0.25,0.75" 解析成模式列表。

    带逗号的裸数字对（如 0.25,0.75）当成自定义分数偏移，标签写成 off(0.25,0.75)。
    这样界面上的「自由移动」不用另开一条代码路径。
    """
    out, seen = [], set()
    for raw in (spec or "aa,ab").replace(";", ",").split(","):
        t = raw.strip().lower()
        if not t:
            continue
        if t in PRESET:
            label, off = PRESET[t]
            key = t
        else:
            try:
                off = (float(t), float(t))          # 单个数 → 对角滑移
                label = f"off({off[0]:.3f},{off[1]:.3f})"
                key = "off_%.3f_%.3f" % off
            except ValueError:
                continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": key, "label": label, "offset": [float(off[0]), float(off[1])]})
    return out or [{"id": "aa", "label": "AA", "offset": [0.0, 0.0]}]


# --------------------------------------------------------------------------
# 结构处理
# --------------------------------------------------------------------------
def layer_thickness(atoms) -> float:
    z = atoms.positions[:, 2]
    return float(z.max() - z.min())


def split_from_multilayer(atoms, tol: float = 1.2):
    """输入如果是多层（比如界面上建的 AA2L），只留最下面那一层。

    判据不能只看「最大空隙够大」——褶皱厉害的单层内部也有大空隙。
    所以要求最大空隙同时满足：> 1.2 Å，且至少是第二大空隙的 1.8 倍。
    真双层的层间空隙会明显甩开层内空隙，褶皱层则不会。
    """
    z = np.sort(atoms.positions[:, 2])
    if len(z) < 8:
        return atoms, 1
    gaps = np.sort(np.diff(z))[::-1]
    if gaps[0] < tol or (len(gaps) > 1 and gaps[0] < 1.8 * gaps[1]):
        return atoms, 1
    cut = z[int(np.argmax(np.diff(z)))] + 1e-6
    keep = atoms.positions[:, 2] <= cut
    lo = int(keep.sum())
    if lo < 4 or lo > len(atoms) - 4 or lo < 0.25 * len(atoms):
        return atoms, 1
    return atoms[keep], 2


def prep_layer(atoms):
    """把单层平移到 z∈[0, t]，去掉多余的 c 轴真空。"""
    lay = atoms.copy()
    z = lay.positions[:, 2]
    lay.positions[:, 2] = z - z.min()
    return lay


def build_bilayer(lay, offset, gap, vacuum, c_ref=None):
    """按给定分数偏移和最近原子间距 d 搭双层。

    下层占 [0, t]，上层占 [t+d, 2t+d]；c = 2t + d + vacuum。
    """
    cell = np.array(lay.cell, dtype=float)
    t = layer_thickness(lay)
    bot = lay.copy()
    top = lay.copy()
    ox, oy = offset
    if ox or oy:
        shift = ox * cell[0] + oy * cell[1]
        top.positions[:, 0] += shift[0]
        top.positions[:, 1] += shift[1]
    top.positions[:, 2] += t + gap
    two = bot + top
    c = c_ref if c_ref else (2 * t + gap + vacuum)
    two.set_cell(slab_cell(lay, c), scale_atoms=False)
    return two, t


def slab_cell(lay, c_len):
    """面内 a/b 原样保留，c 轴沿 z 拉到 c_len。

    COF 单层胞几乎都是 c ⊥ ab，直接换 c 长度就行；万一 a/b 带 z 分量
    （斜胞），也不改写它们 —— 只把 c 换成竖直方向，原子坐标不动。
    """
    cell = np.array(lay.cell, dtype=float)
    if abs(cell[0][2]) > 0.05 or abs(cell[1][2]) > 0.05:
        print("[提示] 面内矢量带 z 分量（斜胞），c 轴按竖直处理", flush=True)
    new = np.zeros((3, 3))
    new[0] = cell[0]
    new[1] = cell[1]
    new[2] = [0.0, 0.0, float(c_len)]
    return new


def bilayer_cell(lay, gap, vacuum):
    """双层胞：c = 2t + d + vacuum。"""
    t = layer_thickness(lay)
    return slab_cell(lay, 2 * t + gap + vacuum), t


def cell_area(lay) -> float:
    cell = np.array(lay.cell, dtype=float)
    return float(np.linalg.norm(np.cross(cell[0], cell[1])))


def min_interlayer_dist(two, n_layer: int) -> float:
    """两层之间真实的最近原子距离（用来检查有没有叠穿）。"""
    a = two.positions[:n_layer]
    b = two.positions[n_layer:]
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(math.sqrt(d2.min()))


# --------------------------------------------------------------------------
# 势函数
# --------------------------------------------------------------------------
def make_calc(args, atoms=None):
    """复用 opt_cof 的加载逻辑（QCOF 落盘、MACE 缓存预检都在里面）。

    opt_cof.make_calculator 会拿结构做元素覆盖 / 截断半径体检，
    所以要把单层结构传进去（传 None 会直接 AttributeError）。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import opt_cof
    from types import SimpleNamespace

    ns = SimpleNamespace(
        potential=args.potential, model=args.model, model_file=args.model_file,
        allow_download=args.allow_download, device=args.device,
        dtype=args.dtype, no_dispersion=args.no_dispersion,
    )
    calc, label, _r = opt_cof.make_calculator(ns, atoms)
    return calc, label


def relax_atoms(atoms, calc, fmax=0.03, steps=300):
    """固定晶胞、只动原子。返回 (能量, 最大残余力, 实际步数, 是否收敛)。"""
    from ase.optimize import FIRE

    work = atoms.copy()
    work.calc = calc
    conv, n = False, 0
    try:
        opt = FIRE(work, logfile=None, maxstep=0.15)
        opt.run(fmax=fmax, steps=steps)
        n = int(opt.get_number_of_steps())
        conv = bool(opt.converged())
    except Exception as exc:                      # 不收敛也别把整个任务带崩
        print(f"      [弛豫异常] {type(exc).__name__}: {exc}", flush=True)
    f = work.get_forces()
    fmax_now = float(np.sqrt((f ** 2).sum(axis=1)).max())
    return float(work.get_potential_energy()), fmax_now, n, conv, work


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def parabolic_min(xs, ys, i):
    """三点抛物线插值，返回极小点的 x 与 y 估计。"""
    if i <= 0 or i >= len(xs) - 1:
        return float(xs[i]), float(ys[i])
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    den = (y0 - 2 * y1 + y2)
    if abs(den) < 1e-12:
        return float(x1), float(y1)
    # 等间距三点：顶点偏移 = 0.5·(y0-y2)/den · h
    h = x1 - x0
    dx = 0.5 * (y0 - y2) / den * h
    dx = max(-h, min(h, dx))
    # 用顶点式求 y 更稳
    a = den / (2 * h * h)
    return float(x1 + dx), float(y1 - a * dx * dx)


def run(args) -> dict:
    from ase.io import read

    t0 = time.time()
    src = Path(args.input)
    if not src.is_file():
        raise RuntimeError(f"找不到结构文件 {src}")

    atoms = read(str(src))
    if not len(atoms):
        raise RuntimeError("结构里没有原子")
    atoms, n_layers = split_from_multilayer(atoms)
    if n_layers > 1:
        print(f"[提示] 输入里有不止一层，只取最下面一层来算"
              f"（单层 {len(atoms)} 原子）", flush=True)

    lay = prep_layer(atoms)
    n_layer = len(lay)
    t = layer_thickness(lay)
    area = cell_area(lay)
    syms = lay.get_chemical_symbols()
    print(f"[结构] {src.name}  单层 {n_layer} 原子  "
          f"a={np.linalg.norm(lay.cell[0]):.3f} b={np.linalg.norm(lay.cell[1]):.3f} Å  "
          f"层厚 {t:.2f} Å  面积 {area:.2f} Å²", flush=True)

    calc, label = make_calc(args, lay)
    print(f"[势函数] {label}  potential={args.potential}"
          + ("  +D3(BJ)" if args.potential == "mace-mp" and not args.no_dispersion else ""),
          flush=True)
    if args.potential == "qcof":
        print("[警告] QCOF 是短程势（r_max=4 Å），没有长程色散项 —— "
              "层间结合能会被严重低估，甚至算成正值。比堆叠请用 mace-mp + D3。",
              flush=True)

    # ---- 单层参考：与双层同 c 的胞里算一次，避免胞长带来的零点偏移 ----
    gap_lo, gap_hi = args.gap_min, args.gap_max
    new_cell, _ = bilayer_cell(lay, gap_lo, args.vacuum)
    ref = lay.copy()
    ref.set_cell(new_cell, scale_atoms=False)
    ref.calc = calc
    e1_raw = float(ref.get_potential_energy())
    print(f"[单层] 参考能量 {e1_raw:.4f} eV  ({e1_raw / n_layer:.4f} eV/atom)", flush=True)

    mono_relaxed = None
    if args.relax_mono:
        print("[单层] 先弛豫（固定晶胞，只动原子）…", flush=True)
        e1, fmax1, steps1, conv1, mono_best = relax_atoms(
            ref, calc, args.mono_fmax, args.mono_steps)
        mono_relaxed = {"E_before": e1_raw, "E_after": e1,
                        "fmax": fmax1, "steps": steps1, "converged": conv1}
        print(f"[单层] 弛豫后 {e1:.4f} eV  最大力 {fmax1:.4f} eV/Å  "
              f"{steps1} 步  {'收敛' if conv1 else '未完全收敛'}", flush=True)
        # 用弛豫后的层重算参考能量（同一 c，保证 E2 - 2E1 里没有胞长假象）
        lay = prep_layer(mono_best)
        t = layer_thickness(lay)
        new_cell, _ = bilayer_cell(lay, gap_lo, args.vacuum)
        ref = lay.copy()
        ref.set_cell(new_cell, scale_atoms=False)
        ref.calc = calc
        e1_raw = float(ref.get_potential_energy())

    modes = parse_modes(args.modes)
    gaps = []
    g = gap_lo
    while g <= gap_hi + 1e-9:
        gaps.append(round(g, 3))
        g += args.gap_step
    if not gaps:
        gaps = [gap_lo]

    total_pts = len(gaps) * len(modes)
    print(f"[扫描] {len(modes)} 种堆叠 × {len(gaps)} 个间距 = {total_pts} 个点  "
          f"d ∈ [{gap_lo:.2f}, {gap_hi:.2f}] Å  step {args.gap_step}", flush=True)

    results, k = [], 0
    for m in modes:
        rows = []
        for d in gaps:
            k += 1
            two, _ = build_bilayer(lay, m["offset"], d, args.vacuum)
            two.set_cell(new_cell, scale_atoms=False)
            dmin = min_interlayer_dist(two, n_layer)
            two.calc = calc
            e2 = float(two.get_potential_energy())
            eb = (e2 - 2.0 * e1_raw) / 2.0            # eV / 界面
            eb2 = eb * 1000.0 / area                  # meV / Å²
            eba = eb * 1000.0 / n_layer                # meV / atom（每层每原子）
            rows.append({"gap": d, "E": e2, "Eb_meV_A2": eb2,
                         "Eb_meV_atom": eba, "dmin": dmin})
            print(f"[{k}/{total_pts}] {m['label']:<14s} d={d:5.2f}  "
                  f"E={e2:11.4f} eV  Eb={eb2:8.2f} meV/Å²  "
                  f"({eba:7.2f} meV/atom)  dmin={dmin:.2f}", flush=True)
            if dmin < 1.2:
                print(f"      [警告] 两层最近原子只有 {dmin:.2f} Å —— "
                      f"这个间距已经在叠穿了，只作参考", flush=True)
            del two
        ys = [r["Eb_meV_A2"] for r in rows]
        i = int(np.argmin(ys))
        dminv, yminv = parabolic_min([r["gap"] for r in rows], ys, i)
        # Eb(meV/atom) = Eb(meV/Å²) × 面积 / 每层原子数 —— 两个单位只差常数
        best = {"id": m["id"], "label": m["label"], "offset": m["offset"],
                "rows": rows,
                "gap_min": dminv,
                "Eb_min_meV_A2": yminv,
                "Eb_min_meV_atom": yminv * area / n_layer,
                "E_min": rows[i]["E"],
                # 插值极小点对应的双层总能量（仅作展示）
                "E_min_interp": 2.0 * e1_raw + 2.0 * (yminv / 1000.0) * area}
        print(f">>> {m['label']}: d* = {dminv:.3f} Å   "
              f"Eb = {yminv:.2f} meV/Å²  ({best['Eb_min_meV_atom']:.2f} meV/atom)",
              flush=True)
        results.append(best)

    # ---- 最优点弛豫（可选）：把层间的褶皱/滑移放出来 ----
    if args.relax_min:
        print("[弛豫] 对每种堆叠的最优间距做一次原子弛豫（固定晶胞）…", flush=True)
        for r in results:
            two, _ = build_bilayer(lay, r["offset"], r["gap_min"], args.vacuum)
            two.set_cell(new_cell, scale_atoms=False)
            two.calc = calc
            e, fmax, steps, conv, best = relax_atoms(
                two, calc, args.relax_fmax, args.relax_steps)
            eb = (e - 2.0 * e1_raw) / 2.0
            r["relaxed"] = {
                "E": e, "fmax": fmax, "steps": steps, "converged": conv,
                "Eb_meV_A2": eb * 1000.0 / area,
                "Eb_meV_atom": eb * 1000.0 / n_layer,
            }
            # 滑移量：上下层质心在面内的相对位移
            ca = best.positions[:n_layer].mean(axis=0)
            cb = best.positions[n_layer:].mean(axis=0)
            r["relaxed"]["slide"] = [round(float(cb[0] - ca[0]), 3),
                                     round(float(cb[1] - ca[1]), 3)]
            r["relaxed"]["dmin"] = min_interlayer_dist(best, n_layer)
            print(f"      {r['label']}: 弛豫后 Eb = {r['relaxed']['Eb_meV_atom']:.2f} meV/atom "
                  f"({r['relaxed']['Eb_meV_A2']:.2f} meV/Å²)  "
                  f"fmax={fmax:.3f}  滑移 {r['relaxed']['slide']}", flush=True)
            del best, two

    # ---- 两两比较 ----
    delta = {}
    basis = "relaxed" if any(m.get("relaxed") for m in results) else "rigid"
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            ea = a.get("relaxed", {}).get("Eb_meV_atom", a["Eb_min_meV_atom"])
            eb_ = b.get("relaxed", {}).get("Eb_meV_atom", b["Eb_min_meV_atom"])
            delta[f"{a['label']}|{b['label']}"] = {
                "dE_meV_atom": eb_ - ea,
                "preferred": a["label"] if ea < eb_ else b["label"],
            }
            print(f">>> {b['label']} − {a['label']} = {(eb_ - ea):+.2f} meV/atom  "
                  f"→ 更稳的是 {a['label'] if ea < eb_ else b['label']}", flush=True)

    out = {
        "ok": True,
        "input": src.name,
        "potential": args.potential,
        "potential_label": label,
        "dispersion": bool(args.potential == "mace-mp" and not args.no_dispersion),
        "mono": {
            "file": src.name, "n_atoms": n_layer,
            "formula": lay.get_chemical_formula(),
            "from_layers": n_layers,
            "a": round(float(np.linalg.norm(lay.cell[0])), 4),
            "b": round(float(np.linalg.norm(lay.cell[1])), 4),
            "thickness": round(t, 3),
            "area": round(area, 3),
            "E": e1_raw,
            "E_per_atom": e1_raw / n_layer,
            "relaxed": mono_relaxed,
        },
        "settings": {
            "gap_min": gap_lo, "gap_max": gap_hi, "gap_step": args.gap_step,
            "vacuum": args.vacuum, "relax_min": bool(args.relax_min),
            "modes": [m["label"] for m in modes],
        },
        "modes": results,
        "delta": delta,
        "delta_basis": basis,
        "elapsed": round(time.time() - t0, 1),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="COF Studio 层间堆叠能扫描")
    ap.add_argument("input", help="单层结构（cif/xyz/vasp）")
    ap.add_argument("-o", "--out", default="stack.json")
    ap.add_argument("-p", "--potential", default="mace-mp",
                    choices=["mace-mp", "mace-off", "qcof", "file"])
    ap.add_argument("--model", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--no-dispersion", action="store_true")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--modes", default="aa,ab",
                    help="aa / ab / abc / 单个数字（对角滑移），逗号分隔")
    ap.add_argument("--gap-min", type=float, default=3.0)
    ap.add_argument("--gap-max", type=float, default=6.0)
    ap.add_argument("--gap-step", type=float, default=0.2)
    ap.add_argument("--vacuum", type=float, default=20.0)
    ap.add_argument("--relax-mono", action="store_true", default=True)
    ap.add_argument("--no-relax-mono", dest="relax_mono", action="store_false")
    ap.add_argument("--mono-fmax", type=float, default=0.03)
    ap.add_argument("--mono-steps", type=int, default=300)
    ap.add_argument("--relax-min", action="store_true", default=True)
    ap.add_argument("--no-relax-min", dest="relax_min", action="store_false")
    ap.add_argument("--relax-fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=300)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if a.gap_max <= a.gap_min:
        a.gap_max = a.gap_min + 1.0
    if a.gap_step <= 0:
        a.gap_step = 0.2
    n_pt = (a.gap_max - a.gap_min) / a.gap_step + 1
    if n_pt > 120:
        print(f"[提示] 扫描点太多（{n_pt:.0f}），步长自动调整为 0.25 Å", flush=True)
        a.gap_step = 0.25
    if a.vacuum < 12:
        a.vacuum = 12.0

    try:
        r = run(a)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        sys.exit(1)

    if a.csv:
        try:
            with open(a.csv, "w", encoding="utf-8") as f:
                f.write("mode,gap_A,E_eV,Eb_meV_A2,Eb_meV_atom,dmin_A,relaxed\n")
                for m in r["modes"]:
                    for row in m["rows"]:
                        f.write(f"{m['label']},{row['gap']:.3f},{row['E']:.6f},"
                                f"{row['Eb_meV_A2']:.4f},{row['Eb_meV_atom']:.4f},"
                                f"{row['dmin']:.3f},0\n")
                    rl = m.get("relaxed")
                    if rl:
                        f.write(f"{m['label']}(relax),{m['gap_min']:.3f},"
                                f"{rl['E']:.6f},{rl['Eb_meV_A2']:.4f},"
                                f"{rl['Eb_meV_atom']:.4f},{rl['dmin']:.3f},1\n")
            r["csv"] = Path(a.csv).name
        except Exception as exc:
            print(f"[警告] CSV 写不出来：{exc}", flush=True)

    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"[完成] 结果写入 {a.out}  用时 {r['elapsed']}s", flush=True)
    print(json.dumps({"ok": True, "out": str(a.out), "elapsed": r["elapsed"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
