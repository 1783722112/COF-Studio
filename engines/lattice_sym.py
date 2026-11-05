# -*- coding: utf-8 -*-
"""把建出的二维层投影到它「应该有」的空间群（P4 / P6），让模型本身就对称。

为什么要单独一个模块
--------------------
`COF-Pipeline/symmetrize.py` 的思路是「问 spglib：这个结构里已经有什么对称性」，
再把原子投影过去。这条路对自由碱卟啉走不通：两个吡咯 N–H 只坐在两个 N 上，
spglib 看到的最多是 P2 甚至 P1，它永远不会告诉你「这该是 P4」。

本模块反过来 —— 由**晶格形状**决定「应该是什么」：

    a≈b, γ=90°   → nfold 4  → P4  (No.75)
    a≈b, γ=120°  → nfold 6  → P6  (No.174)
    其他          → 不投影（斜方/正交，留给 spglib 自己找）

然后自己构造目标操作集合 {R_k, t_k}，把原子投影进去。差别就是 C4/C6 能不能被
「想出来」，而不是「找出来」。

四个必须处理的坑（都是实测踩出来的）
------------------------------------
1. **轴心不唯一**。四方格子里 (0,0) 与 (½,½) 都是四重轴（P4 的 1a / 1b），
   解 (I−R)c = t 只给出其中一个。用「重原子残差最小」挑，不要用几何重心 ——
   hcb 胞里有两个节点 + 三个连接子，重心离节点中心 16 Å。
2. **自由碱卟啉的两个吡咯 N–H 在 C4 下无处可去**（偏差 ~0.93 Å）。这是化学
   事实不是数值误差，`nh` 参数提供四种交代方式。
3. **投影平均时每一项都要先做最小镜像**，否则贴着晶胞边界的原子会被平均到
   0 与 a 之间，位移假报十几埃。
4. **置换表的方向容易搞反**：要找的是「op_k 把原子 i 送到哪个原子」，
   即拿 q[i] 与所有 p 比，不是拿 q 与 p[i] 比。搞反了 180° 旋转会混进来。

作为模块
--------
    from lattice_sym import apply
    new, info = apply(atoms)                    # 默认 nh="keep"
    new, info = apply(atoms, nh="disorder")     # 吡咯 NH 转 1/2 占位
    new, info = apply(atoms, nh="metal", metal="Zn")

命令行（单文件）
----------------
    python lattice_sym.py in.cif -o out.cif [--nh keep|disorder|metal|bare]
"""
from __future__ import print_function

import numpy as np

NUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
       'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30}

TOL_ORPHAN = 0.55        # 轨道匹配容差（Å）：超过认为这个原子没有对称像
MAX_SHIFT = 0.80         # 允许的最大投影位移（Å）；再大就不是"投影"而是"掰弯"
AXIS_TOL = 0.35          # 重原子残差超过它，就认为这个阶数的旋转轴不存在


# ---------------------------------------------------------------- 向量工具
def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


def cellpar(cell):
    a, b, c = np.linalg.norm(cell, axis=1)
    al = np.degrees(np.arccos(np.clip(np.dot(cell[1], cell[2]) / b / c, -1, 1)))
    be = np.degrees(np.arccos(np.clip(np.dot(cell[0], cell[2]) / a / c, -1, 1)))
    ga = np.degrees(np.arccos(np.clip(np.dot(cell[0], cell[1]) / a / b, -1, 1)))
    return a, b, c, al, be, ga


def mic_vec(a, b, cell):
    """b − a 的最小镜像矢量（笛卡尔）。"""
    v = np.asarray(b, float) - np.asarray(a, float)
    f = v @ np.linalg.inv(cell)
    f -= np.round(f)
    return f @ cell


def mic_centroid(p, idx, cell, iters=60):
    """一组原子的最小镜像重心（直接算术平均会把跨边界的原子平均到中间去）。"""
    idx = list(idx)
    c = np.array(p[idx[0]], float)
    for _ in range(iters):
        acc = np.zeros(3)
        for i in idx:
            acc += mic_vec(c, p[i], cell)
        step = acc / len(idx)
        c = c + step
        if np.linalg.norm(step) < 1e-7:
            break
    return c


# ---------------------------------------------------------------- 群与操作
def lattice_nfold(cell, tol=0.02, ang_tol=1.5):
    """从晶胞形状推断层内旋转轴的阶数：4（四方）/ 6（六方）/ 1（不投影）。"""
    a, b, _, _, _, ga = cellpar(cell)
    if abs(a - b) > tol * a:
        return 1
    if abs(ga - 90.0) <= ang_tol:
        return 4
    if abs(ga - 120.0) <= ang_tol:
        return 6
    return 1


def group_ops(nfold, c):
    """绕通过 c=(cx,cy) 的 z 轴的 n 重循环群：op_k(x) = R_k x + t_k，t_k = c − R_k c。"""
    c = np.asarray(c, float)
    Rs, ts = [], []
    for k in range(nfold):
        R = rot_z(360.0 / nfold * k)
        Rs.append(R)
        t = np.array(c, float) - R @ c
        t[2] = 0.0
        ts.append(t)
    return Rs, ts


def axis_dev(p, sym, cell, c, nfold, mask=None):
    """n 重轴成立到什么精度：每个原子在其 op 下的最近同类原子距离，取最大。"""
    Rs, ts = group_ops(nfold, c)
    Minv = np.linalg.inv(cell)
    idx = np.arange(len(p)) if mask is None else np.where(mask)[0]
    worst = 0.0
    for k in range(1, nfold):
        q = (Rs[k] @ p[idx].T).T + ts[k]
        df = p[None, :, :] - q[:, None, :]
        df = df @ Minv
        df -= np.round(df)
        dist = np.linalg.norm(df @ cell, axis=2)
        for a_, i in enumerate(idx):
            cand = np.where(sym == sym[i])[0]
            worst = max(worst, float(dist[a_, cand].min()))
    return worst


def orbits(p, sym, cell, c, nfold, tol=TOL_ORPHAN):
    """perm[k][i] = op_k 把原子 i 送到了哪个原子；orphan = 无配对的原子。

    方向不能搞反：要找的是 q[i] = op_k(p_i) 落在哪个 p_j 上。
    """
    Rs, ts = group_ops(nfold, c)
    Minv = np.linalg.inv(cell)
    n = len(p)
    perm = np.zeros((nfold, n), dtype=int)
    orphan = {}
    for k in range(nfold):
        q = (Rs[k] @ p.T).T + ts[k]
        for i in range(n):
            df = p - q[i]
            df = df @ Minv
            df -= np.round(df)
            dist = np.linalg.norm(df @ cell, axis=1)
            cand = np.where(sym == sym[i])[0]
            j = int(cand[np.argmin(dist[cand])])
            perm[k, i] = j
            if dist[j] > tol:
                orphan.setdefault(i, []).append((k, j, float(dist[j])))
    return perm, orphan


def group_closed(perm, nfold, fixed=None):
    """置换表是否构成群作用：perm[k]∘perm[k2] == perm[(k+k2)%nfold]。"""
    n = perm.shape[1]
    test = np.arange(n) if fixed is None else np.where(~fixed)[0]
    for k in range(1, nfold):
        for k2 in range(1, nfold):
            kk = (k + k2) % nfold
            if np.any(perm[k][perm[k2][test]] != perm[kk][test]):
                return False
    return True


def refine_axis(p, sym, cell, c, nfold, perm, mask=None):
    """给定置换表，解轴心的修正量 δ（不是解绝对坐标）。

    为什么用增量形式：约束 R_k(p_i − c) + c = p_{P(i)}。直接解
    (I − R_k)c = p_{P(i)} − R_k p_i 时，跨边界的 p_{P(i)} 各自带着不同的
    整格平移，平均之后就偏了（实测把 (0,0) 拉到 (13, 12.6)）。
    改成残差形式：r_i = R(p_i − c) + c − mic_p(P(i))，它对最优 c* = c + δ 满足
    (I − R)δ = −r_i，右边的每一项都是小量，不会再带整格。
    """
    idx = np.arange(len(p)) if mask is None else np.where(mask)[0]
    A, b = [], []
    for k in range(1, nfold):
        R = rot_z(360.0 / nfold * k)
        acc = np.zeros(3)
        for i in idx:
            tgt = p[i] + mic_vec(p[i], p[perm[k, i]], cell)
            acc += (R @ (p[i] - c) + c) - tgt
        A.append(np.eye(3) - R)
        b.append(-acc / len(idx))
    delta, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    delta[2] = 0.0
    return np.asarray(c, float) + delta


def find_axis(p, sym, cell, nfold, verbose=False, rounds=5):
    """定 n 重轴：多候选 → 迭代精修 → 取重原子残差最小的那个。"""
    sym = np.asarray(sym)
    heavy = sym != 'H'
    hidx = np.where(heavy)[0]
    a, b = np.linalg.norm(cell[0]), np.linalg.norm(cell[1])

    cands = []
    cc = mic_centroid(p, hidx, cell)
    cands.append(np.array([cc[0], cc[1], 0.0]))

    # (I−R)c = t 的解：拿最少见的元素当锚点，逐个同种原子当「像」
    R1 = rot_z(360.0 / nfold)
    A1inv = np.linalg.pinv(np.eye(3) - R1)
    cnt = {e: int((sym == e).sum()) for e in set(sym.tolist())}
    rare = min(cnt, key=lambda e: cnt[e])
    pool = np.where(sym == rare)[0]
    i0 = int(pool[np.argmin(np.linalg.norm(p[pool] - p[pool].mean(axis=0), axis=1))])
    for j in np.where(sym == sym[i0])[0]:
        c0 = A1inv @ (p[j] - R1 @ p[i0])
        c0[2] = 0.0
        cands.append(np.array(c0))

    # 四方格里 (½,½) 也是四重轴（P4 的 1b 位置）；六方格里那是三重轴，不能加
    if nfold == 4:
        cands += [c0 + np.array([a / 2.0, b / 2.0, 0.0]) for c0 in list(cands)]

    best = None
    for c0 in cands:
        c = np.array(c0, float)
        # 初值本身也算一个候选：精修万一被错误的置换带跑，还能留住它
        dev0 = axis_dev(p, sym, cell, c, nfold, mask=heavy)
        if best is None or dev0 < best[0]:
            best = (dev0, c.copy())
        ok = True
        for _ in range(rounds):
            perm, _ = orbits(p, sym, cell, c, nfold, tol=1e9)    # 强制全配
            c_new = refine_axis(p, sym, cell, c, nfold, perm, mask=heavy)
            if np.linalg.norm(mic_vec(c, c_new, cell)) > 3.0:    # 跑飞了就弃
                ok = False
                break
            c = c_new
        if not ok:
            continue
        dev = axis_dev(p, sym, cell, c, nfold, mask=heavy)
        if dev < best[0]:
            best = (dev, c)

    if best is None:
        return np.array([0.0, 0.0, 0.0]), 1e9
    if verbose:
        print("  轴心 (%.4f, %.4f) Å，重原子 C%d 残差 %.4f Å"
              % (best[1][0], best[1][1], nfold, best[0]))
    return best[1], best[0]


# ---------------------------------------------------------------- 投影
def project(p, cell, Rs, ts, perm, fixed=None, iters=400):
    """投影到群的不动点集：y_i ← (1/|G|) Σ_k R_k⁻¹ (p_{perm[k][i]} − t_k)。

    fixed 里的原子原样保留（吡咯 NH 的 H 用它，见 apply 的说明）。
    每项都先做最小镜像拉回 p_i 附近 —— 否则跨边界的原子会被平均到
    y≈0 与 y≈a 之间，位移假报十几 Å。
    """
    Minv = np.linalg.inv(cell)
    invs = [np.linalg.inv(R) for R in Rs]
    cur = np.array(p, float)
    for _ in range(iters):
        acc = np.zeros_like(cur)
        for Rinv, t, P in zip(invs, ts, perm):
            term = (Rinv @ (cur[P] - t).T).T
            d = term - p
            f = d @ Minv
            f -= np.round(f)
            acc += p + f @ cell
        new = acc / len(Rs)
        d = new - p
        f = d @ Minv
        f -= np.round(f)
        new = p + f @ cell
        if fixed is not None:
            new[fixed] = p[fixed]
        step = new - cur
        sf = step @ Minv
        sf -= np.round(sf)
        cur = new
        if np.abs(sf @ cell).max() < 1e-11:
            break
    return cur


# ---------------------------------------------------------------- 挂单的 X–H
def find_xh(sym, d, cutoff=1.95, r_xh=1.25):
    """找「2 配位 sp2 杂原子上的 H」—— 吡咯 / 咪唑 / 酰胺型。

    这类 H 在 n 重轴（n≥3）下没有对称像：n 个 N 等价，但它只坐在其中两个上。
    返回 [(X_idx, H_idx)]。
    """
    sym = np.asarray(sym)
    out = []
    for i in np.where(np.isin(sym, ['N', 'O']))[0]:
        nb = [j for j in range(len(sym)) if j != i and d[i, j] < cutoff]
        hs = [j for j in nb if sym[j] == 'H' and d[i, j] < r_xh]
        heavy = [j for j in nb if sym[j] != 'H']
        if len(hs) == 1 and len(heavy) == 2:
            out.append((int(i), int(hs[0])))
    return out


# ---------------------------------------------------------------- 主入口
def apply(atoms, nfold=None, nh="keep", metal=None, verbose=True,
          tol=TOL_ORPHAN, max_shift=MAX_SHIFT, blen=1.01):
    """把 atoms 投影到晶格允许的 n 重轴空间群，返回 (新 atoms, info)。

    nh 决定那两个「挂单」的 X–H 怎么交代：

      keep      留在原位、且不参与投影（化学正确，MACE 可以直接弛豫）。
                骨架严格是 P4，但 spglib 会判 P2 —— 因为那 2 个 H 真的只坐在
                2 个 N 上，这是自由碱卟啉的化学事实，不是数值误差。
      disorder  H 按 1/2 占位摊到 n 个 X 上（化学式不变，晶体学惯例）。
                整个结构严格 P4，spglib 也判 P4。送 DFT 时程序若不认占位，
                会把它当成 n 个整 H（化学式多 2 个 H），务必注意。
      metal     删掉 H 并在环心插金属（D4h，无占位问题）。MACE-OFF 训练集
                不含 Zn/Co，这条路只能建模后直接交 DFT。
      bare      只删 H 不补（化学式不完整，仅用于看骨架）。

    info 字段：nfold / sg_in / sg_out / shift / shift_rms / nh_mode /
              nh_n / metal / axis / dev_before / dev_after / applied / note
    """
    cell = np.array(atoms.cell[:])
    p = atoms.get_positions()
    sym = np.asarray(atoms.get_chemical_symbols())
    n = len(p)

    info = dict(nfold=0, sg_in=None, sg_out=None, sg_skeleton=None,
                shift=0.0, shift_rms=0.0, nh_mode=nh, nh_n=0, metal=None,
                axis=None, dev_before=None, dev_after=None, applied=False, note="")
    fail = atoms.copy()

    nf = nfold if nfold else lattice_nfold(cell)
    if nf < 2:
        info["note"] = "晶格不是四方/六方（a≠b 或 γ 不是 90/120°），不做旋转轴投影"
        if verbose:
            print("[对称] " + info["note"])
        return fail, info

    # ---- 1. 定轴
    # 晶格是六方，结构本身可能只有三重轴：hcb 网里两个节点不等价时（TpPa 型
    # β-酮烯胺就是），C6 不存在但 C3 精确成立。这时 P-6 由 "C3 + 水平镜面"
    # 生成，投影到 C3 就够。所以按 6→3→1、4→2→1 的顺序降级试。
    c, dev0 = find_axis(p, sym, cell, nf, verbose=False)
    if nfold is None and dev0 > AXIS_TOL:
        for cand in ([3, 2] if nf == 6 else [2]):
            cc, dd = find_axis(p, sym, cell, cand, verbose=False)
            if dd <= AXIS_TOL:
                if verbose:
                    print("  晶格看着是 C%d，但结构只到 C%d（残差 %.3f → %.4f Å），按 C%d 投影"
                          % (nf, cand, dev0, dd, cand))
                nf, c, dev0 = cand, cc, dd
                break
        else:
            info["nfold"] = nf
            info["dev_before"] = round(float(dev0), 4)
            info["note"] = ("连 C2 都不成立（重原子残差 %.2f Å），结构没有可投影的旋转轴"
                            % dev0)
            if verbose:
                print("[对称][!] " + info["note"])
            return fail, info

    info["nfold"] = nf
    info["axis"] = (round(float(c[0]), 4), round(float(c[1]), 4))
    info["dev_before"] = round(float(dev0), 4)
    if verbose:
        print("  轴心 (%.4f, %.4f) Å，重原子 C%d 残差 %.4f Å"
              % (c[0], c[1], nf, dev0))

    # ---- 2. 找出「挂单」的 X–H，判断要不要摘掉
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)
    xh = find_xh(sym, d)
    info["nh_n"] = len(xh)

    perm_all, orph_all = orbits(p, sym, cell, c, nf, tol=tol)
    xh_h = {h for _, h in xh}
    orph_h = {i for i in orph_all if sym[i] == 'H'}
    removable = orph_h & xh_h                  # 只有这类 H 允许摘
    stubborn = set(orph_all) - removable       # 摘不掉的才是真问题

    if stubborn:
        worst = max(max(x[2] for x in orph_all[i]) for i in stubborn)
        info["note"] = ("%d 个原子在 C%d 下找不到配对（最大偏差 %.2f Å），"
                        "结构本身不是这个点群" % (len(stubborn), nf, worst))
        if verbose:
            print("[对称][!] " + info["note"])
            for i in sorted(stubborn)[:6]:
                print("        %s%d" % (sym[i], i))
        return fail, info

    drop = set(removable) if (removable and nh != "keep") else set()
    keep = np.array([i for i in range(n) if i not in drop])
    p2, sym2 = p[keep], sym[keep]
    idx_of = {int(i): int(j) for j, i in enumerate(keep)}
    h2x = {int(h): int(x) for x, h in xh}      # 每个挂单 H 挂在哪一个 X 上
    fixed_x_of = h2x

    # ---- 3. 置换表
    perm, orphan = orbits(p2, sym2, cell, c, nf, tol=tol)
    fixed = None
    if removable and nh == "keep":
        # 保留模式下把这 2 个 H 从对称化里摘出去：它们各自映射到自己，
        # 投影不动它们，N–H 键长得以保持；它们的"姐妹 N"照样对称。
        fixed = np.zeros(len(p2), dtype=bool)
        for i in removable:
            if i in idx_of:
                fixed[idx_of[i]] = True
        for i in np.where(fixed)[0]:
            perm[:, i] = i
        orphan = {i: v for i, v in orphan.items() if not fixed[i]}

    if orphan:
        worst = max(max(x[2] for x in v) for v in orphan.values())
        info["note"] = "仍有 %d 个原子无对称像（最大偏差 %.2f Å），放弃投影" % (
            len(orphan), worst)
        if verbose:
            print("[对称][!] " + info["note"])
        return fail, info

    if not group_closed(perm, nf, fixed):
        info["note"] = "置换表不构成群作用（结构偏离太远），放弃投影"
        if verbose:
            print("[对称][!] " + info["note"])
        return fail, info

    # ---- 4. 投影
    Rs, ts = group_ops(nf, c)
    p3 = project(p2, cell, Rs, ts, perm, fixed=fixed)
    shift = np.linalg.norm(p3 - p2, axis=1)
    smax, srms = float(shift.max()), float(np.sqrt((shift ** 2).mean()))
    info["shift"], info["shift_rms"] = round(smax, 4), round(srms, 5)

    if smax > max_shift:
        info["note"] = ("投影要挪 %.2f Å，超过 %.2f Å 的容忍线，判定为硬掰，保持原样"
                        % (smax, max_shift))
        if verbose:
            print("[对称][!] " + info["note"])
        return fail, info

    dev1 = axis_dev(p3, sym2, cell, c, nf, mask=(sym2 != 'H'))
    info["dev_after"] = round(float(dev1), 5)
    if verbose:
        print("  投影位移 max %.4f Å  mean %.5f Å；C%d 重原子残差 %.4f → %.5f Å"
              % (smax, srms, nf, dev0, dev1))

    # keep 模式下被冻住的 X–H：H 不参与投影，但要跟着它的 X 一起走，
    # 否则 X 挪了 0.02 Å 而 H 没动，N–H 键长就变了。
    if fixed is not None and fixed.any():
        for i_h in np.where(fixed)[0]:
            i_x = fixed_x_of.get(int(keep[i_h]))
            if i_x is None:
                continue
            j_x = idx_of[i_x]
            v = mic_vec(p2[j_x], p2[i_h], cell)      # 原始 N→H 矢量（长度 1.01）
            p3[i_h] = p3[j_x] + v

    # ---- 5. 补回 1/2 占位 H / 插金属
    sym3 = sym2.copy()
    occ = np.ones(len(p3))
    extra_p, extra_s, extra_o = [], [], []

    if removable and nh in ("disorder", "metal"):
        # 挑一个挂单 H 所属的 N，取它的 n 重轨道 —— 轨道成员必然铺满整组等价 N
        cand_n = [i for i, _ in xh if i in idx_of]
        i_n = idx_of[cand_n[0]]
        orb = [int(perm[k, i_n]) for k in range(nf)]

        if nh == "metal":
            mp = mic_centroid(p3, orb, cell)
            mp[2] = float(np.mean(p3[orb, 2]))
            extra_p.append(mp)
            extra_s.append(metal or "Zn")
            extra_o.append(1.0)
            info["metal"] = metal or "Zn"
        else:
            # N–H 方向：取原来那个 H 相对 N 的单位矢量（去掉投影带来的偏移），
            # 再随 C_n 转到每个 N 上
            h_raw = [h for n_, h in xh if n_ == cand_n[0]][0]
            v = mic_vec(p[cand_n[0]], p[h_raw], cell)
            if np.linalg.norm(v) < 0.6:
                v = np.array([0.0, 0.0, 1.0])
            v = v / np.linalg.norm(v)
            for k, jn in enumerate(orb):
                extra_p.append(p3[jn] + blen * (rot_z(360.0 / nf * k) @ v))
                extra_s.append('H')
                extra_o.append(0.5)
            info["nh_mode"] = "disorder"

    if extra_p:
        p3 = np.vstack([p3, np.array(extra_p)])
        sym3 = np.concatenate([sym3, np.array(extra_s)])
        occ = np.concatenate([occ, np.array(extra_o)])

    # ---- 6. 组装
    from ase import Atoms
    out = Atoms(symbols=list(sym3), positions=p3, cell=cell, pbc=True)
    out.set_array('occupancy', occ)
    out.wrap()
    info["applied"] = True
    if nh == "disorder" and removable:
        info["note"] = "吡咯 H 按 1/2 占位摊到 %d 个 N 上（化学式不变）" % nf
    elif nh == "metal" and removable:
        info["note"] = "已在环心插入 %s（D4h，无占位）" % info["metal"]
    elif removable:
        info["note"] = ("骨架 C%d 精确成立；%d 个吡咯 N–H 保留原位"
                        "（化学上必然的局部破缺，不参与对称化）" % (nf, len(removable)))
    else:
        info["note"] = "C%d 精确成立" % nf

    # ---- 7. 自检：spglib 怎么说
    try:
        import spglib
        inv = np.linalg.inv(cell)
        ds_in = spglib.get_symmetry_dataset(
            (cell, p @ inv, [NUM.get(x, 0) for x in sym]), symprec=1e-3)
        ds_out = spglib.get_symmetry_dataset(
            (cell, out.get_positions() @ inv,
             [NUM.get(x, 0) for x in out.get_chemical_symbols()]), symprec=1e-3)
        info["sg_in"] = ds_in.international if ds_in is not None else None
        info["sg_out"] = ds_out.international if ds_out is not None else None

        # 骨架空间群：把"挂单"的 X–H 摘掉再判。那 2 个 H 是化学事实（自由碱卟啉
        # 就是有两个 NH），但它们掩盖了骨架其实严格是 P4 这件事。
        info["sg_skeleton"] = info["sg_out"]
        if fixed is not None and fixed.any():
            m = ~fixed
            pw = out.get_positions()[:len(fixed)][m]
            nw = [NUM.get(x, 0) for x in np.asarray(sym3)[:len(fixed)][m]]
            ds_w = spglib.get_symmetry_dataset((cell, pw @ inv, nw), symprec=1e-3)
            if ds_w is not None:
                info["sg_skeleton"] = ds_w.international

        if verbose:
            sk = info["sg_skeleton"]
            extra = ("" if sk in (None, info["sg_out"])
                     else "；摘掉挂单 X–H 后骨架是 %s" % sk)
            print("  spglib(1e-3)：%s → %s%s"
                  % (info["sg_in"], info["sg_out"], extra))
    except ImportError:
        pass

    return out, info


# ---------------------------------------------------------------- 化学式
def formula(atoms):
    """带占位意识的化学式。

    `atoms.get_chemical_formula()` 不认 occupancy，--nh disorder 下 4 个 0.5 的 H
    会被算成 4 个整 H（C60H38N8 变成 C60H40N8）。这里按占位求和。
    """
    syms = np.asarray(atoms.get_chemical_symbols())
    occ = (atoms.get_array('occupancy') if 'occupancy' in atoms.arrays
           else np.ones(len(atoms)))
    tot = {e: float(occ[syms == e].sum()) for e in sorted(set(syms.tolist()))}
    order = [e for e in ('C', 'H') if e in tot]
    order += sorted(e for e in tot if e not in ('C', 'H'))
    out = ""
    for e in order:
        v = tot[e]
        if abs(v - round(v)) < 0.02:
            n = int(round(v))
            out += e if n == 1 else "%s%d" % (e, n)
        else:
            out += "%s%.2f" % (e, v)
    return out


def n_effective(atoms):
    """按占位折算的有效原子数。"""
    occ = (atoms.get_array('occupancy') if 'occupancy' in atoms.arrays
           else np.ones(len(atoms)))
    return float(occ.sum())


# ---------------------------------------------------------------- 写 CIF
def write_cif(path, atoms, title=None):
    """写 CIF，**保留 occupancy**。

    为什么不用 `ase.io.write`：ASE 的 CIF writer 把占位丢了，读回来 4 个 0.5 的
    H 就变成 4 个整 H —— 化学式从 C60H38N8 变成 C60H40N8，而且多出来的 2 个 H
    在 DFT 里没有任何意义。这里写成 P1 全胞 + occupancy 列：任何工具读进来
    原子数都对，spglib 也照样认得出 P4（它不看占位，只看位置）。
    """
    cell = np.array(atoms.cell[:])
    a, b, c, al, be, ga = cellpar(cell)
    syms = atoms.get_chemical_symbols()
    occ = (atoms.get_array('occupancy') if 'occupancy' in atoms.arrays
           else np.ones(len(atoms)))
    frac = atoms.get_scaled_positions()
    frac -= np.floor(frac)
    title = title or str(path).replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]

    L = ["data_" + title,
         "_symmetry_space_group_name_H-M   'P 1'",
         "_symmetry_Int_Tables_number      1",
         "_cell_length_a     %.6f" % a,
         "_cell_length_b     %.6f" % b,
         "_cell_length_c     %.6f" % c,
         "_cell_angle_alpha  %.4f" % al,
         "_cell_angle_beta   %.4f" % be,
         "_cell_angle_gamma  %.4f" % ga,
         "_cell_volume       %.4f" % abs(np.linalg.det(cell)),
         "_chemical_formula_sum  '%s'" % atoms.get_chemical_formula(),
         "",
         "loop_",
         " _symmetry_equiv_pos_site_id",
         " _symmetry_equiv_pos_as_xyz",
         "1 'x, y, z'",
         "",
         "loop_",
         " _atom_site_label",
         " _atom_site_type_symbol",
         " _atom_site_fract_x",
         " _atom_site_fract_y",
         " _atom_site_fract_z",
         " _atom_site_occupancy"]
    for i, e in enumerate(syms):
        L.append("%-6s %-3s %12.6f %12.6f %12.6f %6.3f"
                 % ("%s%d" % (e, i + 1), e, frac[i, 0], frac[i, 1], frac[i, 2], occ[i]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    from ase.io import read

    ap = argparse.ArgumentParser(description="把结构投影到晶格允许的空间群")
    ap.add_argument("cif")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--nfold", type=int, default=None, help="强制 4/6（默认按晶胞判断）")
    ap.add_argument("--nh", default="keep",
                    choices=["keep", "disorder", "metal", "bare"])
    ap.add_argument("--metal", default="Zn")
    ap.add_argument("--max-shift", type=float, default=MAX_SHIFT)
    a = ap.parse_args()

    at = read(a.cif)
    print("=" * 68)
    print(a.cif)
    new, info = apply(at, nfold=a.nfold, nh=a.nh, metal=a.metal,
                      max_shift=a.max_shift)
    print("  → %s" % info["note"])
    if a.out:
        write_cif(a.out, new)
        occ = new.get_array("occupancy")
        n_half = int((occ < 0.99).sum())
        extra = "，其中 %d 个 1/2 占位 H" % n_half if n_half else ""
        print("  写出 %s（%d 原子%s）" % (a.out, len(new), extra))


if __name__ == "__main__":
    main()
