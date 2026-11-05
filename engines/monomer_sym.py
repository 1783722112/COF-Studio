"""把单体投影到它自身的点群（n 重旋转 + 竖直镜面）。

为什么需要
----------
RDKit 的 ETKDG 嵌入 + MMFF 优化给出的是**某一个构象**，不是对称构象。
大环或长臂单体上，这个构象的对称破缺可以到 0.5~1.0 Å：

    卟啉 4 臂节点（缩合后）绕自身中心转 90°
        重原子平均位移 0.45 Å，最大 0.97 Å，47/52 个原子超 0.1 Å

拼成周期层之后，这个破缺会原样传下去，整层只能判成 P1 —— 再怎么做
后处理对称化都救不回来（要动 1 Å 以上，键长会烂）。

所以在 `prepare()` 出坐标之后、装配之前，先把单体投影回它自己的点群。
这一步只动坐标、不动拓扑，键长变化在 0.02 Å 量级。

只对重原子判定
--------------
自由碱卟啉的两个 NH 是反式的，点群只有 D2h，转 90° 会把 N-H 搬到没有氢的
氮上。但那是**氢的层次**的事，骨架本身仍然是 D4h。所以：
  · 判断某个操作成不成立 → 只看重原子；
  · 投影 → 只投影重原子；
  · 氢 → 跟着母原子一起走（保持局部键长键角）。
这样骨架被拉到严格 C4，NH 再按反式摆回去，结构既对称又化学上正确。
"""

import numpy as np

# 判定操作成立的最大残差（Å）。单体构象破缺实测 0.5~1.0 Å，取 1.4 留余量；
# 太大了会误判（把 C2 当 C4 也会「通过」），下面还有逐点复核。
VERIFY_TOL = 1.4
# 投影后允许的最大累计位移（Å）。超过说明这个操作本来就不该有，放弃。
MAX_SHIFT = 1.2


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mirror_vert(deg):
    """含 z 轴、沿方向 (cos, sin) 的镜面。"""
    a = np.radians(deg)
    c, s = np.cos(2 * a), np.sin(2 * a)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])


def _match(pos, sym, R, t, pool):
    """按 R,t 变换后，同元素最近匹配；返回 (perm, 最大残差)。"""
    y = pos @ R.T + t
    perm = np.arange(len(pos))
    worst = 0.0
    for e in set(sym[i] for i in pool):
        idx = [i for i in pool if sym[i] == e]
        sub = pos[idx]
        for i in idx:
            d = np.linalg.norm(sub - y[i], axis=1)
            k = int(np.argmin(d))
            perm[i] = idx[k]
            worst = max(worst, float(d[k]))
    return perm, worst


def _apply_op(R, t, perm, pos, pool, n_atoms):
    """正交投影一步：y_i = R^-1 (y_{P(i)} - t)，带最小镜像无关（分子内无周期）。"""
    out = np.array(pos, dtype=float)
    for i in pool:
        out[i] = R.T @ (pos[perm[i]] - t)
    return out


def detect_ops(pos, sym, n, heavy, verbose=False, conn=None, mirror=True):
    """找出这个单体真正具有的 n 重旋转（以及可能的竖直镜面）。

    坐标系必须是 `monomer_frame` 给出的标准系（长轴 x、面法向 z），
    否则「绕 z 转 360/n」对线形连接子是没有意义的。

    旋转中心必须用 `_best_center` 解出来的那个，**不能用重原子质心**。
    质心只在分子质量分布对称时才落在轴上；一旦某条臂上挂着取向不一的
    侧基（羧基的 =O、醛基的 =O），质心会偏轴零点几埃，于是"看起来"有
    0.5 Å 的残差，却仍被 1.4 Å 的松阈值放行 —— 拿这个操作去投影，等于
    把侧基当噪声平均掉：实测 C=O 从 1.22 Å 压到 0.78 Å，整层报原子重叠。

    mirror=False 时不检测竖直镜面。投影**只该用纯旋转**：竖直镜面不是
    二维层装配所必需的（σh 由压平自动保证），而拿一个近似镜面去平均，
    会把层平面内的手性构象直接抹平。
    """
    heavy = list(heavy)
    c, _ = _best_center(pos, sym, heavy, n, conn=conn)
    R1 = _rot_std(360.0 / n, n)
    ops = []
    for k in range(1, n):
        R = np.linalg.matrix_power(R1, k)
        t = c - R @ c
        perm, worst = _match(pos, sym, R, t, heavy)
        ok = worst < VERIFY_TOL
        if verbose:
            print(f"      C{n} 第 {k} 步: 重原子最大残差 {worst:.3f} Å  "
                  f"{'✓' if ok else '✗'}")
        if ok:
            ops.append((R, t))

    # 竖直镜面：试各条臂的方位角，以及它们之间的平分线。
    # 只对星形节点有意义——线形连接子的镜面是分子平面本身（压平后自动成立）。
    if mirror and n >= 3:
        v = pos[heavy] - c
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        cand = sorted({round(a % 180.0, 3) for a in az})
        cand += [round((cand[i] + cand[(i + 1) % len(cand)]) / 2.0 % 180.0, 3)
                 for i in range(len(cand))] if cand else []
        kept = []
        for a in cand:
            R = _mirror_vert(a)
            t = c - R @ c
            perm, worst = _match(pos, sym, R, t, heavy)
            if worst < VERIFY_TOL * 0.8:
                kept.append((R, t, a, worst))
        if kept:
            kept.sort(key=lambda x: x[3])
            ops.append((kept[0][0], kept[0][1]))
            if verbose:
                print(f"      镜面 σ(φ={kept[0][2]:.2f}°): 残差 {kept[0][3]:.3f} Å ✓")
    return ops, c


def _best_center(pos, sym, heavy, n, conn=None, iters=8):
    """估对称中心：先用连接点几何中心（装配就围绕它转），再用最小二乘微调。

    质心不等于对称中心——破缺时质心会偏，把它当中心量出来的残差不可信。
    微调解 (I - R) c = p' - R p，但**只在残差下降时接受**：初始匹配一旦
    错配，逐次迭代会发散（实测能飞出 40 Å），所以要留最优解兜底。

    只在垂直于转轴的平面里解（n≥3 是 xy，n=2 是 xz）——沿轴方向 R c = c，
    那一维本来就没有约束。
    """
    R = _rot_std(360.0 / n, n)
    if conn is not None and len(conn) == n:
        c = pos[list(conn)].mean(axis=0).copy()
    else:
        c = pos[heavy].mean(axis=0).copy()

    idx = [0, 1] if n >= 3 else [0, 2]
    A = (np.eye(3) - R)[np.ix_(idx, idx)]
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return c, None

    best, best_worst = c.copy(), None
    for _ in range(iters):
        t = c - R @ c
        perm, worst = _match(pos, sym, R, t, heavy)
        if best_worst is None or worst < best_worst:
            best, best_worst = c.copy(), worst
        rhs = (pos[[perm[i] for i in heavy]] - pos[heavy] @ R.T)[:, idx].mean(axis=0)
        dc = Ainv @ rhs
        if np.linalg.norm(dc) > 2.0:      # 稳健性闸门：不许一步跳太远
            break
        c = c.copy()
        c[idx] += dc
        if dc @ dc < 1e-12:
            break
    return best, best_worst


def monomer_frame(pos, heavy, conn, n):
    """把单体转到「长轴 = x、面法向 = z」的标准系，返回 3x3 正交矩阵 M。

    `prepare()` 出来的朝向是随机的：线形连接子常常**立在** xz 平面里（长轴
    x、环面法向 y），此时还在 xy 平面里绕 z 转就完全测错了。

    旋转轴也随之而定：
      · n ≥ 3（星形节点）→ 绕面法向（标准系 z），因为连接点共面；
      · n == 2（线形连接子）→ 绕面内垂直于长轴的轴（标准系 y）。两端互换
        是 180° 转，转轴必须落在环平面里，绕面法向转只会把环翻个面。
    """
    pos = np.asarray(pos, dtype=float)
    if n >= 3 and conn is not None and len(conn) == n:
        pts = pos[list(conn)]
    else:
        pts = pos[list(heavy)]
    pts = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)
    e1, e3 = vh[0], vh[2]          # 最大方差 = 主轴；最小方差 = 面法向
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, e3)
    e1 /= np.linalg.norm(e1)
    return np.array([e1, e2, e3])


def _rot_std(deg, n):
    """标准系里的对称操作：n≥3 绕 z，n=2 绕 y。"""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if n >= 3:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def check_monomer(sym, pos, n, conn=None, label="单体"):
    """体检：这个单体自身带不带 n 重对称、连接点是不是等间距排列。

    **只读，不改坐标。** 建模一开始就跑一遍，不要等到整层 spglib 判 P1
    才回头怀疑单体（破缺到 1 Å 时，事后晶体对称化已经救不回来了）。

    返回 dict：
        rot_max / rot_mean   绕自身轴转 2πk/n 时重原子的最大/平均位移
        n_step               第几步失败（1-based，0 表示全部通过）
        radius / r_spread    连接点到中心的距离及其极差
        az / az_dev          连接点方位角，以及相对理想等间隔的最大偏差
        h_max                氢原子的最大位移（仅供参考，不参与判定）
        level                OK / WARN / FAIL
    """
    pos = np.asarray(pos, dtype=float)
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    out = {"n": n, "label": label, "rot_max": 0.0, "rot_mean": 0.0,
           "n_step": 0, "radius": [], "r_spread": 0.0,
           "az": [], "az_dev": 0.0, "h_max": 0.0, "level": "OK"}

    # 转到单体自己的坐标系：长轴 = x、面法向 = z
    M = monomer_frame(pos, heavy, conn, n)
    P = pos @ M.T
    c, _ = _best_center(P, sym, heavy, n, conn=conn)
    out["center"] = c
    out["frame"] = M

    # 每一步旋转都要复核：C4 通过不代表 C2 通过
    all_d = []
    for k in range(1, n):
        R = _rot_std(360.0 * k / n, n)
        t = c - R @ c
        perm, worst = _match(P, sym, R, t, heavy)
        all_d.append(worst)
        if worst > out["rot_max"]:
            out["rot_max"] = worst
        if worst > 0.05 and out["n_step"] == 0:
            out["n_step"] = k
    out["rot_mean"] = float(np.mean(all_d)) if all_d else 0.0

    # 氢：同上，只作参考
    hidx = [i for i, s in enumerate(sym) if s == "H"]
    if hidx:
        hm = 0.0
        for k in range(1, n):
            R = _rot_std(360.0 * k / n, n)
            t = c - R @ c
            for i in hidx:
                d = np.linalg.norm(P[hidx] - (R @ (P[i] - c) + c), axis=1)
                hm = max(hm, float(d.min()))
        out["h_max"] = hm

    # 连接点：到轴的距离必须相等、方位角必须等间隔，否则装配后网格必然歪
    if conn is not None and len(conn) == n:
        v = P[list(conn)] - c
        r = np.linalg.norm(v[:, :2], axis=1)
        out["radius"] = r.tolist()
        out["r_spread"] = float(r.max() - r.min())
        az = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0)
        out["az"] = az.tolist()
        gaps = np.diff(np.append(az, az[0] + 360.0))
        out["az_dev"] = float(np.abs(gaps - 360.0 / n).max())

    # 分级。0.05 Å 是「骨架严格对称」的常规判据（约等于 X 射线分辨率极限）
    if out["rot_max"] <= 0.05 and out["r_spread"] <= 0.02:
        out["level"] = "OK"
    elif out["rot_max"] <= 0.30:
        out["level"] = "WARN"
    else:
        out["level"] = "FAIL"
    return out


def format_check(res, prefix=""):
    """把 check_monomer 的结果排成一行，直接进构建日志。"""
    r = res
    txt = (f"{prefix}{r['label']}自检: C{r['n']} 绕轴残差 max {r['rot_max']*1000:.0f} mÅ "
           f"(mean {r['rot_mean']*1000:.0f})")
    if r["n_step"]:
        txt += f" 第 {r['n_step']}/{r['n']} 步就破缺"
    if r["radius"]:
        txt += f" | 连接点半径 {np.mean(r['radius']):.3f} Å (极差 {r['r_spread']*1000:.0f} mÅ)"
        txt += f" | 方位角间隔偏差 {r['az_dev']:.2f}°"
    tag = {"OK": "对称", "WARN": "轻微破缺", "FAIL": "明显破缺"}[r["level"]]
    return f"[{r['level']}] {txt} → {tag}"


def _rebuild_h(pos, sym, heavy, h_parent, fixed_heavy):
    """重原子动完之后，让每个氢跟着母原子的局部坐标系一起转。"""
    pos = np.array(pos, dtype=float)
    # 每个重原子的重原子邻居
    nb = {i: [] for i in heavy}
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    for i in heavy:
        for j in heavy:
            if i != j and d[i, j] < 1.75:
                nb[i].append(j)
    for h, p in h_parent.items():
        old_v = pos[h] - pos[p]
        old_n = nb[p]
        if len(old_n) < 2:
            continue
        # 母原子局部坐标系（由两个重原子邻居 + 法向确定）
        e1 = _unit(pos[old_n[0]] - pos[p])
        w = pos[old_n[1]] - pos[p]
        e2 = _unit(w - w.dot(e1) * e1)
        e3 = np.cross(e1, e2)
        Fo = np.array([e1, e2, e3])
        # 用**投影前**的重原子位置算旧坐标系，投影后的算新坐标系
        e1n = _unit(fixed_heavy[old_n[0]] - fixed_heavy[p])
        w = fixed_heavy[old_n[1]] - fixed_heavy[p]
        e2n = _unit(w - w.dot(e1n) * e1n)
        e3n = np.cross(e1n, e2n)
        Fn = np.array([e1n, e2n, e3n])
        R = Fn.T @ Fo
        pos[h] = fixed_heavy[p] + R @ old_v
    return pos


def flatten_heavy(pos, sym):
    """把重原子压到最佳拟合平面上（沿法向平移，不改面内坐标）。

    投影只保证"对称等价原子同高"，保证不了"整体平"：卟啉核心可以对称地
    呈鞍形（C4 保持、但四个吡咯交替上下）。二维 COF 要求整层平，所以对称
    化之后必须再压一次面。
    """
    pos = np.array(pos, dtype=float)
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    c = pos[heavy].mean(axis=0)
    _, _, vh = np.linalg.svd(pos[heavy] - c)
    n = vh[2]
    for i in heavy:
        pos[i] -= np.dot(pos[i] - c, n) * n
    return pos


def symmetrize_monomer(sym, pos, n, verbose=False, conn=None):
    """把单体投影到 n 重点群。返回 (新坐标, 最大位移, 用到的操作数)。

    找不到任何操作时原样返回 (pos, 0.0, 0)。

    内部先转到单体自己的标准系（长轴 x、面法向 z）再投影，最后转回原坐标系：
    `prepare()` 出来的朝向是任意的，直接在原坐标里绕 z 转是错的。

    conn: 连接原子索引。给了就用**连接点所在平面**定标准系的 z 轴
    （n≥3 时连接点共面，这就是真正的 n 重轴）；不给就用全部重原子的
    PCA 平面。这个区别对卟啉很关键：meso 苯环是垂直于层扭转的，重原子
    PCA 的法向会被这些苯环带偏几度，投影出来的 C4 轴跟着偏，装配后
    （`node_frame` 按连接点定平面）四条臂就落在不同高度，整层又没了 C4。
    """
    pos = np.array(pos, dtype=float)
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    if len(heavy) < 3 or n < 2:
        return pos, 0.0, 0

    if conn is not None:
        conn = [int(c) for c in conn]
        if len(conn) != n:
            conn = None
    M = monomer_frame(pos, heavy, conn, n)
    P = pos @ M.T

    ops, c = detect_ops(P, sym, n, heavy, verbose=verbose,
                        conn=conn, mirror=False)
    if not ops:
        if verbose:
            print("      [跳过] 没找到可用的对称操作")
        return pos, 0.0, 0

    start = P.copy()
    work = P.copy()
    for it in range(60):
        base = work.copy()
        acc = base.copy()                      # 恒等操作的贡献先放进去
        for R, t in ops:
            perm, worst = _match(base, sym, R, t, heavy)
            for i in heavy:
                acc[i] += R.T @ (base[perm[i]] - t)
        new = base.copy()
        for i in heavy:
            new[i] = acc[i] / (len(ops) + 1)   # 含恒等操作
        moved = np.linalg.norm(new[heavy] - base[heavy], axis=1).max()
        work = new
        if moved < 1e-5:
            break

    shift = float(np.linalg.norm(work[heavy] - start[heavy], axis=1).max())
    if shift > MAX_SHIFT:
        if verbose:
            print(f"      [跳过] 需要位移 {shift:.2f} Å，超出 {MAX_SHIFT} Å，"
                  f"判定为操作不成立")
        return pos, 0.0, 0

    # 氢跟着母原子走
    h_parent = {}
    d = np.linalg.norm(start[:, None, :] - start[None, :, :], axis=-1)
    for h, s in enumerate(sym):
        if s != "H":
            continue
        for i in heavy:
            if d[h, i] < 1.25:
                h_parent[h] = i
                break
    out = _rebuild_h(start, sym, heavy, h_parent, work)
    for i in heavy:
        out[i] = work[i]

    out = out @ M                              # 标准系 → 原坐标系（M 正交）

    # 键长守卫：投影是"把不等价的原子平均到一起"，一旦某个基团在几条臂上
    # 取向本来就不同，平均出来的不是对称结构而是坏键。这里正面拦一道：
    # 任何一根键的长度变化超过 0.10 Å，就认定这次投影不成立，原样返回。
    bad = _bond_drift(pos, out)
    if bad is not None:
        if verbose:
            print(f"      [跳过] 投影会把 {bad[0]}-{bad[1]} 键从 {bad[2]:.3f} Å "
                  f"改成 {bad[3]:.3f} Å —— 各臂侧基取向不一致，先统一构象")
        return pos, 0.0, 0

    if verbose:
        print(f"      [对称化] 用 {len(ops)} 个操作，重原子最大位移 {shift*1000:.0f} mÅ")
    return out, shift, len(ops)


def _bond_drift(p0, p1, tol=0.10, cut=1.90):
    """投影前后逐键比较长度，返回第一根超差的 (i, j, 旧长, 新长)，全合格返回 None。"""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n = len(p0)
    d0 = np.linalg.norm(p0[:, None, :] - p0[None, :, :], axis=-1)
    d1 = np.linalg.norm(p1[:, None, :] - p1[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    sel = d0[iu] < cut
    if not sel.any():
        return None
    a = d0[iu][sel]
    b = d1[iu][sel]
    k = int(np.argmax(np.abs(b - a)))
    if abs(b[k] - a[k]) <= tol:
        return None
    return (int(iu[0][sel][k]), int(iu[1][sel][k]), float(a[k]), float(b[k]))
