# -*- coding: utf-8 -*-
"""
build_cof_smiles.py
===================
通用 COF 搭建模板：输入两个单体的 SMILES，输出二维 COF 单层的周期 CIF。

支持拓扑
--------
    hcb   蜂窝：3 连接节点 + 线形连接子，γ = 120°，a = b = L·√3   （LZU-1 型）
    sql   四方：4 连接节点 + 线形连接子，γ =  90°，a = b = L       （卟啉 COF 型）

节点 SMILES 里有几个 ``*`` 就自动判成几连接（3 → hcb，4 → sql），
也可以用 ``--topology`` 手动指定。

**也可以完全不写 ``*``**：只给单体本身的分子 SMILES，脚本自己判断
哪些原子是 sp2、能当连接位，再挑最对称的一组（3 臂 120°、4 臂 90°/菱形、
2 臂对位、半径一致），并生成等价的带 ``*`` SMILES。这样就不用背
``[*][CH]c1nc([CH][*])nc([CH][*])n1`` 这种写法，也不会误把 CH2 当连接位。

    python build_cof_smiles.py --node "c1ncncn1" --arms 3 \
        --linker "c1ccc(C#Cc2ccc(C#Cc3ccccc3)cc2)cc1" --bond 1.47
    python build_cof_smiles.py --inspect "CC1=NC=NC=N1"    # 体检：每个原子几个氢

注意：``*`` 后面的 C 是一个**真实碳原子**，不是占位符。写成 ``[*]CC1=...``
等于在星号和环之间多加了一个 CH2（RDKit 会自动补 H），接出来的层不可能共面。

约定（很重要）
--------------
1. 单体要写成"缩合反应之后"的样子，即连接原子上已经是最终化学环境；
2. 用 ``*`` 标记连接位点，脚本会删除 ``*``，再把两个连接原子按 --bond 键长对接；
3. 用方括号显式写氢数，例如 [CH]、[NH]，否则 RDKit 会按价态自动补满，
   导致多出/缺少一个 H（这是最常见的错误来源）。
4. 芳香性/氢数写错（比如卟啉写成 4 个 [nH]）会让环严重翘曲、甚至压平后键长
   崩坏；脚本会打印"出平面起伏"警告，看到就回去检查 SMILES。

示例：三嗪-乙烯基-二乙炔 COF（hcb）
    node   = "*[CH]c1nc([CH]*)nc([CH]*)n1"
    linker = "*[CH]c1ccc(C#CC#Cc2ccc([CH]*)cc2)cc1"
    bond   = 1.34    (新生成的 C=C)

示例：亚胺 COF（三醛基苯 + 对苯二胺，LZU-1 型，hcb）
    node   = "*[CH]c1cc([CH]*)cc([CH]*)c1"    # 醛碳缩合后保留 1 个 H
    linker = "*[N]c1ccc([N]*)cc1"             # 注意: 亚胺 N 上没有 H！
    bond   = 1.28    (C=N)

用法
----
    python build_cof_smiles.py --preset triazine                  # 零参数也行
    python build_cof_smiles.py --node "..." --linker "..." --bond 1.34 --out cof.cif
"""

import argparse
import os
from itertools import combinations, product

import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms


# ---------------------------------------------------------------- 单体对称性
# 单体自检（默认开）：只测不改。每次 prepare() 打一行 `[单体] ...`，
# 记录这个单体自身带不带 C_n、连接点是否等距。单体不对称 → 装配出来的
# 整层不可能高对称，这是整条流水线里最早能拦住的地方。
#     COF_CHECK_MONOMER=0 python cof_builder.py ...     关掉
CHECK_MONOMER = os.environ.get("COF_CHECK_MONOMER", "1") == "1"
MONOMER_CHECKS = []          # 本进程内所有单体的体检结果，上层可读取

# 单体点群对称化（默认开）：自检不达标时把骨架投影回 C_n 再装配。
# 关掉就退回旧行为——构象带多少破缺，整层就有多少破缺。
#     COF_SYMMETRIZE_MONOMER=0 python cof_builder.py ...  关掉
SYMMETRIZE_MONOMER = os.environ.get("COF_SYMMETRIZE_MONOMER", "1") == "1"


# ---------------------------------------------------------------- 向量工具
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def rot_from_to(a, b):
    """把单位向量 a 转到 b 的旋转矩阵（Rodrigues）。"""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rot_z(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(u, deg):
    """绕单位轴 u 旋转 deg 度（Rodrigues 公式）。"""
    u = unit(u)
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    ux, uy, uz = u
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(u, u)


def flatten_rings(mol, pos, axis=None, cap=(), normal=None):
    """把每个芳环绕其 ipso–para 轴刚性旋转，使环落在目标平面里。

    只做刚体旋转，键长/键角完全不变；这样二维 COF 的层厚度才不会被
    扭转的苯环撑大（否则 c=3.6 Å 装不下一层，折回晶胞后看着像两层）。

    两种用法：

      · ``axis``：给一条"分子长轴"，环被转成平面包含该轴（线形连接子用）；
      · ``normal``：直接给目标平面法向，环被转成与该法向垂直（C3/C4 星形
        节点用——三条臂没有共同长轴，必须用节点自身所在平面的法向）。

    cap: 封端 H 的原子索引集合。它们虽然不是重原子，但代表真实的外伸键
    （连接位点），必须算作环外取代基，否则像二苯乙炔这种两端都以连接位
    收尾的环会被判成"只有一个取代基"而漏掉不转。
    """
    cap = set(cap)
    ri = mol.GetRingInfo()
    if not ri.AtomRings():
        Chem.GetSymmSSSR(mol)          # AddHs / RemoveAtom 之后环信息可能失效
        ri = mol.GetRingInfo()
    if normal is None:
        u = unit(axis)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, u))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        n_target = unit(np.cross(u, ref))
    else:
        n_target = unit(normal)

    def is_sub(nb, ring_set):
        if nb.GetIdx() in ring_set:
            return False
        return nb.GetAtomicNum() > 1 or nb.GetIdx() in cap

    moved = 0
    for ring in ri.AtomRings():
        # 五元环也必须处理：噻吩 / 呋喃 / 吡咯类连接子和节点很常见，
        # 漏掉它们会让整条臂保持扭转构象，单层厚度被撑到 3~4 Å。
        # 判据不是环大小，而是"环外取代基正好两个"——那样才有一条
        # 穿过两个连接原子的转轴可用（2,5-二取代噻吩同样成立）。
        if len(ring) not in (5, 6):
            continue
        ring_set = set(ring)
        # 只统计"非氢"的环外邻居，否则补氢后每个环碳都算有取代基
        subs = [i for i in ring
                if any(is_sub(nb, ring_set)
                       for nb in mol.GetAtomWithIdx(i).GetNeighbors())]
        if len(subs) != 2:
            continue
        a, b = subs[0], subs[1]
        pa = pos[a]
        u_ring = unit(pos[b] - pa)

        # 环法向：用两条相邻环键的叉积（稳健；对位轴上的原子不能用来定平面）
        order = list(ring)
        e1 = pos[order[1]] - pos[order[0]]
        e2 = pos[order[2]] - pos[order[1]]
        n_cur = np.cross(e1, e2)
        if np.linalg.norm(n_cur) < 1e-6:
            continue
        n_cur = unit(n_cur - np.dot(n_cur, u_ring) * u_ring)
        nt = unit(n_target - np.dot(n_target, u_ring) * u_ring)
        ang = np.degrees(np.arctan2(float(np.dot(np.cross(n_cur, nt), u_ring)),
                                    float(np.dot(n_cur, nt))))
        R = rot_axis(u_ring, ang)
        # 环碳和它挂的氢必须一起转，否则氢会留在原来的平面外
        moving = list(ring)
        for i in ring:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    moving.append(nb.GetIdx())
        for i in moving:
            pos[i] = pa + R @ (pos[i] - pa)
        moved += 1
    return moved


def _side_size(mol, start, block):
    """剪断 start–block 这根键后，start 那一侧有多少个原子（含氢）。"""
    seen, stack, n = {block}, [start], 0
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        n += 1
        for nb in mol.GetAtomWithIdx(k).GetNeighbors():
            if nb.GetIdx() not in seen:
                stack.append(nb.GetIdx())
    return n


def flatten_torsions(mol, pos, tol=1.0, max_iter=3):
    """把所有可旋转单键的二面角拧到 0°/180°，让共轭骨架真正共面。

    为什么光靠 `flatten_rings` 不够
    ------------------------------
    `flatten_rings` 是"绕环的两个取代基连线把环转平"。对位二取代的六元环
    恰好成立：两个外伸键与转轴共线，转环不会动到外伸键。

    但噻吩/呋喃/吡咯这类五元环不成立——2,5-二取代噻吩的两根外伸键是**收敛**
    的，不共线。绕 C2–C5 轴把环转平以后，两侧苯环还停在原处，于是环与苯环
    之间出现 20~35° 的二面角，单层厚度反而被撑到 2 Å 以上。

    所以先做这一步：遍历所有"非环单键"，把二面角拧到最接近的 0° 或 180°。
    每一步都是绕该键的刚体转动，键长/键角一点也不变。拧完再交给
    `flatten_rings` 做整体对轴，二者叠加才得到严格平面的单体。
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    conf = work.GetConformer()

    ri = work.GetRingInfo()
    rot = []
    for b in work.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if ri.NumBondRings(b.GetIdx()) > 0:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if work.GetAtomWithIdx(i).GetAtomicNum() == 1 or \
           work.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        # 键两端各取一个环外邻居来定义二面角，优先取重原子
        def outer(k, skip):
            cand = [nb.GetIdx() for nb in work.GetAtomWithIdx(k).GetNeighbors()
                    if nb.GetIdx() != skip]
            if not cand:
                return None
            heavy = [c for c in cand if work.GetAtomWithIdx(c).GetAtomicNum() > 1]
            return (heavy or cand)[0]
        a, c = outer(i, j), outer(j, i)
        if a is None or c is None:
            continue
        # 炔键两侧的 sp 碳是直线型的：a–i–j 或 i–j–c 一旦接近 180°，
        # 这个二面角就是退化的（三点共线，转多少度都"合法"）。
        # 强行把它拧到 0°/180° 等于把整条臂绕轴乱转 —— 实测一个含 3 条
        # 二乙炔臂的单体，离面量从 0.74 Å 恶化到 1.31 Å。直接跳过。
        def ang(x, y, z):
            u = pos[x] - pos[y]
            v = pos[z] - pos[y]
            cosv = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
        if ang(a, i, j) > 165.0 or ang(i, j, c) > 165.0:
            continue
        # SetDihedralDeg 只转**末端原子 l 所在的半侧**。若这个半侧是分子主体，
        # 一次设定就把整个分子拧翻过去（末端乙烯基的二面角会把另外 60 个原子
        # 一起转走，离面量反而从 0.74 涨到 1.37）。所以按两侧原子数排序，
        # 永远让小的半侧去转。
        if _side_size(work, i, j) < _side_size(work, j, i):
            rot.append((c, j, i, a))        # 小侧在 i 端 → 末端写 a
        else:
            rot.append((a, i, j, c))        # 小侧在 j 端 → 末端写 c

    moved = 0
    for _ in range(max_iter):
        changed = 0
        for a, i, j, c in rot:
            try:
                cur = rdMolTransforms.GetDihedralDeg(conf, a, i, j, c)
            except Exception:
                continue
            tgt = 0.0 if abs(cur) <= 90.0 else 180.0
            d = cur - tgt
            if abs(d) < tol:
                continue
            try:
                rdMolTransforms.SetDihedralDeg(conf, a, i, j, c, tgt)
                changed += 1
            except Exception:
                pass
        moved += changed
        if changed == 0:
            break
    pos[:] = np.array(conf.GetPositions())
    return moved


def _rotor_bond(mol, c):
    """连接原子 c 的"末端转子"键，返回 (sub_root, anchor)。

    从 c 沿主链往里走，直到撞上第一个环原子 anchor；转子键就是
    anchor–sub_root 这一根。绕它转 180° 能把末端基团翻到臂轴的另一侧，
    键长键角全不变。

    例：Ar–CH=CH2 的末端连接原子是外侧那个 =CH2 碳，转子键就是 Ar–CH=
    （绕 Ar–C 单键转 180°，末端碳从臂轴外翻到轴内侧，半径差 0.38 Å）。

    连接原子本身就在环上（三嗪、卟啉、芘）时没有这种转子，返回 None。
    走到分叉点（例如间位取代的苯环）也返回 None —— 翻哪一边有歧义。
    """
    ring = set()
    for r in mol.GetRingInfo().AtomRings():
        ring.update(r)
    if c in ring:
        return None
    prev, cur = None, c
    for _ in range(60):
        if cur in ring:
            if prev is None:
                return None
            b = mol.GetBondBetweenAtoms(prev, cur)
            # 只认单键。末端 =CH– 与环之间如果是双键（TFP 的 =C1C(=O)...），
            # 绕它转 180° 等于把顺式烯烃翻成反式，是化学上不允许的操作。
            if b is None or b.GetBondType() != Chem.BondType.SINGLE:
                return None
            return (prev, cur)
        nb = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
              if n.GetIdx() != prev and n.GetAtomicNum() > 1]
        if len(nb) != 1:
            # 末端基团在连接原子处分叉：羧基 -C(=O)OH 的碳同时连着环碳和氧，
            # 按"分叉就有歧义"处理会直接返回 None，于是这类单体永远没有可翻
            # 的转子，三条臂上 -C(=O) 的取向只能各翻各的（实测同一单体里两条
            # 臂在 −58°、一条在 +58°，镜像关系）。后果不是"稍微不对称"：
            # 后面 `symmetrize_monomer` 把不等价的侧基一起平均，C=O 从
            # 1.22 Å 压到 0.78 Å，整层报原子重叠。
            # 只要"能通到环的邻居恰好一个、其余都是只连这一个原子的端基"，
            # 转子就是唯一的 cur–环侧邻居，翻它等于把整个末端基团翻过去。
            if prev is None and nb:
                to_ring = [x for x in nb if x in ring]
                ends = [y for y in nb if y not in to_ring]
                if len(to_ring) == 1 and all(
                        len([m for m in mol.GetAtomWithIdx(y).GetNeighbors()
                             if m.GetAtomicNum() > 1]) == 1 for y in ends):
                    prev, cur = cur, to_ring[0]
                    continue
            return None
        prev, cur = cur, nb[0]
    return None


def _subtree(mol, start, block):
    """剪断 start–block 后 start 所在的那一侧（含氢）。"""
    out, seen, stack = [], {block}, [start]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() not in seen:
                stack.append(nb.GetIdx())
    return out


def _arm_spread(pos, conn):
    """各条臂末端连接原子的"共圆程度"：(半径极差, 方位角间隔最大偏差)。"""
    q = np.asarray(pos)[conn]
    c = q.mean(axis=0)
    v = q - c
    nz = unit(np.linalg.svd(v)[2][2])
    v = v - np.outer(v @ nz, nz)
    r = np.linalg.norm(v, axis=1)
    e1 = unit(v[0])
    e2 = np.cross(nz, e1)
    az = np.sort(np.degrees(np.arctan2(v @ e2, v @ e1)) % 360.0)
    gaps = np.diff(np.append(az, az[0] + 360.0))
    return float(r.max() - r.min()), float(np.abs(gaps - 360.0 / len(conn)).max())


def _arm_sym_residual(pos, sym, conn):
    """各条臂绕自身 C_n 轴转到同一方位后，重原子的最大不重合量（Å）。

    为什么光看"连接原子共圆"不够
    ----------------------------
    `_arm_spread` 只比较连接原子本身。可臂上往往挂着取向可翻的侧基
    （羧基的 =O、醛基的 =O、-OH）。侧基翻到哪一侧不影响连接原子的位置，
    却让整个单体失去 C_n。这一步量的就是那个差异：把每条臂转回第一条臂
    的方位，看同种原子最远差了多远。

    只统计距轴 > 0.8 Å 的原子（轴上的原子转多少度都落在原地）。
    """
    p = np.asarray(pos, dtype=float)
    n = len(conn)
    q = p[list(conn)]
    cc = q.mean(axis=0)
    v = q - cc
    nz = unit(np.linalg.svd(v)[2][2])
    v = v - np.outer(v @ nz, nz)
    e1 = unit(v[0])
    e2 = np.cross(nz, e1)
    az_arm = np.degrees(np.arctan2(v @ e2, v @ e1)) % 360.0

    heavy = [i for i, s in enumerate(sym) if s != "H"]
    w = p[heavy] - cc
    flat = w - np.outer(w @ nz, nz)          # 只在平面内的分量，用来定方位
    r = np.linalg.norm(flat, axis=1)
    az = np.degrees(np.arctan2(flat @ e2, flat @ e1)) % 360.0
    diff = (az[:, None] - az_arm[None, :] + 180.0) % 360.0 - 180.0
    arm_of = np.argmin(np.abs(diff), axis=1)
    elems = [sym[i] for i in heavy]
    far = r > 0.8
    # 比较用完整三维矢量：侧基"朝上还是朝下"（轴向分量）同样是 C_n 的一部分，
    # 只比平面内分量会把上下翻转看成一致。
    base_sel = far & (arm_of == 0)
    if not base_sel.any():
        return 0.0
    base = w[base_sel]
    base_el = [e for e, m in zip(elems, base_sel) if m]

    worst = 0.0
    for k in range(1, n):
        sel = far & (arm_of == k)
        if not sel.any():
            continue
        R = rot_axis(nz, -(az_arm[k] - az_arm[0]))
        moved = w[sel] @ R.T
        for row, e in zip(moved, [e for e, m in zip(elems, sel) if m]):
            cand = [b for b, be in zip(base, base_el) if be == e]
            if not cand:
                continue
            worst = max(worst, float(np.min(np.linalg.norm(np.array(cand) - row, axis=1))))
    return worst


def unify_arms(mol, pos, conn, verbose=True):
    """把星形单体各条臂末端的可旋转基团统一到同一手性。

    为什么必须做
    ------------
    `flatten_torsions` 对每根可旋转单键独立地取"最近的 0°/180°"。
    对三连接节点，末端乙烯基绕 Ar–C 单键转 0° 还是 180° 是等价的两种
    构象，但两种朝向让连接原子落在臂轴的外侧或内侧：实测同一个单体里
    两条臂转了 0°、一条转了 180°，三条臂的连接原子半径变成
    13.31 / 12.93 / 13.20 Å（差 0.38 Å），方位角间隔 121.6/119.6/118.8°。

    臂长 13 Å 时 2.8° 就是 0.6 Å 的切向错位。装配阶段要求"三条臂各自
    的平移量一致"，这个单体给出的平移量相差 1.05 Å，晶胞根本拼不起来
    （报错："两个节点的臂接不上"）。而 monomer 本身没有任何化学问题，
    只是构象手性没统一。

    做法：枚举每条臂翻/不翻（2^n，n≤4），取共圆程度最好的组合。
    只在确实能改善时才写回，避免动到本来就对称的单体。
    """
    rotors = [_rotor_bond(mol, c) for c in conn]
    if not any(rotors):
        return 0

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    r0, g0 = _arm_spread(pos, conn)
    s0 = _arm_sym_residual(pos, syms, conn)
    best = None
    for bits in product((0, 1), repeat=len(conn)):
        p = np.array(pos, dtype=float)
        for bit, rb in zip(bits, rotors):
            if not bit or rb is None:
                continue
            sub, anchor = rb
            ax = unit(p[sub] - p[anchor])
            R = rot_axis(ax, 180.0)
            for i in _subtree(mol, sub, anchor):
                p[i] = p[anchor] + R @ (p[i] - p[anchor])
        dr, dg = _arm_spread(p, conn)
        ds = _arm_sym_residual(p, syms, conn)
        cost = (dr + ds + np.radians(dg) * float(np.mean(
            np.linalg.norm(np.asarray(p)[conn] - np.asarray(p)[conn].mean(0), axis=1))))
        if best is None or cost < best[0]:
            best = (cost, p, dr, dg, ds, sum(bits))
    _, p, r1, g1, s1, nflip = best

    gain = (r0 - r1) + (s0 - s1) + np.radians(g0 - g1) * 13.0
    if gain > 0.02:                      # 至少改善 0.02 Å 才写回
        pos[:] = p
        if verbose:
            print(f"[提示] 末端转子未统一：三条臂半径差 {r0:.2f} → {r1:.2f} Å，"
                  f"方位角偏差 {g0:.1f}° → {g1:.1f}°"
                  f"，侧基不重合 {s0:.2f} → {s1:.2f} Å（翻了 {nflip} 条臂）")
        return nflip
    return 0


def idealize_sp_sp(pos, mol, target=1.375, tol=0.02):
    """修正共轭二炔中间那根 C(sp)–C(sp) 单键的键长。

    RDKit 的 MMFF94 对这类键没有合适的参数，优化完会停在 ~1.54 Å
    （参见 `CC#CC#CC`：C–C(sp) 1.46、C(sp)–C(sp) 1.54），比晶体学值
    1.37 Å 长 0.17 Å。

    实现要点：**必须把"一侧整个片段"沿键轴刚性平移**，不能只挪两个端点。
    -C#C-C#C- 是一条直线，若只把两端各挪 0.08 Å，缩掉的量会等量灌进相邻
    的两根三键（1.20 → 1.28 Å）：臂的总长一点没变，三键全废了。
    平移整侧既不碰三键，也真正缩短了这条臂。
    """
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    def far_side(i, j):
        """剪断 i–j 后 j 所在的那一整侧（含氢）。"""
        seen = {j}
        stack = [j]
        while stack:
            k = stack.pop()
            for m in adj[k]:
                if m == i or m in seen:
                    continue
                seen.add(m)
                stack.append(m)
        return seen

    n_fix = 0
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (mol.GetAtomWithIdx(i).GetHybridization() != Chem.HybridizationType.SP or
                mol.GetAtomWithIdx(j).GetHybridization() != Chem.HybridizationType.SP):
            continue
        v = pos[j] - pos[i]
        d = float(np.linalg.norm(v))
        if d < 1e-6 or abs(d - target) < tol:
            continue
        side = np.fromiter(far_side(i, j), dtype=int, count=-1)
        pos[side] -= v / d * (d - target)     # j 侧整体朝 i 平移
        n_fix += 1
    return n_fix


# ---------------------------------------------------------------- 单体准备
def prepare(smiles, n_expected, flatten=False, seed=20261018, n_trials=4):
    """SMILES → 封端补氢 → 3D 嵌入 → 力场优化 → 摘掉封端原子。

    关键：连接原子在真实框架里要成键，直接删掉 * 会留下自由基，导致
    ETKDG/MMFF 严重失真（键长被压到 0.8 Å）。所以先把 * 换成 H 生成一个
    完整的闭壳层分子做优化，最后再把这些封端 H 去掉，只保留坐标。

    返回 (符号, 坐标, 连接原子索引, 连接原子原有氢数, 连接键方向)。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")

    mol = prune_star_chains(mol)          # 规整 *[*] 这类"星号连星号"的写法
    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != n_expected:
        raise ValueError(f"SMILES 中 * 的个数应为 {n_expected}，实际为 {len(dummies)}: {smiles}")

    # 记录连接原子（* 的唯一邻居）
    conn_old = [mol.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx() for d in dummies]

    # * → H 封端，得到完整分子
    rw = Chem.RWMol(mol)
    for d in dummies:
        rw.GetAtomWithIdx(d).SetAtomicNum(1)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m)
    syms_all = [a.GetSymbol() for a in m.GetAtoms()]
    heavy_all = [i for i, s in enumerate(syms_all) if s != "H"]

    # 多构象择优：二维 COF 的单体应该尽量平，翘曲的构象会让层厚失控
    #
    # 嵌入分两轮：先用 ETKDG 默认初始化，全失败再换 useRandomCoords=True。
    # 默认初始化对"又长又直"的单体（几条 -C#C-C#C- 刚性棒）经常直接失败：
    # 共线的 sp 片段越多，距离几何的边界平滑越不容易收敛，90 个原子以上
    # 基本必败；而随机初始坐标几乎都能出来，只是构象偏"卷"，所以只在
    # 默认策略全军覆没时才退到它。
    best = None
    for strat in ({"useRandomCoords": False}, {"useRandomCoords": True}):
        for k in range(max(1, n_trials)):
            trial = Chem.Mol(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed + 1013 * k
            for key, val in strat.items():
                setattr(params, key, val)
            if AllChem.EmbedMolecule(trial, params) != 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(trial, maxIters=5000)
            except Exception:
                AllChem.UFFOptimizeMolecule(trial, maxIters=5000)
            p = np.array(trial.GetConformer().GetPositions())
            idealize_sp_sp(p, trial)          # 修 MMFF94 对二炔中段单键的偏差
            q = p[heavy_all] - p[heavy_all].mean(axis=0)
            flat = float(np.linalg.svd(q, compute_uv=False)[-1])   # 出平面量级
            dd = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(dd, 99.0)
            clash = max(0.0, 0.95 - float(dd.min()))               # 原子重叠罚分
            score = flat + 30.0 * clash
            if best is None or score < best[0]:
                best = (score, trial, p)
        if best is not None:            # 默认初始化解出来了，就不用退到随机坐标
            break
    if best is None:
        raise RuntimeError(
            f"3D 嵌入失败（{len(heavy_all)} 个重原子）: {smiles}\n"
            f"  常见原因：单体太长/太刚（多条 -C#C-C#C- 刚性棒），"
            f"或环系本身不是平面芳香体系。\n"
            f"  可先把单体拆短试一下；若确认结构没问题，就是构象搜索的极限，"
            f"换 RDKit 版本或手搭初始坐标。")
    _, m, pos = best

    # 把芳环转平（二维 COF 必须是平面层）。
    #  · 线形连接子（2 个连接位）用"分子长轴"当基准：环平面包含该轴；
    #  · 星形节点（3/4 个连接位）没有共同长轴，改用节点自身平面的法向
    #    （由任意两条臂的方向叉乘得到）——否则 ETKDG 会把每条臂末端的苯环
    #    扭成螺旋桨，整层厚度窜到 4 Å 以上，相邻单体直接撞在一起。
    def _offplane():
        q = pos[[i for i, s in enumerate(syms_all) if s != "H"]]
        q = q - q.mean(axis=0)
        return float(np.abs(q @ np.linalg.svd(q)[2][2]).max())

    def _rings():
        # 绕"两个取代基的连线"把每个芳环转平。对位二取代的六元环恰好成立：
        # 两根外伸键与转轴共线，转环不会动到外伸键。
        if len(conn_old) == 2:
            flatten_rings(m, pos, pos[conn_old[1]] - pos[conn_old[0]], cap=dummies)
        elif len(conn_old) >= 3:
            cc = pos[conn_old].mean(axis=0)
            d0 = unit(pos[conn_old[0]] - cc)
            d1 = unit(pos[conn_old[1]] - cc)
            flatten_rings(m, pos, cap=dummies, normal=unit(np.cross(d0, d1)))

    if flatten:
        # 两种顺序各有擅长，而且**任何一步都可能把已经压平的结构弄歪**
        # （含噻吩的连接子：拧完二面角已经 0.013 Å，再转一次环就窜到 0.86 Å）。
        # 所以每走一步都记一次离面量，最后取历史上最平的那个状态，
        # 而不是盲信"最后一步"。
        def _attempt(seq):
            keep = pos.copy()

            def _score():
                # 只看离面量不够：四条臂的星形节点里，末端苯环翻转 180° 虽然
                # 一样平，却会把两个氢怼到 0.8 Å（芘-苯并双噻唑体系实测）。
                # 原子重叠必须一起算进代价。
                dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                np.fill_diagonal(dd, 99.0)
                return _offplane() + 30.0 * max(0.0, 0.95 - float(dd.min()))

            best_sc, best_pos = _score(), keep.copy()
            for step in seq:
                if step == "rings":
                    _rings()
                else:
                    flatten_torsions(m, pos)
                sc = _score()
                if sc < best_sc:
                    best_sc, best_pos = sc, pos.copy()
            pos[:] = keep
            return best_sc, best_pos

        _sc, _best = min((_attempt(o) for o in
                          (("rings", "torsions", "rings"),
                           ("torsions", "rings", "torsions"))),
                         key=lambda t: t[0])
        pos[:] = _best

    # 压平之后各条臂末端基团可能被翻到不同的手性上（flatten_torsions 逐根
    # 单键取"最近的 0°/180°"，三条臂各翻各的）。臂一长，0.38 Å 的半径差
    # 就让装配时的三个平移量对不上，晶胞拼不起来。这里统一回 C3。
    if flatten and len(conn_old) >= 3:
        unify_arms(m, pos, conn_old)

    # 单体自检：装配之前先量这个单体自己带不带 C_n。
    # 破缺会原样传进整层——等到 spglib 判 P1 才发现就晚了（要动 1 Å 以上才能
    # 拉回来，键长会烂）。所以每次建模都打一行，超标就明说。
    _chk_before = None
    if CHECK_MONOMER and len(conn_old) >= 2:
        try:
            import monomer_sym as _ms
            _chk_before = _ms.check_monomer(syms_all, pos, len(conn_old),
                                            conn=conn_old)
            MONOMER_CHECKS.append(_chk_before)
            print("[单体] " + _ms.format_check(_chk_before))
        except Exception as _e:
            print(f"[提示] 单体自检跳过（{type(_e).__name__}: {_e}）")

    # 单体点群对称化：把构象投影回它自己的 C_n。
    # ETKDG + MMFF 给的是"某一个构象"，不是对称构象：卟啉四臂节点绕自身轴
    # 转 90°，重原子平均要挪 0.45 Å、最大 0.97 Å。装配会把这个破缺原样传下去。
    if SYMMETRIZE_MONOMER and len(conn_old) >= 2:
        try:
            import monomer_sym as _ms
            _p, _shift, _nops = _ms.symmetrize_monomer(
                syms_all, pos, len(conn_old), verbose=False, conn=conn_old)
            if _nops and _shift > 0.01:
                # 只投影、不再跑力场。MMFF 会把卟啉核心压成鞍形（离面 1.25 Å），
                # 反而毁掉前面辛苦压平的一层。投影本身对键长的影响很小
                # （C–C 1.372–1.499 Å），剩下的交给 MACE 弛豫。
                pos[:] = _p
                _after = _ms.check_monomer(syms_all, pos, len(conn_old),
                                           conn=conn_old)
                MONOMER_CHECKS.append(_after)
                print(f"[单体] {_ms.format_check(_after, prefix='投影后')}"
                      f"（原残差 {_shift*1000:.0f} mÅ，{_nops} 个操作）")
        except Exception as _e:
            print(f"[提示] 单体对称化跳过（{type(_e).__name__}: {_e}）")

    # 封端 H 所在方向 = 该连接原子真正的外伸键方向（装配时要用它定倾角）
    cap_dir = np.zeros_like(pos)
    for d, c in zip(dummies, conn_old):
        cap_dir[c] = unit(pos[d] - pos[c])

    # 摘掉封端 H，重排索引
    cap = set(dummies)
    # 连接原子上原有的 H 也一并摘掉：封端让它们暂时按 sp3 摆放，
    # 装配后会用两个重原子邻居重新生成（理想 sp2 位置，与层共面）
    drop = set(cap)
    conn_h_count = []
    for c in conn_old:
        hs = [nb.GetIdx() for nb in m.GetAtomWithIdx(c).GetNeighbors()
              if nb.GetAtomicNum() == 1 and nb.GetIdx() not in cap]
        conn_h_count.append(len(hs))
        drop.update(hs)
    keep = [i for i in range(m.GetNumAtoms()) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    sym = [m.GetAtomWithIdx(i).GetSymbol() for i in keep]
    conn = [remap[c] for c in conn_old]
    return sym, pos[keep], conn, conn_h_count, cap_dir[keep]


def add_conn_hydrogens(atoms, conn_spec, blen=1.09, cutoff=1.95, verbose=False):
    """按连接原子原有的氢数补 H（理想 sp2：与两个重原子键共面、120°）。

    这里有两个坑，都是实测踩出来的：

    ① ``neighbor_list`` 每对原子只回一次（i<j）。早先只收 ``i == c`` 那一侧，
       索引比连接原子小的邻居**全被漏掉**，于是 =CH– 只剩一个邻居，按"单
       邻居反式"的规则把 H 摆到 C=N 键上：实测 ∠H–C–N = 13~28°、H···N 1.24 Å，
       亚胺那个"氢贴到氮上"的老毛病就是它。必须两侧都登记。

    ② 装配是刚体拼的，连接原子的两根重键可能被压成 150° 甚至反式（腙、
       酰胺尤其明显）。这时"反平分"正好把 H 推进旁边那根键里。所以方向
       不再按公式硬算，而是在层内扫一圈 (±z 也带上)，按"离其它原子最近
       距离"择优，角度只用来打破平局——不打架优先于角度好看。
    """
    from ase.neighborlist import neighbor_list

    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(atoms)
    i_idx, j_idx, D = neighbor_list("ijD", atoms, cutoff)

    # 双向登记（见 ① ）
    nb = {}
    for k, i in enumerate(i_idx):
        j = int(j_idx[k])
        if symbols[j] == "H":
            continue
        v = np.asarray(D[k], dtype=float)
        dv = float(np.linalg.norm(v))
        nb.setdefault(int(i), []).append((dv, v))
        nb.setdefault(j, []).append((dv, -v))

    # 最小镜像用：分数坐标换算矩阵（非周期结构就不折）
    try:
        cell_arr = np.asarray(atoms.cell, dtype=float)
        inv_cell = np.linalg.inv(cell_arr) if atoms.pbc.any() and cell_arr[0][0] > 0.1 else None
    except Exception:
        inv_cell = None

    z = np.array([0.0, 0.0, 1.0])
    extra = []
    for c, n_h in conn_spec:
        c = int(c)
        if n_h <= 0:
            continue
        # 只认真正的成键邻居：晶胞偏紧时 1.95 Å 内会混进隔壁单体的原子，
        # 把它们的矢量也算进"合方向"，H 会被甩到 0.3 Å 处（实测过）。
        #
        # 还要**按方向去重**：neighbor_list 会把同一根键的几个周期镜像都列出来
        # （亚胺的 C–N 在 1.28 Å 处出现两遍），直接取前两个就得到两个同向矢量，
        # 夹角算成 0°、局部法向退化，=CH– 的氢被摆到垂直层面（实测 1.09 Å）。
        near = sorted(nb.get(c, []), key=lambda t: t[0])
        vecs, seen_dirs = [], []
        for dv, v in near:
            if dv > 1.62 and vecs:
                break
            u = unit(v)
            if any(abs(float(np.dot(u, s))) > 0.985 for s in seen_dirs):
                continue
            seen_dirs.append(u)
            vecs.append(v)
            if len(vecs) == 2:
                break
        if not vecs:
            continue
        units = [unit(v) for v in vecs]
        bonded = [j for j, _ in near[:len(vecs)]]

        # 候选方向：层内每 15° 一圈 + 上下法向（层法向 = 全局 z，2D 层就是按 z 叠的）。
        # 曾在"局部平面（u1×u2）"和"层平面"之间纠结过，实测结论是**用层法向**：
        # 局部平面会被邻位取代基（烯醇式的 OH）带歪，氢跟着歪出层面；层法向
        # 则保证 12 个体系的重原子离面都是 0.00 Å。
        nrm = z

        e1 = units[0]
        e2 = np.cross(nrm, e1)
        if np.linalg.norm(e2) < 1e-6:
            e2 = np.cross(np.array([1.0, 0.0, 0.0]), e1)
        e2 = unit(e2)
        cands = [np.cos(np.radians(t)) * e1 + np.sin(np.radians(t)) * e2
                 for t in range(0, 360, 15)]
        cands += [nrm, -nrm]

        # 只在 H 可能落到的范围内找干扰原子。**成键邻居必须算进来** ——
        # 把它们排除掉的话，"顺着键往外伸"这个方向对其它原子的距离是无穷大，
        # 评分反而最高，H 就被硬塞到键里（实测 ∠H–C–N 28°、H···N 1.2 Å）。
        # 距离一律按最小镜像算：薄层胞在 xy 方向只有几个 Å，不折回来会把
        # 隔壁胞的原子当成"很远"，H 又被推到周期边界外撞上它们。
        keep = np.array([i for i in range(n) if i != c], dtype=int)

        def plain(d):
            """不看出平面的原始评分：最近距离 + 一点角度偏好。"""
            q = pos[c] + blen * d
            if inv_cell is None:
                dmin = float(np.linalg.norm(pos[keep] - q, axis=1).min()) if len(keep) else 9.0
            else:
                _f = (pos[keep] - q) @ inv_cell
                _f -= np.round(_f)
                dmin = float(np.linalg.norm(_f @ cell_arr, axis=1).min()) if len(keep) else 9.0
            # 与各重键的最小夹角（理想 sp2 = 120°，sp3 由调用方按 2 个 H 处理）
            ang = min(float(np.degrees(np.arccos(np.clip(np.dot(d, u), -1.0, 1.0))))
                      for u in units)
            return dmin, dmin + 0.0020 * min(ang, 120.0)

        # 层内到底有没有"能站住"的位置？最近距离 ≥1.25 Å 才算站得住。
        # 站得住就把 H 压回分子平面；站不住（氰基烯烃旁边全是氰基/邻位氢，
        # 实测层内最好只有 1.1 Å 出头）才允许翻到法向去，否则压平反而撞得更狠
        # ——实测 cyanovinylene 从 1.087 Å 掉到 0.575 Å。
        flat_ok = max(plain(d)[0] for d in cands[:24]) >= 1.25

        def score(d):
            dfree, base = plain(d)
            if not flat_ok:
                return base
            # 出平面惩罚：score 是**取最大**，这里必须减不能加 ——
            # 早先写成 "+ 0.55*oop"，等于奖励出平面，=CH– 每次都被翻上去。
            # ±nrm 的 dmin 天然偏大（法向没有邻居，实测能高 0.9 Å），不扣
            # 分就永远赢。
            return base - 1.20 * abs(float(np.dot(d, nrm)))

        picked = []
        pool = list(cands)
        for _ in range(int(n_h)):
            best = max(pool, key=score)
            picked.append(best)
            # 还要放第二个 H 时，避开已选方向（CH2，罕见）
            pool = [d for d in pool
                    if float(np.dot(unit(d), unit(best))) < 0.5] or pool
        for d in picked:
            extra.append(pos[c] + blen * unit(d))

    return extra


def repair_conn_h(atoms, conn_spec, blen=1.09):
    """把连接原子上的氢摘下来，按最终的成键几何重摆（骨架不动）。

    为什么要单独一遍：单体里的连接位氢是 RDKit 按"星号方向"嵌的，装配时
    连接键不一定落回那个方向（亚胺差 60°、腙差 30°+），于是 H 被顶到
    邻键上，实测 H···N 1.20 Å、∠H–C–N 60°。这里只动氢的方位，
    重原子坐标一个都不碰，所以不会破坏已经调好的装配。
    """
    sym = list(atoms.get_chemical_symbols())
    pos = np.array(atoms.get_positions())
    n = len(atoms)
    if n == 0:
        return atoms
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)

    drop, respec = set(), []
    for c, n_h in conn_spec:
        c = int(c)
        hs = [j for j in range(n) if sym[j] == "H" and d[c, j] < 1.25]
        if not hs:
            continue
        drop.update(hs)
        respec.append((c, len(hs)))
    if not respec:
        return atoms

    idx = [i for i in range(n) if i not in drop]
    remap = {o: k for k, o in enumerate(idx)}
    sub = Atoms(symbols=[sym[i] for i in idx], positions=pos[idx],
                cell=atoms.cell, pbc=True)
    extra = add_conn_hydrogens(sub, [(remap[c], k) for c, k in respec])
    if not extra:
        return atoms
    return Atoms(symbols=[sym[i] for i in idx] + ["H"] * len(extra),
                 positions=np.vstack([pos[idx], np.array(extra)]),
                 cell=atoms.cell, pbc=True)


def _bond_pairs(pos, sym, cutoff=1.95):
    """按距离阈值找出所有成键的"重原子-重原子"对（用来检查压平前后的键长）。"""
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    pairs = [(i, j) for a, i in enumerate(heavy) for j in heavy[a + 1:]
             if d[i, j] < cutoff]
    return pairs


def project_planar(pos, sym, max_shift=0.60, verbose=True, max_strain=0.06):
    """把整个片段投影到其重原子的最佳拟合平面（消除构象褶皱）。

    只改垂直于平面的起伏，平面内坐标不变。

    起伏超过 max_shift 时不再一律放弃，而是先检查"压平后键长会变差多少"：
    像三(4-氨基苯基丁二炔基)苯这种又长又刚的平面单体，ETKDG 很容易把它嵌成
    轻微螺旋桨状（出平面 1~2 Å），此时压平只带来百分之几的键长误差，远好过
    留着一层 4 Å 厚的褶皱（那会让相邻单体直接撞在一起）。只有当键长真的被压
    坏（例如卟啉氢数/芳香性写错、翘曲 4.9 Å）时才放弃。
    """
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    c = pos[heavy].mean(axis=0)
    n = unit(np.linalg.svd(pos[heavy] - c)[2][2])
    shift = (pos - c) @ n
    dev = float(np.abs(shift[heavy]).max())
    if dev > max_shift:
        flat = pos - np.outer(shift, n)
        pairs = _bond_pairs(pos, sym)
        if pairs:
            ii = np.array([p[0] for p in pairs])
            jj = np.array([p[1] for p in pairs])
            L0 = np.linalg.norm(pos[ii] - pos[jj], axis=1)
            L1 = np.linalg.norm(flat[ii] - flat[jj], axis=1)
            strain = float((np.abs(L1 - L0) / L0).max())
        else:
            strain = 0.0
        if strain > max_strain:
            if verbose:
                print(f"[警告] 单体出平面起伏达 {dev:.2f} Å，压平会使键长变化 "
                      f"{100 * strain:.1f}%，已跳过；请检查单体 SMILES 是否正确"
                      f"（氢数/芳香性写错会让环严重翘曲）")
            return False
        if verbose:
            print(f"[提示] 单体出平面起伏 {dev:.2f} Å（刚性平面单体被嵌歪），"
                  f"已压平（键长最大变化 {100 * strain:.1f}%，可忽略）")
    pos -= np.outer(shift, n)
    return True


# ---------------------------------------------------------------- 连接位规整
def prune_star_chains(mol):
    """把 ``*[*]N`` 这种"星号连星号"的写法规整成单个星号。

    从文献 / 建模软件里拷出来的片段常把连接位写成 ``*[*]``，RDKit 会解析成
    "一个末端星号挂在一个连着真实原子的星号上"。这里删掉多余的末端星号，
    只保留直接连在真实原子上的那个。
    """
    while True:
        lone = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetAtomicNum() == 0
                and all(nb.GetAtomicNum() == 0 for nb in a.GetNeighbors())]
        if not lone:
            return mol
        rw = Chem.RWMol(mol)
        for i in sorted(lone, reverse=True):
            rw.RemoveAtom(i)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)


def count_stars(smiles):
    """数连接位（``*``）的个数。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")
    mol = prune_star_chains(mol)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


# ---------------------------------------------------------------- 自动选连接位
# 目的：让用户只写"分子本身"的 SMILES（完全不带 *），脚本自己判断
#   ① 哪些原子是 sp2、可以当连接位；
#   ② 挑哪几个组合最对称（3 臂 120°、4 臂 90°、半径一致）；
#   ③ 生成等价的带 * SMILES 交给后面的流程。
# 这样就不必背 [*][CH]c1nc([*])nc([*])n1 这种写法，也不会误把 CH2 当连接位。

_TIER_TEXT = {0: "sp2 碳", 1: "氨基 N", 2: "芳环 N–H"}


def _hyb_text(a):
    tbl = {
        Chem.HybridizationType.SP: "sp",
        Chem.HybridizationType.SP2: "sp2",
        Chem.HybridizationType.SP3: "sp3",
    }
    return tbl.get(a.GetHybridization(), "其它")


def strip_stars(smiles):
    """去掉所有 ``*``，还原成"纯分子"SMILES（被删掉的位置按价态补回 H）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")
    mol = prune_star_chains(mol)
    ds = sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0],
                reverse=True)
    if not ds:
        return Chem.MolToSmiles(mol)
    rw = Chem.RWMol(mol)
    for i in ds:
        rw.RemoveAtom(i)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


def scan_sites(smiles):
    """列出分子里所有**化学上**能当连接位的原子。

    只认三类（其余一律不算，尤其是 CH3 / CH2 / sp3 CH）：

      0  sp2 碳   芳香碳或双键碳，带 1 个 H、重原子配位 2
                  → 接上去正好 3 配位，保持平面三角
      1  氨基 N   带 H、重原子配位 1（-NH2 / -NH-）
                  → 亚胺缩合会去掉 1 个 H，缩合后就是 sp2 N
      2  芳环 N–H 吡咯型，也能接，但少见，排在最后

    甲基 CH3 重原子配位是 1 → 天然排除；亚甲基 CH2 带 2 个 H → 排除。
    这就是"为什么 *CC1=NC(...) 接不上"的根源：那个 C 真的是个 CH2。

    返回 [dict(idx, sym, hyb, nh, tier)]，按原子序号升序。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")
    mol = prune_star_chains(mol)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("这个 SMILES 里已经带了 *；自动选位只能对'纯分子'使用")

    sp2ish = (Chem.HybridizationType.SP2, Chem.HybridizationType.SP)
    out = []
    for a in mol.GetAtoms():
        z, nh, deg = a.GetAtomicNum(), a.GetTotalNumHs(), a.GetDegree()
        hyb, ring = a.GetHybridization(), a.IsInRing()
        tier = None
        if z == 6:
            if nh == 1 and deg == 2 and hyb in sp2ish:
                tier = 0
        elif z == 7:
            if nh >= 1 and deg == 1 and hyb != Chem.HybridizationType.SP3:
                tier = 1
            elif nh == 1 and deg == 2 and ring:
                tier = 2
        if tier is not None:
            out.append(dict(idx=a.GetIdx(), sym="C" if z == 6 else "N",
                            hyb=_hyb_text(a), nh=nh, tier=tier))
    return out


def _site_geometry(smiles, idxs):
    """铺 2D 坐标，返回 (全部坐标, {连接位: 外伸单位矢量})。

    用 2D 而不是 ETKDG 3D：目标分子本身就是平面芳香体系，2D 布局给出的
    邻/间/对关系比任意一个 3D 构象更可靠，而且快、可复现。
    外伸方向取"重原子邻居的负矢量和"，等价于被取代掉的那个 C–H 的朝向。
    """
    mol = prune_star_chains(Chem.MolFromSmiles(smiles))
    mh = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mh)
    pos = np.array(mh.GetConformer().GetPositions(), dtype=float)
    dirs = {}
    for i in idxs:
        a = mh.GetAtomWithIdx(i)
        nb = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() != 1]
        v = -sum(unit(pos[j] - pos[i]) for j in nb)
        if np.linalg.norm(v) < 0.15:
            # 醛基碳这种"两个重原子邻居几乎共线"的情形，负矢量和会退化，
            # 改用原有的 C–H 方向（也是取代后新键的大致朝向）
            hs = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1]
            v = (pos[hs[0]] - pos[i]) if hs else np.array([1.0, 0.0, 0.0])
        dirs[i] = unit(v)
    return pos, dirs


_IDEAL_GAPS = {
    2: [[180.0]],
    3: [[120.0, 120.0, 120.0]],
    4: [[90.0, 90.0, 90.0, 90.0], [60.0, 60.0, 120.0, 120.0]],
    6: [[60.0] * 6],
}


def _site_score(pos, dirs, combo):
    """给一组候选位点打分（越小越好）。

    除了"臂角 / 半径 / 朝外"三项，2 连接还要额外偏好**两端最远**的一对：
    线形连接子必须横跨整个分子的长轴。否则一条 Ph-C≡C-C6H4-C≡C-Ph 的棒，
    会挑到中间那个环上的一对对位碳（角度 180°、半径也相等，几何上完美），
    接出来的层只有 2.8 Å 宽，直接塌成一团。
    """
    n = len(combo)
    P = pos[list(combo)]
    c = P.mean(axis=0)
    R = P - c
    radii = np.linalg.norm(R, axis=1)
    spread = float(radii.max() - radii.min())

    ang = np.sort(np.degrees(np.arctan2(R[:, 1], R[:, 0])))
    gaps = np.diff(np.append(ang, ang[0] + 360.0))
    ideals = _IDEAL_GAPS.get(n)
    if ideals:
        gap_dev = min(float(np.abs(np.sort(gaps) - np.sort(ideal)).max())
                      for ideal in ideals)
    else:
        gap_dev = 0.0

    outward = 0.0
    for k, i in enumerate(combo):
        r = float(np.linalg.norm(R[k]))
        if r < 1e-6:            # 质心正好落在某个位点上（退化组合），直接重罚
            outward += 9.0
            continue
        dot = float(np.dot(dirs[i], R[k] / r))
        outward += max(0.0, 0.35 - dot)      # 朝里指的臂要罚

    span = 0.0
    if n == 2:
        span = -float(np.linalg.norm(P[1] - P[0]))     # 越远越好（单位 Å）

    return (gap_dev + 25.0 * spread + 40.0 * outward + span), gap_dev, spread


def _site_priority(mol, idx):
    """化学偏好：数字越小越优先当连接位。

    0  羰基/醛基碳、腈碳、氨基 N —— 缩合化学里天然就是成键的地方
    1  链上 sp2 碳（乙烯基、亚胺碳）
    2  芳环 C–H
    3  芳环 N–H
    没有这一项的话，"三醛基苯"会挑到苯环上的 C–H，而不是三个醛基碳。
    """
    a = mol.GetAtomWithIdx(idx)
    if a.GetAtomicNum() == 7:
        return 3 if a.IsInRing() else 0
    for b in a.GetBonds():
        if b.GetBondTypeAsDouble() == 2.0 and \
                b.GetOtherAtom(a).GetAtomicNum() in (7, 8):
            return 0
        if b.GetBondType() == Chem.BondType.TRIPLE:
            return 0
    return 2 if a.IsInRing() else 1


def pick_sites(smiles, n, verbose=False, max_combos=200000):
    """从纯分子 SMILES 里自动挑 n 个最对称的 sp2 连接位。

    两道筛选：
      ① **对称性**：先用 RDKit 的原子对称等价类（CanonicalRankAtoms）把位点
         分组，只在同一个等价类里挑 — 这样 4 个位点必然取自芘的 1,3,6,8
         这种同源位置，不会出现"一边两个、一边一个"的歪搭配；
      ② **几何**：臂角偏离理想值(°) + 25×半径差(Å) + 40×朝内惩罚 + 8×化学偏好。

    返回 (位点序号列表, 诊断字符串)。
    """
    mol = prune_star_chains(Chem.MolFromSmiles(smiles))
    sites = scan_sites(smiles)
    if len(sites) < n:
        cnt = {}
        for s in sites:
            cnt[_TIER_TEXT[s["tier"]]] = cnt.get(_TIER_TEXT[s["tier"]], 0) + 1
        detail = "、".join(f"{k} {v} 个" for k, v in cnt.items()) or "一个都没有"
        raise ValueError(
            f"这个分子只有 {len(sites)} 个可用的 sp2 连接位（{detail}），"
            f"做不了 {n} 连接。\n"
            f"  可用位点：{[(s['idx'], s['sym'], _TIER_TEXT[s['tier']]) for s in sites]}\n"
            f"  需要 {n} 个：换一个对称性够的单体，或减少连接数。")

    all_idx = [s["idx"] for s in sites]
    pos, dirs = _site_geometry(smiles, all_idx)
    prio = {i: _site_priority(mol, i) for i in all_idx}

    # ① 按对称等价类分组；某个类够 n 个就只在类内枚举
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    groups = {}
    for i in all_idx:
        groups.setdefault(ranks[i], []).append(i)
    pools = [g for g in groups.values() if len(g) >= n] or [all_idx]

    best = None
    cnt = 0
    for pool in pools:
        group_prio = min(prio[i] for i in pool)
        for combo in combinations(pool, n):
            cnt += 1
            if cnt > max_combos:
                break
            sc, gap_dev, spread = _site_score(pos, dirs, combo)
            sc += 8.0 * max(prio[i] for i in combo)
            if best is None or sc < best[0]:
                best = (sc, combo, gap_dev, spread, group_prio)
        if cnt > max_combos:
            break
    if best is None:
        raise ValueError(f"无法从 {len(sites)} 个位点里凑出 {n} 个组合")

    sc, combo, gap_dev, spread, _ = best
    combo = list(combo)
    if verbose:
        print(f"[自动选位] {n} 连接：取原子 {combo}，"
              f"臂角偏差 {gap_dev:.1f}°，半径差 {spread:.2f} Å")
    return combo, f"臂角偏差 {gap_dev:.1f}°，半径差 {spread:.2f} Å"


def star_smiles(smiles, sites):
    """在指定原子上挂 ``*``，返回带星号的 SMILES。"""
    mol = prune_star_chains(Chem.MolFromSmiles(smiles))
    rw = Chem.RWMol(mol)
    for c in sites:
        d = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(int(c), int(d), Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.MolToSmiles(m)


def diagnose_stars(smiles):
    """指出星号挂错在哪：返回 [(星号原子, 连接原子, 元素, 杂化, 氢数)]。"""
    mol = prune_star_chains(Chem.MolFromSmiles(smiles))
    rows = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        c = a.GetNeighbors()[0]
        rows.append(dict(star=a.GetIdx(), idx=c.GetIdx(), sym=c.GetSymbol(),
                         hyb=_hyb_text(c), nh=c.GetTotalNumHs(),
                         deg=c.GetDegree()))
    return rows


def auto_star(smiles, n, role="单体", verbose=True, force=False, allow_sp3=False):
    """把用户输入统一成"正好 n 个 ``*``"的 SMILES。

    · 已有 n 个 * 且没要求重选 → 原样返回（尊重显式指定）
    · 一个 * 都没有             → 自动挑 n 个最对称的 sp2 位点
    · force=True                → 不管原来怎么写，全部重选
    """
    have = count_stars(smiles)
    if have == n and not force:
        if not allow_sp3:
            for r in diagnose_stars(smiles):
                if r["nh"] >= 2:
                    raise ValueError(
                        f"{role}的连接位是 sp3：原子 {r['idx']} 是 {r['sym']}，"
                        f"挂着 {r['nh']} 个氢（CH2/CH3），四面体构型，接不出平面层。\n"
                        f"  那是你在 * 后面多写了一个 C —— 它是个真实碳，"
                        f"不是占位符，RDKit 会自动给它补满 H，于是就成了 CH2。\n"
                        f"  解决办法：把这个 C 删掉，让 * 直接连芳环碳，例如\n"
                        f"    [*]CC1=NC([*])=NC([*])=N1   →   [*]c1nc([*])nc([*])n1\n"
                        f"  或者干脆不加 *，只给分子 SMILES，让脚本自动选位：\n"
                        f"    --node \"c1ncncn1\" --arms 3\n"
                        f"  如果就是要按这个写法接、H 自己到 MS 里补，加 --allow-sp3。")
        return smiles, f"沿用你给的 {n} 个 *"

    plain = strip_stars(smiles)
    if force and have:
        bad = [r for r in diagnose_stars(smiles) if r["nh"] >= 2]
        if verbose and bad:
            print(f"[自动选位] 丢弃原来的 {have} 个 *"
                  f"（其中 {len(bad)} 个挂在 sp3 碳上），按分子本身重新选位")

    sites, diag = pick_sites(plain, n, verbose=verbose)
    new = star_smiles(plain, sites)
    if verbose:
        print(f"[自动选位] {role}: {plain}  →  {new}")
    return new, diag


def inspect_molecule(smiles):
    """体检：把每个重原子的杂化/氢数/可连接性列出来，并直接给出可用 SMILES。"""
    plain = strip_stars(smiles)
    mol = prune_star_chains(Chem.MolFromSmiles(plain))
    print(f"\n分子: {plain}")
    print(f"  重原子 {mol.GetNumAtoms()} 个，"
          f"氢 {sum(a.GetTotalNumHs() for a in mol.GetAtoms())} 个（全部自动补齐）\n")
    print("  序号  元素  杂化    H数  配位  能否当连接位")
    print("  " + "-" * 46)
    mark = {}
    for s in scan_sites(plain):
        mark[s["idx"]] = s["tier"]
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in mark:
            tag = f"✓ {_TIER_TEXT[mark[i]]}"
        elif a.GetTotalNumHs() >= 2 and a.GetAtomicNum() == 6:
            tag = f"✗ {a.GetTotalNumHs()} 个 H，四面体"
        elif a.GetAtomicNum() == 6 and a.GetDegree() == 1:
            tag = "✗ 甲基/端基"
        else:
            tag = "✗ 已被取代"
        print(f"  {i:>4}  {a.GetSymbol():<4}  {_hyb_text(a):<5}  "
              f"{a.GetTotalNumHs():>3}  {a.GetDegree():>4}  {tag}")

    print()
    for n, label in ((3, "3 连接节点（hcb）"), (4, "4 连接节点（sql）"),
                     (2, "2 连接连接子")):
        try:
            sites, diag = pick_sites(plain, n)
            print(f"  {label:<18} {star_smiles(plain, sites)}")
            print(f"  {'':<18} （{diag}）")
        except ValueError as e:
            print(f"  {label:<18} 不可用：{str(e).splitlines()[0]}")
    print("\n直接改用上面任意一行即可，例如：")
    print('  python build_cof_smiles.py --node "<节点>" --linker "<连接子>" --bond 1.47\n')


# ---------------------------------------------------------------- 自动缩合
def _conn_atoms(mol):
    return [a.GetNeighbors()[0].GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _is_amine(mol):
    """连接原子全是带 H 的胺氮。

    注意 ``*N`` 里的 N 只有 1 个氢（另一个价位被连接位占了），
    真正的 -NH2 是 "N + 连接键 + 2 个 H" 里的那 2 个中剩下的 1 个。
    """
    ca = _conn_atoms(mol)
    return bool(ca) and all(
        mol.GetAtomWithIdx(c).GetAtomicNum() == 7
        and mol.GetAtomWithIdx(c).GetTotalNumHs() >= 1 for c in ca)


def _is_aldehyde(mol):
    """连接原子全是醛基碳（C=O + 1 个 H）。"""
    ca = _conn_atoms(mol)
    if not ca:
        return False
    for c in ca:
        a = mol.GetAtomWithIdx(c)
        if a.GetAtomicNum() != 6:
            return False
        has_o = any(b.GetOtherAtom(a).GetAtomicNum() == 8
                    and b.GetBondTypeAsDouble() == 2.0 for b in a.GetBonds())
        if not has_o:
            return False
    return True


def condense_imine(node, linker):
    """把 -NH2 + O=CH- 自动缩合成 -N=CH-（亚胺 COF）。

    很多文献给的单体是"缩合前"的形态（三胺 + 三醛）。直接拿去建模会多出
    2 个 H（胺）和 1 个 O（醛）。这里只要检测到一端全是 NH2、另一端全是
    醛基，就自动改成缩合后的形态：N 上 0 个 H，醛碳留 1 个 H 并去掉 =O。

    返回 (node, linker, 提示语)；未触发时提示语为空。
    """
    ma, mb = Chem.MolFromSmiles(node), Chem.MolFromSmiles(linker)
    if ma is None or mb is None:
        return node, linker, ""
    ma, mb = prune_star_chains(ma), prune_star_chains(mb)

    if _is_amine(ma) and _is_aldehyde(mb):
        amine, ald, first = ma, mb, "节点"
    elif _is_amine(mb) and _is_aldehyde(ma):
        amine, ald, first = mb, ma, "连接子"
    else:
        return node, linker, ""

    # 胺：NH2 → N（去掉两个 H，sp2 共轭）
    rw = Chem.RWMol(amine)
    for c in _conn_atoms(amine):
        a = rw.GetAtomWithIdx(c)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetHybridization(Chem.HybridizationType.SP2)
    amine2 = rw.GetMol()
    Chem.SanitizeMol(amine2)

    # 醛：-C(=O)H → -CH-（摘掉 O，保留 1 个 H）
    rw = Chem.RWMol(ald)
    drop = []
    for c in _conn_atoms(ald):
        for b in rw.GetAtomWithIdx(c).GetBonds():
            o = b.GetOtherAtom(rw.GetAtomWithIdx(c))
            if o.GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2.0:
                drop.append(o.GetIdx())
    for i in sorted(set(drop), reverse=True):
        rw.RemoveAtom(i)
    ald2 = rw.GetMol()
    for c in _conn_atoms(ald2):
        a = ald2.GetAtomWithIdx(c)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1)
        a.SetHybridization(Chem.HybridizationType.SP2)
    Chem.SanitizeMol(ald2)

    sn, sl = Chem.MolToSmiles(amine2), Chem.MolToSmiles(ald2)
    if first == "节点":
        node2, linker2 = sn, sl
    else:
        node2, linker2 = sl, sn
    note = (f"检测到 {'节点' if first == '节点' else '连接子'} 上是 -NH2、"
            f"另一个是 -CHO，已自动按亚胺缩合（-NH2 + O=CH- → -N=CH- + H2O）处理")
    return node2, linker2, note


# ---------------------------------------------------------------- 单体几何
def outward_dirs(pos, sym, conn, cap, cutoff=1.75, normal=None):
    """推算每个连接原子在装配时"新键"的指向。

    比直接用封端氢方向可靠得多：封端会让 sp2 碳变成 sp3（120° → 109°），
    亚胺 N 更是会歪到 70° 以上，导致晶胞常数算错、小晶胞直接撞车。

    规则：
      · 连接原子已有 ≥2 个重原子邻居 → 取这些键的负矢量和（sp2 缺失方向）；
      · 只有 1 个重原子邻居 → 把"指向邻居"的键绕**节点平面法向**转 ±120°。

    最后这条原来绕的是"邻居所在芳环的法向"。对平躺的苯环两者等价，但卟啉
    的 meso 苯环是垂直于层的（邻位 H 与吡咯 β-H 只隔 0.73 Å，转平就撞），
    此时环法向躺在**层内**，绕它转 120° 会把新键甩到层外——实测 od 的 z 分量
    达 ±0.85，反推出的倾角是 23.7°（正确值 60°），亚胺 Ar-N=C 变成了 157.8°。
    亚胺的 sp2 平面由共轭决定，与芳环自身扭转无关：把 Ar-N 键在层内转 120°
    才是化学上正确的第三条键。所以这里显式接收节点平面法向。
    """
    n = len(sym)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nbr = [[j for j in range(n) if j != i and sym[j] != "H" and d[i, j] < cutoff]
           for i in range(n)]
    out = {}
    for c in conn:
        nb = nbr[c]
        if len(nb) >= 2:
            v = -sum(unit(pos[j] - pos[c]) for j in nb)
            if np.linalg.norm(v) < 1e-3:
                v = cap[c]
        elif len(nb) == 1:
            m = nb[0]
            u_in = unit(pos[m] - pos[c])
            ring = [j for j in nbr[m] if j != c]
            if len(ring) >= 2:
                if normal is not None:
                    nz = unit(normal)
                else:
                    nz = unit(np.cross(pos[ring[0]] - pos[m], pos[ring[1]] - pos[m]))
                cand = [unit(rot_axis(nz, a) @ u_in) for a in (120.0, -120.0)]
                # 有封端氢时按"和氢同侧"选支（保持分子原有构象）；没有氢
                # （亚胺 N 这类）两支只是镜面对称，倾角大小一样，取第一支即可
                cuff = unit(cap[c]) if float(np.linalg.norm(cap[c])) > 0.5 else None
                v = (max(cand, key=lambda x: float(np.dot(x, cuff)))
                     if cuff is not None else cand[0])
            else:
                v = cap[c]
        else:
            v = cap[c]
        out[c] = unit(v)
    return out


def node_frame(pos, conn, od):
    """把单体摆到 z=0 平面、以连接原子质心为原点。

    返回 (坐标, 各连接原子的径向, 各连接原子的外伸方向, 平均半径)。
    """
    c = pos[conn].mean(axis=0)
    p0 = pos - c
    dirs = [unit(p0[i]) for i in conn]
    r = float(np.mean([np.linalg.norm(p0[i]) for i in conn]))
    nz = (unit(np.linalg.svd(p0)[2][2]) if len(conn) == 2
          else unit(np.cross(dirs[0], dirs[1])))
    Rp = rot_from_to(nz, np.array([0.0, 0.0, 1.0]))
    return ((Rp @ p0.T).T, [unit(Rp @ d) for d in dirs],
            {k: Rp @ v for k, v in od.items()}, r)


def to_vec(deg):
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])


def place(p1, dirs, conn, targets, offset):
    """把单体转到目标连接方向。

    枚举所有候选转角取总误差最小者，顺带解决 SMILES 里 * 的书写顺序
    与目标方向的对应关系（不需要单体自身有对称性假设）。
    返回 (坐标, 各目标方向对应的连接原子索引, 平均角偏差, 转角)。

    目标方向与连接原子必须**一对一**。早先的写法是对每个目标方向各自取
    最近的连接原子，臂角不标准时（比如三条臂是 100/120/140）会出现两条
    目标方向抢同一个臂、另一条臂空着；于是装配用的平移量退化成
    "一条臂的位移 ×3"，第二个单体会被直接摞到第一个单体身上，
    出来一个所有原子 z 都等于 0.5、看着完全共面、实际间距 0.3 Å 的废结构。
    """
    from itertools import permutations

    tg = [to_vec(t) for t in targets]
    n = len(tg)
    m = len(conn)
    if m < n:
        raise ValueError(f"单体只有 {m} 个连接位，需要 {n} 个")

    best = None
    for d in dirs:
        a_d = np.degrees(np.arctan2(d[1], d[0]))
        for t in targets:
            delta = (t - a_d) % 360.0
            R = rot_z(delta)
            rd = [unit(R @ v) for v in dirs]
            # 代价矩阵：第 k 个目标方向 ↔ 第 j 个连接原子 的夹角
            cost = [[float(np.degrees(np.arccos(np.clip(np.dot(tg[k], rd[j]), -1, 1))))
                     for j in range(m)] for k in range(n)]
            # 连接位数很少（3~4），直接枚举一一对应
            for perm in permutations(range(m), n):
                e = sum(cost[k][perm[k]] for k in range(n))
                if best is None or e < best[0]:
                    best = (e, delta, rd, perm)
    err, delta, rd, perm = best
    p = (rot_z(delta) @ p1.T).T + np.asarray(offset, dtype=float)
    assign = [conn[j] for j in perm]
    return p, assign, err / n, delta


def tilt_mag(pos, assign, od, targets):
    """各连接原子的外伸键相对径向的倾角大小（度，取绝对值均值）。

    符号不可靠：封端氢会落在 sp2 的两个等价位置上，各条臂随机选边，
    所以只取大小，装配时再用统一手性重建方向。
    """
    angs = []
    for t, ci in zip(targets, assign):
        tg = to_vec(t)
        v = unit(od[ci])
        angs.append(abs(np.degrees(np.arctan2(float(np.cross(tg, v)[2]),
                                              float(np.dot(tg, v))))))
    return float(np.mean(angs))


# ---------------------------------------------------------------- 主构建
# 倾斜装配时，连接键方向朝 od 靠的比例（1 = 严格 sp2，0 = 沿臂轴）。
# 小一点 = 连接子沿臂轴伸得更远，短臂/大侧基的节点不容易撞；但 sp2 角度更歪。
# cof_builder 会按连接类型覆盖它（烯烃 0.55、酰胺/酰亚胺 0.70）。
PAIR_SOFT = 0.70

# 拓扑 → 节点连接数
TOPOLOGIES = {"hcb": 3, "sql": 4}
# 拓扑下每条臂的目标方位角
ARM_ANGLES = {3: [90.0, 210.0, 330.0], 4: [0.0, 90.0, 180.0, 270.0]}


def build(node_smiles, linker_smiles, topology="auto", bond=1.34, c=3.60,
          scale_a=1.0, tilt=None, condense=True, verbose=True, allow_sp3=False):
    """搭建二维 COF 单层周期结构，返回 (ase.Atoms, 信息字典)。

    按两个单体的连接数自动选择装配模式：

      edge    节点 n 连接 + 线形连接子（2 连接），连接子架在两条臂中间
      vertex  两个单体连接数相同（C3+C3、C4+C4），直接成键，
              交替占据蜂窝 / 四方网格的两类顶点（没有独立的连接子）

    topology: "hcb"（蜂窝，3 连接）、"sql"（四方，4 连接）、"auto"（按节点判断）
    tilt:     连接键相对节点连线（径向）的倾角，单位度。
              默认 None = 由连接原子的化学环境自动推算（sp2 连接原子 60°，
              卟啉 meso 位 0°）。量错会让每条臂整体平移，小晶胞直接撞在一起。
    condense: 是否自动把 -NH2 / -CHO 缩合成亚胺（真 = 允许自动处理）
    """
    if condense:
        node_smiles, linker_smiles, note = condense_imine(node_smiles, linker_smiles)
        if note and verbose:
            print(f"[提示] {note}")

    nA = count_stars(node_smiles)
    nB = count_stars(linker_smiles)

    if topology == "auto":
        topology = {3: "hcb", 4: "sql"}.get(nA)
        if topology is None:
            raise ValueError(f"节点有 {nA} 个连接位，无法判断拓扑；"
                             f"目前支持 3 连接（hcb）和 4 连接（sql）")
    if topology not in TOPOLOGIES:
        raise ValueError(f"不支持的拓扑 '{topology}'，可选: {list(TOPOLOGIES)}")
    if TOPOLOGIES[topology] != nA:
        raise ValueError(f"拓扑 {topology} 需要 {TOPOLOGIES[topology]} 连接节点，"
                         f"但节点有 {nA} 个连接位")

    mode = "vertex" if nB == nA else "edge"
    if mode == "edge" and nB != 2:
        raise ValueError(f"连接子有 {nB} 个连接位：做线形连接子需要 2 个，"
                         f"做第二种节点需要和节点相同的 {nA} 个")

    # ---------- 两个单体的局部坐标系 ----------
    def prep(smiles, n, flatten=False):
        sym, pos, conn, n_h, cap = prepare(smiles, n, flatten=flatten)
        if n >= 3:
            project_planar(pos, sym, verbose=True)
        # 节点平面法向：由两条臂的方向叉乘（四条臂共面时就是层法向）。
        # 传给 outward_dirs 用来定"只有 1 个重原子邻居"的连接原子的第三条键，
        # 避免它跟着扭转的芳环跑到层外去（见 outward_dirs 的说明）。
        _nrm = None
        if n >= 3:
            _cc = pos[conn].mean(axis=0)
            _nrm = unit(np.cross(unit(pos[conn[0]] - _cc),
                                 unit(pos[conn[1]] - _cc)))
        od = outward_dirs(pos, sym, conn, cap, normal=_nrm)
        if allow_sp3:
            # CH2/CH3 型的连接位只有 1 个重原子邻居，outward_dirs 会按"sp2 的
            # 第三个取代基"给一条偏 120° 的方向 —— 那是把臂掰弯了 60°，小晶胞
            # 里两条臂互相顶死。既然这里的 H 已经决定丢掉（交给 MS 补），
            # 就把这条臂当"二配位碳"处理：沿原键延长，臂是直的。
            for c, h in zip(conn, n_h):
                if h < 2:
                    continue
                nb = [j for j in range(len(sym))
                      if sym[j] != "H" and j != c
                      and np.linalg.norm(pos[j] - pos[c]) < 1.75]
                if len(nb) == 1:
                    od[c] = unit(pos[c] - pos[nb[0]])
        p1, dirs, od, r = node_frame(pos, conn, od)
        return dict(sym=sym, p1=p1, dirs=dirs, od=od, r=r, n_h=n_h, conn=conn)

    fA = prep(node_smiles, nA, flatten=(nA >= 3))
    fB = prep(linker_smiles, nB, flatten=(mode == "edge" or nB >= 3))

    # 连接原子必须是 sp2（平面三角）。连着两个氢就是 CH2/CH3，四面体构型，
    # 两个氢必然一个朝上一个朝下 —— 这种单体无论怎么摆都做不出平面层。
    for _f, _label, _smi in ((fA, "节点", node_smiles),
                             (fB, "连接子/第二节点", linker_smiles)):
        if max(_f["n_h"], default=0) >= 2 and not allow_sp3:
            _bad = max(_f["n_h"])
            raise ValueError(
                f"{_label}的连接位是 sp3：连接原子上挂着 {_bad} 个氢（CH2/CH3），"
                f"是四面体碳，接不出平面层。\n"
                f"  原因几乎都是 * 后面多写了一个 C —— 那是真实碳原子，不是占位符，\n"
                f"  RDKit 会按价态自动补 H，于是成了 CH2。\n"
                f"  三种改法，任选其一：\n"
                f"    (1) 删掉那个 C，* 直接连芳环：\n"
                f"        [*]CC1=NC([*])=NC([*])=N1  →  [*]c1nc([*])nc([*])n1\n"
                f"    (2) 干脆不写 *，只给分子本身，让脚本自动选位：\n"
                f"        --node \"c1ncncn1\" --arms 3\n"
                f"    (3) 让脚本丢掉你写的 * 重新选：\n"
                f"        --node \"{_smi}\" --fix-sites\n"
                f"  先用 --inspect \"<分子SMILES>\" 可以看清每个原子到底几个氢。\n"
                f"  如果你就是要按这个写法接、H 自己到 MS 里补，加 --allow-sp3 放行。")
    if allow_sp3:
        _n_sp3 = sum(1 for _f in (fA, fB)
                     for _h in _f["n_h"] if _h >= 2)
        if _n_sp3:
            # 那两个 H 不能补回来：sp3 碳本来只有 1 个重原子邻居，补 H 时按
            # "平面三角"摆放，正好把位置让给了新接进来的键，实测 H–新键 0.33 Å。
            # 这里干脆留空（2 配位碳），进 MS 之后 Adjust Hydrogens 一次就补齐。
            fA["n_h"] = [0 if _h >= 2 else _h for _h in fA["n_h"]]
            fB["n_h"] = [0 if _h >= 2 else _h for _h in fB["n_h"]]
            print(f"[放行] {_n_sp3} 个 sp3 连接位按平面三角摆放；"
                  f"那些碳上原来的 H 不再自动补（会和进来的新键撞在同一个方向），"
                  f"进 MS 后用 Adjust Hydrogens 补齐即可。")

    tA = ARM_ANGLES[nA]
    rhomb = 90.0
    if topology == "sql":
        # 四连接节点的四条臂不一定是 90°/90°/180°。芘-1,3,6,8（PyTTA 型）、
        # 四苯基乙烯这类"菱形"节点实测是 62°/118°：两个方向各一对共线臂，
        # 两轴夹角只有 60 多度。硬套 90° 会把每条臂拧 14° 以上，臂和连接子
        # 直接撞在一起（最近非键距离掉到 1.9 Å）。
        # 这里按实测的相邻臂夹角取**斜方（oblique）胞**；正四方的节点
        # （卟啉、四苯基甲烷型）测出来仍是 90°，结果跟以前完全一致。
        ang = np.sort(np.array([np.degrees(np.arctan2(d[1], d[0])) % 360.0
                                for d in fA["dirs"]]))
        rhomb = float(np.diff(np.append(ang, ang[0] + 360.0)).min())
        if abs(rhomb - 90.0) < 4.0:      # 本来就是正四方，别为了 3° 的构象噪声搞成斜方
            rhomb = 90.0
        tA = [0.0, rhomb, 180.0, 180.0 + rhomb]
    pA, asgA, eA, dlA = place(fA["p1"], fA["dirs"], fA["conn"], tA, [0.0, 0.0, 0.0])
    odA = {k: rot_z(dlA) @ v for k, v in fA["od"].items()}   # 转到装配后的朝向

    # ---------- 模式一：两个节点直接成键（C3+C3 / C4+C4） ----------
    if mode == "vertex":
        tB = [t + 180.0 for t in tA]
        pB0, asgB, eB, dlB = place(fB["p1"], fB["dirs"], fB["conn"], tB, [0.0, 0.0, 0.0])
        odB = {k: rot_z(dlB) @ v for k, v in fB["od"].items()}

        tA_mag = tilt_mag(pA, asgA, odA, tA)
        tB_mag = tilt_mag(pB0, asgB, odB, tB)

        # 统一手性：对称节点的每条臂必须朝同一侧偏，否则各臂位移向量对不上。
        # 两种手性给出互为镜像的两套结构，取不打架的那个。
        def oriented(assign, targets, mag, sign):
            return {ci: unit(rot_z(sign * mag) @ to_vec(t))
                    for t, ci in zip(targets, assign)}

        def assemble(sign):
            oA = oriented(asgA, tA, tA_mag, sign)
            oB = oriented(asgB, tB, tB_mag, sign)
            deltas = np.array([pA[asgA[k]] + bond * oA[asgA[k]] - pB0[asgB[k]]
                               for k in range(len(tA))])
            Ls = np.linalg.norm(deltas[:, :2], axis=1)
            delta = deltas[0]                     # 同一晶胞里只放这一个 B
            pB = pB0 + delta

            # 晶胞：a1 = δ - Rθδ，a2 = Rθ a1 （γ 恒为 θ，hcb 120° / sql 90°）
            th = 120.0 if topology == "hcb" else 90.0
            a1 = delta - rot_z(th) @ delta
            a2 = rot_z(th) @ a1
            a = float(np.linalg.norm(a1))
            ang = np.degrees(np.arctan2(a1[1], a1[0]))
            R = rot_z(-ang)                       # 把 a1 转到 x 轴，方便读 CIF
            pA_r, pB_r = (R @ pA.T).T, (R @ pB.T).T
            a1r, a2r = R @ a1, R @ a2
            a1r, a2r = a1r * scale_a, a2r * scale_a

            positions = np.vstack([pA_r, pB_r])
            symbols = fA["sym"] + fB["sym"]
            cell = np.array([[a1r[0], a1r[1], 0.0],
                             [a2r[0], a2r[1], 0.0],
                             [0.0, 0.0, c]])
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            conn_spec = ([(ci, h) for ci, h in zip(fA["conn"], fA["n_h"])] +
                         [(len(fA["sym"]) + ci, h)
                          for ci, h in zip(fB["conn"], fB["n_h"])])
            info = dict(topology=topology, mode="vertex", L=float(Ls.mean()),
                        a=a * scale_a, c=c, r_node=0.5 * (fA["r"] + fB["r"]),
                        len_linker=0.0, sym_err=max(eA, eB),
                        tilt=0.5 * (tA_mag + tB_mag), bond=bond,
                        tilt_mode="manual",
                        thickness=0.0, c_grown=False,
                        delta_spread=float(Ls.max() - Ls.min()),
                        r_a=fA["r"], r_b=fB["r"],
                        tilt_a=tA_mag, tilt_b=tB_mag)
            return atoms, info, conn_spec

        best = None
        for sign in (+1, -1):
            cand, ci_, spec = assemble(sign)
            d = cand.get_all_distances(mic=True)
            np.fill_diagonal(d, 99.0)
            sc = float(d.min())
            if best is None or sc > best[0]:
                best = (sc, cand, ci_, spec)
        _, atoms, info, conn_spec = best

        # 三条臂算出来的平移量必须一致，否则说明两个单体的臂根本配不上：
        # 此时 B 只按第 1 条臂对齐，另外两条臂的原子会插进 A 里面。
        if info["delta_spread"] > 0.50:
            raise ValueError(
                f"两个节点的臂接不上：三条臂各自要求的平移量相差 "
                f"{info['delta_spread']:.2f} Å（正常应 < 0.1 Å）。\n"
                f"节点半径 {info['r_a']:.2f} / {info['r_b']:.2f} Å，"
                f"连接方向平均偏差 {info['sym_err']:.1f}°。\n"
                f"通常是其中一个单体的三条臂不是 120°，"
                f"或两边连接原子的化学环境差太多（一边 sp2、一边 sp3）。")

    # ---------- 模式二：节点 + 线形连接子 ----------
    else:
        rod = fB["p1"][fB["conn"][1]] - fB["p1"][fB["conn"][0]]
        len_linker = float(np.linalg.norm(rod))
        rod_dir = unit(rod)
        l_normal = unit(np.linalg.svd(fB["p1"] - fB["p1"].mean(axis=0))[2][2])
        r_node = fA["r"]
        zaxis = np.array([0.0, 0.0, 1.0])

        # ---- 连接子的"理想 N 位置"：C_conn + bond × 外伸方向 ----
        # 这两个点就是装配后两侧节点的 N 应该落在的位置。用它当锚点，
        # 而不是"把 C=N 键硬摆到臂轴线上"：
        #   · 硬摆：两端 C=N 同侧（顺式），苯环被整体横推 bond·sin60 ≈ 1.16 Å，
        #     连接子只有镜面（Cs），整层最多 Cm；
        #   · 锚定：按 sp2 几何自然给出反式，两个 N 分居苯环两侧、苯环居中，
        #     连接子有对称中心（Ci），整层才有可能四重。
        _c0i, _c1i = fB["conn"][0], fB["conn"][1]
        _od0 = unit(np.asarray(fB["od"][_c0i], dtype=float))
        _od1 = unit(np.asarray(fB["od"][_c1i], dtype=float))
        _ep = unit(np.cross(zaxis, rod_dir))          # 面内、垂直长轴
        _flipped = False
        if float(np.dot(_od0, _ep)) * float(np.dot(_od1, _ep)) > 0.0:
            # 两端外伸方向同侧 → 把远端镜面翻过来，取低能的"全反式"
            _od1 = _od1 - 2.0 * float(np.dot(_od1, _ep)) * _ep
            fB["od"][_c1i] = _od1
            _flipped = True
        nloc = np.array([fB["p1"][_c0i] + bond * _od0,
                         fB["p1"][_c1i] + bond * _od1])
        nn_vec = nloc[1] - nloc[0]
        nn_len = float(np.linalg.norm(nn_vec))
        nn_dir = unit(nn_vec)
        # 实际倾角 = N→C 键与臂轴（N–N 连线）的夹角
        _nc = unit(fB["p1"][_c0i] - nloc[0])
        tilt_geo = float(np.degrees(np.arccos(np.clip(float(np.dot(_nc, nn_dir)),
                                                      -1.0, 1.0))))

        tilt_auto = tilt is None

        # 节点侧连接原子自己带着"留下来的双键取代基"时（酰胺 -C(=O)-NH-、
        # 羧酸酯、酮类节点），连接键必须沿那条 sp2 的第三方向走，不能沿臂轴。
        # 沿臂轴摆的后果不是"稍微不对"：实测 N 直接撞进 C=O 里，N–O 只有
        # 0.43 Å，∠O–C–N 变成 18°（应为 120°），整个结构作废。
        # 判据要两个条件同时成立 ——
        #   · 连接原子有 ≥2 个重原子邻居（od 由真实键决定，而非几何构造）；
        #   · od 确实偏离径向 >15°。
        # 只判第一条会误伤 Suzuki（硼酸连着环碳，od 沿径向、不偏），
        # 只判第二条会误伤亚胺（氨基 N 只有一个重邻居，60° 是几何构造出来的，
        # 反式锚定的 link_geo 本来就是为它写的）。
        #
        # 计数必须排除连接原子自身：`p1[i] - p1[_c]` 在 i == _c 处是 0，
        # 稳稳落在 1.75 Å 阈值内，于是每个连接原子都白多算一个邻居。
        # 后果不是"多个一两度"——氨基 N 的真实重邻居只有 1 个（环碳），
        # 被数成 2 之后 `>= 2` 成立，亚胺整个被当成酰胺式的分叉碳，
        # 套上 tilt=60° 的倾斜装配：连接子苯环被逼着在层内偏 17.6°，
        # 质心横向甩出 1.1 Å，四条臂的方位从 ±0.4° 放大到 ±5°
        # （MUL-0012 就是这样变成 80°/100° 的）。
        _nn_arr = np.arange(len(fA["sym"]))
        _n_nb = []
        for _c in fA["conn"]:
            _dv = np.linalg.norm(fA["p1"] - fA["p1"][_c], axis=1)
            _n_nb.append(int(np.sum((_dv < 1.75) & (_dv > 1e-6)
                                   & (_nn_arr != _c)
                                   & (np.array(fA["sym"]) != "H"))))
        _tilt_node = tilt_mag(pA, asgA, odA, tA)
        _slanted = max(_n_nb) >= 2 and _tilt_node > 15.0

        # 连接子苯环在层内的偏转：理想情况下 N–N 轴就是苯环对位轴，两者同向；
        # 反式构象下 N 被推到环的两侧，N–N 轴相对环轴偏开了这个角度。
        _skew = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(rod_dir, nn_dir))),
                                                   0.0, 1.0))))
        if tilt_auto:
            if _slanted:
                # 连接键本身是斜的：每条臂各自按自己的 od 装配（link_od），
                # 不再"反推一个统一倾角 ±两支试"——单体对称性一破，
                # 统一倾角就会把某条臂接到隔壁的羰基氧上。
                tilt_cands = ["pair"]
                if verbose:
                    print(f"[提示] 节点连接原子的 sp2 第三方向偏离臂轴 "
                          f"{_tilt_node:.1f}°，连接键按倾斜模型装配（±两支取优）")
            else:
                # 默认不再"反推一个倾角再硬摆"：倾角由连接子自己的 sp2 几何给出。
                # 同时把"逐臂 sp2 定向"也建一遍 —— 形状不规整的连接子（PMDA
                # 那种五元酐环）用锚定会被甩歪，交给质量评分去挑。
                tilt_cands = [None, "pair"]
            if verbose:
                print(f"[提示] 连接子按几何锚定（两端"
                      f"{'已取反式' if _flipped else '本为反式'}，Ci 对称）：")
                print(f"       N–N 端距 {nn_len:.3f} Å（C…C {len_linker:.3f} + "
                      f"2×{bond:.2f} 反式外伸）；苯环在层内偏转 {_skew:.2f}°")
                print(f"       连接子侧 ∠Ar–C–N = 120.00°（sp2 精确）；"
                      f"节点侧 ∠Ar–N=C = {180.0 - tilt_geo:.1f}°"
                      f"（N=C 偏离臂轴 {tilt_geo:.1f}°，理想 60.0°）")
        else:
            tilt_cands = [tilt]

        def link_geo(d, S):
            """把连接子的"理想 N"对到节点连接点 S，长轴沿 d。"""
            Rt = rot_from_to(nn_dir, d)
            n_a = unit(Rt @ l_normal)
            th = np.degrees(np.arctan2(float(np.dot(np.cross(n_a, zaxis), d)),
                                       float(np.dot(n_a, zaxis))))
            Rtot = rot_axis(d, th) @ Rt
            return (Rtot @ (fB["p1"] - nloc[0]).T).T + S

        def link_tilt(d, S, tl, ci=None):
            """旧路径：--tilt 显式指定倾角时用。"""
            Rt = rot_from_to(rod_dir, d)
            n_a = unit(Rt @ l_normal)
            th = np.degrees(np.arctan2(float(np.dot(np.cross(n_a, zaxis), d)),
                                       float(np.dot(n_a, zaxis))))
            Rtot = rot_axis(d, th) @ Rt
            c1 = S + bond * (rot_z(tl) @ d)
            return (Rtot @ (fB["p1"] - fB["p1"][_c0i]).T).T + c1

        def link_od(d, S, ci):
            """按**该条臂自己**的 sp2 第三方向接连接子（自动路径）。

            以前这里用一个全局倾角 ±tilt 同时套三条臂，前提是三条臂的 od
            相对各自臂轴的偏转完全一样。单体一旦对称性破缺（均苯三甲酸的
            三个羧基、PMDA 的两个酐），其中一条臂的 od 就落在反面，
            连接子被推到隔壁臂的羰基氧上——实测 N–O 只有 0.13 Å。
            逐臂取 od 不会出现这个问题，也省掉了"±两支试一遍"。
            """
            u = unit(np.asarray(odA[ci], dtype=float))
            Rt = rot_from_to(rod_dir, d)
            n_a = unit(Rt @ l_normal)
            th = np.degrees(np.arctan2(float(np.dot(np.cross(n_a, zaxis), d)),
                                       float(np.dot(n_a, zaxis))))
            Rtot = rot_axis(d, th) @ Rt
            return (Rtot @ (fB["p1"] - fB["p1"][_c0i]).T).T + (S + bond * u)

        def soft_od(ci, d):
            """把该臂的 od 往臂轴方向收一点，再返回单位矢量。

            严格按 sp2 的 120° 接，酰胺的 N 会被推到节点半径以内（连接子
            沿臂轴的投影缩短 2·bond·sin(tilt) ≈ 2.3 Å），苯环氢直接插进
            羰基氧（实测 H–O 0.74 Å）。真实分子里酰胺键可绕 C–N 轻微转动，
            不会死守 120°。取 0.7 是在"化学角度"和"不打架"之间折中：
            ∠O–C–N 约 145°，进势函数弛豫后回到 120° 附近。
            """
            u0 = unit(np.asarray(odA[ci], dtype=float))
            c = float(np.clip(np.dot(u0, d), -1.0, 1.0))
            ep = unit(np.cross(zaxis, d))
            sn = float(np.dot(u0, ep))
            T = float(np.arctan2(sn, c)) * PAIR_SOFT
            return unit(np.cos(T) * d + np.sin(T) * ep)

        def link_pair(d, S, ci, D=None):
            """两端各按自己的 sp2 方向接（真正对称的倾斜装配）。

            蜂窝网里第二个节点是第一个节点**转 180°**再平移，所以它的 od
            恰好是 -od。想让两根连接键都严格落在各自 od 上，连接子的
            两个连接原子就不能停在"沿臂轴"的位置，而要沿垂直方向张开
            2·bond·sin(tilt)。张开之后 N–N 距离变了，节点间距必须跟着改，
            这个 D 由余弦定理解出来（见 assemble 里的 _D）。

            旧写法强行让两端连接键平行，等于要求 od_B = od_A；只有 tilt=0
            才成立。酰胺（tilt 60.8°）因此把 N 推到隔壁臂的羰基氧上，
            实测 N–O 0.13 Å。亚胺 tilt 只有 19.5°，误差小到看不出来，
            所以这个问题一直藏着。
            """
            u = soft_od(ci, d)
            n0 = S + bond * u
            n1 = S + D * d - bond * u
            tgt = unit(n1 - n0)
            Rt = rot_from_to(rod_dir, tgt)
            n_a = unit(Rt @ l_normal)
            th = np.degrees(np.arctan2(float(np.dot(np.cross(n_a, zaxis), tgt)),
                                       float(np.dot(n_a, zaxis))))
            Rtot = rot_axis(tgt, th) @ Rt
            return (Rtot @ (fB["p1"] - fB["p1"][_c0i]).T).T + n0

        def assemble(tl):
            if tl is None:
                link = lambda d, S, ci: link_geo(d, S)        # noqa: E731
                L = 2.0 * r_node + nn_len
                _tilt_used = tilt_geo
            elif tl == "pair":
                _cos, _Ds = [], []
                for _t, _ci in zip(tA, asgA):
                    _dd = to_vec(_t)
                    _u = soft_od(_ci, _dd)
                    _c = float(np.dot(_u, _dd))
                    _sn = float(np.dot(_u, unit(np.cross(zaxis, _dd))))
                    _cos.append(_c)
                    _rad = len_linker ** 2 - 4.0 * bond ** 2 * _sn ** 2
                    _Ds.append(2.0 * bond * _c + np.sqrt(max(_rad, 0.25)))
                _D = float(np.mean(_Ds))
                link = lambda d, S, ci: link_pair(d, S, ci, _D)   # noqa: E731
                L = 2.0 * r_node + _D
                _tilt_used = float(np.degrees(np.arccos(
                    np.clip(float(np.mean(_cos)), -1.0, 1.0))))
            else:
                link = lambda d, S, ci: link_tilt(d, S, tl)   # noqa: E731
                L = 2.0 * r_node + len_linker + 2.0 * bond * np.cos(np.radians(tl))
                _tilt_used = float(tl)
            if topology == "hcb":
                pB, _, eB, _ = place(fA["p1"], fA["dirs"], fA["conn"],
                                     [t + 180.0 for t in tA], [0.0, L, 0.0])
                frames = [pA, pB]
                parts = [link(to_vec(t), pA[ci], ci) for t, ci in zip(tA, asgA)]
                errs = [eA, eB]
                a = L * np.sqrt(3.0) * scale_a
                cell = np.array([[a, 0.0, 0.0],
                                 [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
                                 [0.0, 0.0, c]])
            else:                                   # sql
                frames = [pA]
                parts = [link(to_vec(t), pA[ci], ci)
                         for t, ci in zip(tA, asgA) if t < 180]
                errs = [eA]
                a = L * scale_a
                if abs(rhomb - 90.0) < 1.0:         # 正四方
                    cell = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]])
                else:                               # 斜方：两条边夹角 = 菱形锐角
                    th = np.radians(rhomb)
                    cell = np.array([[a, 0.0, 0.0],
                                     [a * np.cos(th), a * np.sin(th), 0.0],
                                     [0.0, 0.0, c]])

            # 连接子绕自身长轴翻转 180° 是等价构象（苯环对称），但端基上的
            # =CH– 氢会因此换到另一侧。氰基乙烯这类端基原子多的连接，
            # 翻错一侧就是 H···N 1.02 Å（实测）。这里逐片比较"翻转前后离
            # 节点骨架的最近距离"，取不打架的那一侧。
            if parts:
                _fp = np.vstack(frames)
                for _k in range(len(parts)):
                    _P = parts[_k]
                    _ax = unit(_P[_c1i] - _P[_c0i])
                    _ctr = 0.5 * (_P[_c0i] + _P[_c1i])
                    _Q = (rot_axis(_ax, 180.0) @ (_P - _ctr).T).T + _ctr
                    # 只比氢：连接原子本身离节点只有一根连接键（1.3~1.4 Å），
                    # 算进去的话最小值恒定不变，翻转永远触发不了。
                    _hs = [m for m, sy in enumerate(fB["sym"]) if sy == "H"]
                    if not _hs:
                        continue
                    _dP = np.linalg.norm(
                        _P[_hs][:, None, :] - _fp[None, :, :], axis=-1).min()
                    _dQ = np.linalg.norm(
                        _Q[_hs][:, None, :] - _fp[None, :, :], axis=-1).min()
                    if _dQ > _dP + 0.05:
                        parts[_k] = _Q

            symbols = fA["sym"] * len(frames) + fB["sym"] * len(parts)
            positions = np.vstack(frames + parts)
            thickness = float(positions[:, 2].max() - positions[:, 2].min())
            c_eff = max(c, thickness + 3.3)         # 层太厚就自动放宽 c
            cell[2, 2] = c_eff
            positions[:, 2] += c_eff / 2.0 - positions[:, 2].mean()
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

            spec, off = [], 0
            for _ in frames:
                spec += [(off + ci, h) for ci, h in zip(fA["conn"], fA["n_h"])]
                off += len(fA["sym"])
            for _ in parts:
                spec += [(off + ci, h) for ci, h in zip(fB["conn"], fB["n_h"])]
                off += len(fB["sym"])
            info = dict(topology=topology, mode="edge", L=L, a=a, c=c_eff,
                        r_node=r_node, len_linker=len_linker, sym_err=max(errs),
                        tilt=_tilt_used, bond=bond,
                        tilt_mode=("geo" if tl is None else
                                   ("per-arm" if tl == "pair" else "manual")),
                        thickness=thickness, rhomb=rhomb,
                        c_grown=c_eff > c + 1e-6)
            return atoms, info, spec

        # 装配模型不止一种解：几何锚定（把连接子的两端当成刚性锚点）和逐臂
        # sp2 定向（两端各按自己的 od 接，节点间距由余弦定理解出来）。
        # 哪个更好取决于单体形状 —— 亚胺用锚定就够，聚酰亚胺（PMDA）用锚定
        # 会把酐环甩进节点，实测 N–O 0.90 Å；换 sp2 定向就干净。
        # 这里两种都建一遍，按"补完氢之后"的真实质量择优：先看有没有原子
        # 重叠（<1.3 Å 一票否决），再比最小 sp2 键角。
        def _quality(a):
            from ase.data import covalent_radii, atomic_numbers
            d = a.get_all_distances(mic=True)
            np.fill_diagonal(d, 99.0)
            dmin = float(d.min())
            s = a.get_chemical_symbols()
            p = a.get_positions()
            nbq = [[] for _ in range(len(a))]
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    if 0.4 < d[i, j] < 1.25 * (covalent_radii[atomic_numbers[s[i]]]
                                               + covalent_radii[atomic_numbers[s[j]]]):
                        nbq[i].append(j)
                        nbq[j].append(i)
            ang = 180.0
            for i in range(len(a)):
                if s[i] not in ("C", "N", "O") or len(nbq[i]) != 3:
                    continue
                for x in range(3):
                    for y in range(x + 1, 3):
                        v1 = p[nbq[i][x]] - p[i]
                        v2 = p[nbq[i][y]] - p[i]
                        c0 = abs(float(np.dot(v1, v2))) / (
                            np.linalg.norm(v1) * np.linalg.norm(v2))
                        ang = min(ang, float(np.degrees(np.arccos(np.clip(c0, -1, 1)))))
            return 100.0 * min(dmin, 1.30) + 0.01 * min(ang, 120.0)

        best = None
        for tl in tilt_cands:
            cand, ci_, spec = assemble(tl)
            _h = add_conn_hydrogens(cand, spec)
            _full = cand
            if _h:
                _full = Atoms(symbols=list(cand.get_chemical_symbols())
                              + ["H"] * len(_h),
                              positions=np.vstack([cand.get_positions(),
                                                   np.array(_h)]),
                              cell=cand.cell, pbc=True)
            sc = _quality(_full)
            if best is None or sc > best[0]:
                best = (sc, cand, ci_, spec)
        _, atoms, info, conn_spec = best

        if verbose and info["c_grown"]:
            print(f"[提示] 单层厚度 {info['thickness']:.2f} Å，"
                  f"c 自动由 {c:.2f} 增大到 {info['c']:.2f} Å")

    # ---------- 补回连接原子上原有的氢 ----------
    if allow_sp3:
        # 放行模式下连接原子上的 H 全部留空：骨架按"重原子连好"交付，
        # 进 MS 后 Adjust Hydrogens 会按实际成键数补齐（三键碳补 1 个、
        # 亚甲基补 2 个，正好是你要的），比自己在这儿硬塞准得多。
        conn_spec = [(c, 0) for c, _ in conn_spec]
    extra_h = add_conn_hydrogens(atoms, conn_spec)
    if extra_h:
        atoms = Atoms(symbols=list(atoms.get_chemical_symbols()) + ["H"] * len(extra_h),
                      positions=np.vstack([atoms.get_positions(), np.array(extra_h)]),
                      cell=atoms.cell, pbc=True)

    # ---------- 连接位的氢按最终几何重摆 ----------
    # 单体自带的连接位氢是按"星号方向"嵌出来的（RDKit 保证 120°），可装配时
    # 连接键并不一定落回那个方向：亚胺的 ∠Ar–C–N 装配后是 138°，星号方向
    # 与真实 C–N 方向差 60°，于是 =CH– 的氢直接被顶到氮上（实测 H···N 1.20 Å，
    # 老版 BF-COF 那个 1.06 Å 也是同一件事）。这里把它们摘掉、按最终成键
    # 几何重摆一遍——重原子骨架不动，只改氢的方位。
    atoms = repair_conn_h(atoms, conn_spec)

    if verbose:
        print(f"=== 几何参数（拓扑: {topology}，装配: {info['mode']}）===")
        _tilt_src = {"geo": "几何锚定", "per-arm": "逐臂 sp2 定向"}.get(
            info.get("tilt_mode"), "手工指定")
        print(f"连接键倾角 tilt      = {info['tilt']:+.2f}°（{_tilt_src}）")
        print(f"节点半径 r           = {info['r_node']:.3f} Å （{nA} 连接）")
        if info["mode"] == "edge":
            print(f"连接子端距 len       = {info['len_linker']:.3f} Å")
        else:
            print(f"两个节点半径         = {info['r_a']:.3f} / {info['r_b']:.3f} Å")
            print(f"各臂节点间距偏差     = {info['delta_spread']:.3f} Å",
                  "[OK]" if info["delta_spread"] < 0.15 else "[警告: 单体不是严格对称]")
        print(f"节点间距 L           = {info['L']:.3f} Å  →  a = b = {info['a']:.3f} Å")
        print(f"单层厚度             = {info['thickness']:.3f} Å",
              "[OK]" if info["thickness"] < 1.0 else "[警告: 层不平, 调大 --c]")
        print(f"连接方向平均偏差     = {info['sym_err']:.2f}°",
              "[OK]" if info["sym_err"] < 5
              else f"[警告: 单体不是 {nA} 重对称平面构型]")
    return atoms, info


def _formula_text(atoms):
    """化学式：有 occupancy 时按占位求和（4 个 0.5 的 H 要算 2 个）。"""
    try:
        import lattice_sym as _ls
        if 'occupancy' in atoms.arrays:
            return _ls.formula(atoms)
    except ImportError:
        pass
    return atoms.get_chemical_formula()


def _formula_neff(atoms):
    """有效原子数（占位求和）；无占位时返回 None。"""
    try:
        import lattice_sym as _ls
        if 'occupancy' in atoms.arrays:
            v = _ls.n_effective(atoms)
            return v if abs(v - len(atoms)) > 0.01 else None
    except ImportError:
        pass
    return None


def report(atoms, info):
    from ase.neighborlist import NeighborList, natural_cutoffs

    cp = atoms.cell.cellpar()
    print("=== 结构检查 ===")
    print(f"晶胞: a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
          f"α={cp[3]:.2f} β={cp[4]:.2f} γ={cp[5]:.2f}")
    _neff = _formula_neff(atoms)
    _tail = (f"{len(atoms)} 个位点/等效 {_neff:.0f}" if _neff else f"{len(atoms)}")
    print(f"化学式: {_formula_text(atoms)}   原子数: {_tail}")

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)
    dmin = float(d.min())
    print(f"最近原子间距: {dmin:.3f} Å", "[OK]" if dmin > 0.9 else "[警告: 存在重叠]")

    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    lonely = [i for i in range(len(atoms)) if len(nl.get_neighbors(i)[0]) == 0]
    print(f"孤立原子: {len(lonely)}", "[OK]" if not lonely else "[警告]")
    return dmin


# ---------------------------------------------------------------- 内置预设
PRESETS = {
    # 你图中的结构：三嗪节点 + 乙烯基 + 二乙炔长臂（hcb）
    "triazine": dict(
        topology="hcb",
        node="*[CH]c1nc([CH]*)nc([CH]*)n1",
        linker="*[CH]c1ccc(C#CC#Cc2ccc([CH]*)cc2)cc1",
        bond=1.34),
    # LZU-1 型亚胺 COF：三醛基苯 + 对苯二胺（hcb）
    "lzul": dict(
        topology="hcb",
        node="*[CH]c1cc([CH]*)cc([CH]*)c1",
        linker="*[N]c1ccc([N]*)cc1",
        bond=1.28),
    # C3+C3 亚胺 COF：三(4-氨基苯基丁二炔基)苯 + 三甲酰基苯。
    # 两个单体都是三连接节点、直接成键（vertex 装配），没有独立连接子。
    # 若给的是缩合前的 -NH2 / -CHO 形态，脚本会自动缩合。
    "c3c3_imine": dict(
        topology="hcb",
        node="*[N]c1ccc(C#CC#Cc2cc(C#CC#Cc3ccc([N]*)cc3)"
             "cc(C#CC#Cc3ccc([N]*)cc3)c2)cc1",
        linker="*[CH]c1cc([CH]*)cc([CH]*)c1",
        bond=1.40),
    # 卟啉 + 丁二炔撑苯 的四方（sql）COF：4 个 meso 位连接
    # 注意 1：卟啉自由碱只有 2 个 NH + 2 个 N。写成 4 个 [nH] 会变成四质子化的
    #         22π 反芳香体系，ETKDG 出来的构象严重翘曲（出平面 4.9 Å），
    #         压平后环内键会被压到 0.63 Å。这里用的是正确的 2 NH 形式。
    # 注意 2：meso 位直接连芳环（苯撑，a≈12.8 Å）在共面层里 β-H 与邻位 H
    #         会撞到 0.55 Å —— 真实体系里苯环只能扭出平面，2D 层装不下。
    #         所以 meso 位接 sp 碳（炔撑），没有邻位 H，a≈18.0 Å 才合理。
    "porphine_sql": dict(
        topology="sql",
        node="*c1c2ccc(n2)c(*)c2ccc([nH]2)c(*)c2ccc(n2)c(*)c2ccc1[nH]2",
        linker="*C#Cc1ccc(C#C*)cc1",
        bond=1.43),
}


# 不带任何参数直接运行时使用的默认组合（改这里即可换默认结构）
DEFAULT_PRESET = "triazine"


def main():
    ap = argparse.ArgumentParser(
        description="两个单体 SMILES → hcb COF 的周期 CIF",
        epilog="内置预设: " + ", ".join(PRESETS) +
               "\n例: python build_cof_smiles.py --preset triazine --out my_cof.cif",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="使用内置的单体组合（无需再输 --node/--linker/--bond）")
    ap.add_argument("--node", help="节点 SMILES。可以完全不写 *（脚本自动找 sp2 连接位），"
                                   "也可以自己用 * 指定；* 的个数 3 → hcb，4 → sql")
    ap.add_argument("--linker", help="第二个单体 SMILES：线形连接子（2 个 *），"
                                     "或第二种节点（与节点相同个数的 *，C3+C3 型）")
    ap.add_argument("--arms", type=int, default=None,
                    help="节点要几个连接臂（3=hcb，4=sql）。只有在节点没写 * 时才需要")
    ap.add_argument("--arms2", type=int, default=None,
                    help="第二个单体几个连接臂（连接子=2）。默认 2")
    ap.add_argument("--fix-sites", action="store_true",
                    help="忽略你自己写的 * 位置，按分子本身重新自动选连接位")
    ap.add_argument("--allow-sp3", action="store_true",
                    help="连接位是 CH2/CH3 也照样接：按平面三角摆，H 放在层内。"
                         "骨架可用，键角是强行压平的，后续自己到 MS 里补 H")
    ap.add_argument("--inspect", metavar="SMILES",
                    help="只体检：列出每个原子的杂化/氢数/能不能当连接位，并给出可用写法")
    ap.add_argument("--topology", choices=["auto", "hcb", "sql"], default="auto",
                    help="拓扑：hcb=蜂窝(3 连接节点)，sql=四方(4 连接节点)，auto=按节点 * 个数判断")
    ap.add_argument("--bond", type=float, help="新生成连接键的键长 (Å)，C=C 1.34 / C=N 1.28 / C-C 1.46")
    ap.add_argument("--c", type=float, default=3.60, help="层间距 (Å)")
    ap.add_argument("--scale-a", type=float, default=1.0, help="a/b 缩放系数")
    ap.add_argument("--tilt", type=float, default=None,
                    help="连接键相对节点连线的倾角 (度)。默认由连接原子几何自动推算，一般不用给")
    ap.add_argument("--no-condense", action="store_true",
                    help="不要自动把 -NH2 / -CHO 缩合成亚胺（默认会自动识别并缩合）")
    ap.add_argument("--out", default="cof_smiles.cif")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # 只体检：把原子的杂化/氢数/可连接性摊开给人看
    if args.inspect:
        inspect_molecule(args.inspect)
        return

    # 合并预设与命令行参数（命令行优先）
    if not (args.node or args.linker or args.preset):
        # 什么都没给 → 用默认预设，直接出结果
        args.preset = DEFAULT_PRESET
        print(f"[提示] 未指定结构，使用默认预设 '{DEFAULT_PRESET}'；"
              f"换结构用 --preset lzul 或 --node/--linker\n")

    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    node = args.node or preset.get("node")
    linker = args.linker or preset.get("linker")
    bond = args.bond if args.bond is not None else preset.get("bond", 1.34)

    if not (node and linker):
        ap.print_help()
        print("\n错误: 必须提供 --node 和 --linker，或使用 --preset。")
        print("\n示例 1（三嗪-二乙炔 COF，节点 + 线形连接子）:")
        print('  python build_cof_smiles.py --preset triazine')
        print("\n示例 2（C3+C3，两个三连接节点直接成键）:")
        print('  python build_cof_smiles.py --node "*[N]c1ccc(C#CC#Cc2cc(C#CC#Cc3ccc([N]*)cc3)'
              'cc(C#CC#Cc3ccc([N]*)cc3)c2)cc1" --linker "*[CH]c1cc([CH]*)cc([CH]*)c1" --bond 1.40')
        raise SystemExit(2)

    # ---------- 连接位：允许完全不写 *，由脚本自己找 ----------
    # 用户只要给出单体本身的 SMILES（如 "c1ncncn1"），这里补上 * 再往下走。
    def _resolve(smi, want, role):
        try:
            have = count_stars(smi)
        except ValueError as e:
            raise SystemExit(str(e))
        if have == 0:
            print(f"[连接位] {role}没有给 *，按分子本身自动找 {want} 个 sp2 位点")
        elif not args.fix_sites:
            # 写了 * 就尊重它，个数以 * 为准（--arms 只是没写 * 时的提示）
            if want and have != want:
                print(f"[提示] {role}写了 {have} 个 *，按 {have} 连接处理"
                      f"（要改选位用 --fix-sites）")
            for r in diagnose_stars(smi):
                if r["nh"] >= 2:
                    tail = ("按 --allow-sp3 强行压平，H 放层内" if args.allow_sp3
                            else "加 --fix-sites 可重选，加 --allow-sp3 可强行接")
                    print(f"[注意] {role}的 * 挂在原子 {r['idx']}（{r['sym']}，"
                          f"{r['hyb']}，{r['nh']} 个 H）上 —— 四面体碳，{tail}。")
            return smi
        else:
            print(f"[连接位] {role}原写了 {have} 个 *，--fix-sites 要求重新选")
        try:
            new, note = auto_star(smi, want, role, force=args.fix_sites)
        except ValueError as e:
            raise SystemExit(f"\n{role} 无法确定连接位：\n  {e}\n")
        return new

    node = _resolve(node, args.arms or 3, "节点")
    wants_linker = args.arms2 or 2
    if count_stars(linker) == 0:
        print(f"[提示] 第二个单体按 {wants_linker} 连接处理；"
              f"若它其实是第二个 {count_stars(node)} 连接节点（C3+C3 型），"
              f"加 --arms2 {count_stars(node)}")
    linker = _resolve(linker, wants_linker, "连接子/第二节点")

    # 拓扑：命令行 > 预设 > 按节点 * 个数自动判断
    topology = args.topology
    if topology == "auto":
        topology = preset.get("topology")
    if not topology:
        try:
            n_star = count_stars(node)
        except ValueError as e:
            raise SystemExit(str(e))
        topology = {3: "hcb", 4: "sql"}.get(n_star)
        if topology is None:
            raise SystemExit(f"节点 SMILES 里有 {n_star} 个连接位，"
                             f"目前只支持 3 连接（hcb）和 4 连接（sql）")
        print(f"[提示] 按节点连接数判断拓扑: {topology}")

    try:
        atoms, info = build(node, linker, topology=topology, bond=bond, c=args.c,
                            scale_a=args.scale_a, tilt=args.tilt,
                            condense=not args.no_condense,
                            allow_sp3=args.allow_sp3)
    except (ValueError, RuntimeError) as e:
        # 这类错误都是"输入的单体有问题"，直接给结论，不要甩一堆 traceback
        raise SystemExit(f"\n搭建失败：{e}\n")
    plot_atoms = atoms.copy()          # 未折回晶胞的副本，画图更完整
    atoms.wrap(center=(0.5, 0.5, 0.5)) # CIF 用折回坐标，并保持层居中
    dmin = report(atoms, info)
    write(args.out, atoms)
    write(args.out.replace(".cif", "_layer.xyz"), plot_atoms)
    print(f"已保存 CIF: {args.out}")

    if not args.no_plot:
        try:
            # 见 cof_builder.py 里的同一处注释：原模块已被移进 _trash_20261023/
            from cof_view import plot_layer
            plot_layer(plot_atoms, out=args.out.replace(".cif", "_layer.png"))
        except Exception as e:
            print(f"[跳过绘图] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
