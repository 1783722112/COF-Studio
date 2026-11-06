#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_cof_3d.py —— 三维 COF 自动搭建器（SMILES + 拓扑 + 连接方式 → CIF）
==========================================================================

给两个单体的 SMILES、一个拓扑名、一种连接方式，直接搭出三维 COF 的周期 CIF。

    python build_cof_3d.py                              # 交互向导
    python build_cof_3d.py --preset bf_cof_1            # 内置 BF-COF-1
    python build_cof_3d.py --preset cof300              # 内置 COF-300（TAM + BDA，dia）
    python build_cof_3d.py --preset taa_bda             # 内置金刚烷四胺 + 对苯二甲醛
    python build_cof_3d.py -n "<节点 SMILES>" -l "<连接子 SMILES>" \\
        --topology ctn --linkage imine -a 24.798 -o BF-COF-1

    -a auto    不给晶胞边长，按两个单体的几何反推，并扫一遍挑"连得上键且不重叠"的
               （单体一大就必须用它 —— 手填的 a 经常装不下）
    --list-topologies / --inspect xxx.cif               # 查表 / 体检

--------------------------------------------------------------------------
脚本在做什么
--------------------------------------------------------------------------
1. **拓扑**：只存"空间群 + 每个不等价位点的 Wyckoff 位置 + 连接数"，以及
   一个可调的几何参数（ctn 的 16c 位点参数 x）。晶胞里所有位点由对称操作
   展开（ctn 是 12 个 4-c + 16 个 3-c），位点之间的连接关系按最近邻算。
2. **分子**：RDKit 从 SMILES 生成 3D 构象，自动认出连接原子
   （醛基碳 / 氨基氮），并把缩合时要脱掉的原子（醛的 =O、胺的 2 个 H）
   删掉。
3. **放置**：把单体的"连接臂向量"用 Kabsch 刚体对齐到拓扑的邻居方向，
   臂和方向的对应关系穷举排列取残差最小的一组。分子中心放在拓扑位点上。
   **每个位点单独对齐**，不靠对称操作复制分子 —— 分子对称性（Td）通常
   高于位点对称性（12b 只有 C2），复制会让一部分分子翻到不相容的取向。
4. **缝合**：沿 C→N 方向把两侧分子各让一半，迭代到每条边的 C…N 距离
   都是 1.28 Å（只做刚体平移，不改单体内部几何）。偏大说明晶胞给大了
   或者单体臂太短，脚本会顺带给出建议的 a。
5. **展开**：写出的是 P1 全晶胞，不写空间群对称操作 —— 这样 VESTA / MS /
   OVITO 都能直接读，但原子数会比空间群描述多几十倍，属于正常现象。

--------------------------------------------------------------------------
重要的物理限制
--------------------------------------------------------------------------
拓扑只约束"节点中心在哪、谁和谁相连"，并不保证分子能严丝合缝地装进去。
如果单体几何与拓扑要求的臂角差别大，对齐残差就会大（脚本会打印），
结构会带内应力。**必须再做一轮弛豫**（推荐 MACE-MP-0 + D3，或 DFT-D），
才能拿去做性质计算。脚本末尾会给出具体的弛豫命令。

单体一大，几何余量就没了
------------------------
ctn 的 4-c 位点在 C2 轴上，四个邻居的夹角是 97.5° / 137.6°，不是理想的
109.5°；3-c 位点约 107°，也不是 120°。臂越长，这个角度差放大成"臂端错位"
就越厉害（错位 ≈ 2 × 臂长 × sin(Δθ/2)）。所以：

  · 小单体（BF-COF-1，臂长 3 Å）—— 错位 1.3 Å，缝合一步就补上了；
  · 中等单体（臂长 7 Å）—— 错位 3–4 Å，必须把晶胞放大才有余量，
    用 -a auto 让脚本自己扫；
  · 又长又刚的四面体节点（如四苯基甲烷，臂 5.8 Å）—— ctn 装不下，
    扫遍晶胞也找不到"既不重叠又连得上键"的尺寸，脚本会明确告诉你。

--------------------------------------------------------------------------
支持的拓扑
--------------------------------------------------------------------------
    ctn   I-43d (220)  4-c 在 12b，3-c 在 16c   —— C3N4 型，COF-105/108、
                                                       BF-COF-1/2 都是它

加新拓扑只要在 TOPOLOGIES 里加一条：空间群号、每个不等价位点的连接数和
分数坐标（可调分量写成字符串 "x"/"y"/"z"）、参数范围。摆放和缝合逻辑
完全不依赖具体拓扑，所以不需要改别的地方。

支持的连接方式
--------------------------------------------------------------------------
    imine       醛 + 胺 → C=N + H2O          （已实测）
    boronate    硼酸 + 邻二酚 → 硼酸酯 + 2H2O （条目在，未实测）
    ketoenamine 醛 + 胺 → β-酮烯胺（需互变异构，未实测）

作者备注
--------
* 所有距离都走最小镜像（MIC），立方/四方晶胞下准确。
* 输出是 P1 全晶胞，原子数会比空间群描述多几十倍，这是正常的。
* 两个单体谁当节点无所谓：脚本按官能团和连接数自动认。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import permutations

import numpy as np

warnings.filterwarnings("ignore")

try:
    from ase import Atoms
    from ase.io import write as ase_write
    from ase.spacegroup import Spacegroup
except ImportError:
    sys.exit("需要 ase：pip install ase")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    sys.exit("需要 rdkit：conda install -c conda-forge rdkit")


# ============================================================ 拓扑库
# frac 里用字符串 "x"/"y"/"z" 表示可调参数；vary 指定哪些维度一起动。
TOPOLOGIES = {
    "ctn": {
        "name": "ctn",
        "desc": "C3N4 型 3,4-连接网（cubic carbon nitride）",
        "spacegroup": 220,
        "symbol": "I-43d",
        # (连接数, 分数坐标模板, 位点标签, 是否参与参数化)
        "sites": [
            {"coord": 4, "frac": [0.0, 0.25, 0.375], "label": "12b"},
            {"coord": 3, "frac": ["x", "x", "x"], "label": "16c"},
        ],
        "param": {"name": "x", "lo": 0.020, "hi": 0.1235, "default": 0.110},
        "note": "COF-105 / COF-108 / BF-COF-1 / DL-COF 系列的标准拓扑",
    },
    "dia": {
        "name": "dia",
        "desc": "金刚石型 4,2-连接网（diamondoid）",
        "spacegroup": 227,
        "symbol": "Fd-3m",
        # 4-c 节点占金刚石的一个子格（8a），2-c 直线连接子落在键中点上（16c）。
        # 8×4 = 16×2 = 32 条边，自洽。
        # 这个拓扑没有自由参数：立方对称把角度全定死了，只有 a 可调。
        "sites": [
            {"coord": 4, "frac": [0.0, 0.0, 0.0], "label": "8a"},
            {"coord": 2, "frac": [0.125, 0.125, 0.125], "label": "16c"},
        ],
        "param": None,
        "note": "COF-300 / LZU-111 型；Td 节点 + 联苯、联吡啶这类直线连接子",
    },
}

# ============================================================ 连接方式
LINKAGES = {
    "imine": {
        "name": "亚胺（Schiff 碱）",
        "equation": "Ar-CHO + H2N-Ar' -> Ar-CH=N-Ar' + H2O",
        "donor": "aldehyde",       # 提供 C 的一端
        "acceptor": "amine",       # 提供 N 的一端
        "bond": 1.28,              # C=N 典型键长
        "bond_name": "C=N",
    },
    "boronate": {
        "name": "硼酸酯",
        "equation": "Ar-B(OH)2 + 2 HO-Ar' -> Ar-B(O-Ar')2 + 2 H2O",
        "donor": "boronic",
        "acceptor": "catechol",
        "bond": 1.48,
        "bond_name": "B-O",
    },
    "ketoenamine": {
        "name": "β-酮烯胺",
        "equation": "Ar-CO-CH3 + H2N-Ar' -> Ar-C(=CH)-NH-Ar' + H2O",
        "donor": "ketone",
        # Tp（2,4,6-三羟基-1,3,5-三甲醛）这类走的是另一条：醛基先与胺缩合成
        # 亚胺，邻位羟基随即把质子交出来、烯醇化锁定成酮烯胺。连接原子还是
        # 醛基碳（去 O 留 C），和甲基酮完全同构，所以直接复用 aldehyde 的
        # 连接点与离去规则，只是键长按 C-N 单键 1.34 Å 摆。
        # 没有这一条时 Tp 会被判"找不到 ketone"，三维 β-酮烯胺 COF 全部建不了。
        "donor_alt": ["aldehyde"],
        "acceptor": "amine",
        "bond": 1.34,
        "bond_name": "C-N",
        "caution": (
            "β-酮烯胺的甲基酮会先烯醇化成 =CH2（酮碳缩合后只剩 Ar、N 两根键，\n"
            "          CH3 留着就是三配位碳），但直线边模型会把 N 摆到与 Ar 反向的\n"
            "          180° 位置，=CH2 与 N 只能挤到 ~1.15 Å。这一支要弛豫后才能用；\n"
            "          需要干净结构请优先用 imine。"),
    },
}

# SMARTS：用来在单体里自动找连接原子
CONNECT_SMARTS = {
    "aldehyde": "[CX3H1](=O)[#6]",        # 芳环上的醛基碳
    "amine": "[NX3;H2][CX4,c]",           # 伯胺氮（sp3 或芳香碳上）
    "boronic": "[BX3](O)O[c]",            # 硼酸
    "catechol": "[OX2H]c",                # 酚羟基氧（每条边用掉一个）
    "ketone": "[CX3](=O)[CH3]",           # 甲基酮
}

# 缩合时要删掉的原子（相对连接原子）
LEAVING_RULES = {
    "aldehyde": {"drop": ["O"], "keep": "C"},
    "amine": {"drop": ["H", "H"], "keep": "N"},
    "boronic": {"drop": ["O", "O"], "keep": "B"},
    "catechol": {"drop": ["H"], "keep": "O"},
    "ketone": {"drop": ["O"], "keep": "C"},
}


# ============================================================ 小工具
def banner(msg, ch="="):
    print("\n" + ch * 74)
    print(msg)
    print(ch * 74)


def mic_vec(frac_a, frac_b, cell):
    """分数坐标差 → 最小镜像的笛卡尔向量。frac_a - frac_b 方向。"""
    d = np.asarray(frac_a, float) - np.asarray(frac_b, float)
    d -= np.round(d)
    return d @ np.asarray(cell, float)


def mic_cart(a, b, cell):
    """从 b 指向 a 的最小镜像向量（笛卡尔）。"""
    inv = np.linalg.inv(cell)
    d = np.asarray(a, float) - np.asarray(b, float)
    f = d @ inv
    f -= np.round(f)
    return f @ cell


def mic_dist(frac_a, frac_b, cell):
    return float(np.linalg.norm(mic_vec(frac_a, frac_b, cell)))


def kabsch(A, B):
    """求旋转 R，使 A @ R ≈ B（A、B 都是 n×3 的行向量）。

    标准 Kabsch：H = A^T B = U S V^T，最优 R = U V^T（det 修正）。
    注意是 U @ Vt，不是它的转置 —— 写成 Vt.T @ U.T 会得到逆旋转，
    RMS 会凭空大一倍，而且看起来"只是拟合得差"，很难察觉。
    """
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def aligned_rotation(arms, targets):
    """穷举臂与目标的对应关系，返回 (R, rms_deg, perm)。

    arms / targets 都是 n×3 的单位向量。
    """
    arms = np.asarray(arms, float)
    targets = np.asarray(targets, float)
    n = len(arms)
    best = None
    for perm in permutations(range(n)):
        B = targets[list(perm)]
        R = kabsch(arms, B)
        pred = arms @ R
        cos = np.clip((pred * B).sum(1), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        rms = float(np.sqrt((ang ** 2).mean()))
        if best is None or rms < best[1]:
            best = (R, rms, perm)
    return best


# ============================================================ 对称操作
CENTERING = {
    "P": [[0, 0, 0]],
    "I": [[0, 0, 0], [0.5, 0.5, 0.5]],
    "F": [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    "A": [[0, 0, 0], [0, 0.5, 0.5]],
    "B": [[0, 0, 0], [0.5, 0, 0.5]],
    "C": [[0, 0, 0], [0.5, 0.5, 0]],
    "R": [[0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]],
}


class Symmetry:
    """空间群操作 + 离心平移。

    ASE 的 Spacegroup.rotations/translations 只给**不带离心**的点操作
    （220 给 24 个、227 给 48 个），面心/体心的平移得自己补：
    220 要 ×2 才有 48 个，227 要 ×4 才有 192 个。少了这一步，
    展开出来的原子数是错的，而且看不出来（只是"位点少了一半"）。
    """

    def __init__(self, number):
        sg = Spacegroup(number)
        R = np.array(sg.rotations, float)
        T = np.array(sg.translations, float)
        letter = sg.symbol.strip()[0].upper()
        cen = np.array(CENTERING.get(letter, [[0, 0, 0]]), float)
        self.R = np.vstack([R] * len(cen))
        self.T = np.vstack([T + c for c in cen])
        self.symbol = sg.symbol
        self.number = number
        self.centering = letter

    def orbit(self, frac, tol=6):
        q = np.einsum("nij,j->ni", self.R, np.asarray(frac, float)) + self.T
        q = q % 1.0
        return np.unique(np.round(q, tol), axis=0)

    def apply(self, frac):
        """返回该点所有对称等效位置（保留重复，用于展开分子）。"""
        q = np.einsum("nij,j->ni", self.R, np.asarray(frac, float)) + self.T
        return q % 1.0


# ============================================================ 单体处理
class Monomer:
    """一个已经"准备好连接"的单体：坐标、元素、连接原子索引。"""

    def __init__(self, smi, kind, n_conn_expected=None, seed=0xF00D, label=""):
        self.smi = smi
        self.kind = kind
        self.label = label or kind
        self._build(seed)
        if n_conn_expected and len(self.conn) != n_conn_expected:
            raise RuntimeError(
                f"{self.label}: 期望 {n_conn_expected} 个连接点，"
                f"但 SMILES 里找到 {len(self.conn)} 个（{kind}）"
            )

    def _build(self, seed):
        mol = Chem.MolFromSmiles(self.smi)
        if mol is None:
            raise RuntimeError(f"SMILES 解析失败：{self.smi}")
        mol = Chem.AddHs(mol)

        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, ps) != 0:
            raise RuntimeError(f"3D 构象生成失败：{self.smi}")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=5000)

        pos = np.array(mol.GetConformer().GetPositions(), float)
        sym = [a.GetSymbol() for a in mol.GetAtoms()]

        patt = Chem.MolFromSmarts(CONNECT_SMARTS[self.kind])
        hits = mol.GetSubstructMatches(patt)
        if not hits:
            raise RuntimeError(
                f"{self.label}: 在 SMILES 里找不到 {self.kind} 连接点\n"
                f"  用的 SMARTS 是 {CONNECT_SMARTS[self.kind]}\n"
                f"  检查一下单体里的官能团写法"
            )
        # 只取每个匹配的第一个原子（醛的 C、胺的 N…）
        conn = sorted({h[0] for h in hits})

        rule = LEAVING_RULES[self.kind]
        drop = []
        for ci in conn:
            neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(ci).GetNeighbors()]
            pool = [j for j in neighbours if sym[j] in rule["drop"]]
            need = rule["drop"].count(sym[pool[0]]) if pool else 0
            # 按需删：醛删 1 个 O；胺删 2 个 H；硼酸删 2 个 O
            want = {
                ("aldehyde"): {"O": 1},
                ("amine"): {"H": 2},
                ("boronic"): {"O": 2},
                ("catechol"): {"H": 1},
                ("ketone"): {"O": 1},
            }[self.kind]
            for el, cnt in want.items():
                cand = [j for j in neighbours if sym[j] == el]
                drop.extend(cand[:cnt])

        # 甲基酮要先烯醇化再缩合（必须在算 keep/positions 之前改坐标）
        if self.kind == "ketone":
            self._enolize(mol, conn, pos, sym, drop)

        drop = sorted(set(drop))
        keep = [i for i in range(len(sym)) if i not in drop]

        self.elements = [sym[i] for i in keep]
        self.positions = pos[keep]
        remap = {old: new for new, old in enumerate(keep)}
        self.conn = [remap[c] for c in conn]
        self.conn_elements = [self.elements[c] for c in self.conn]
        self.center = self.positions[self.conn].mean(0)
        self.arms = self.positions[self.conn] - self.center
        self.arm_len = np.linalg.norm(self.arms, axis=1)
        self.arm_dirs = self.arms / self.arm_len[:, None]
        self.n_conn = len(self.conn)

    def _enolize(self, mol, conn, pos, sym, drop):
        """甲基酮 → 烯醇式：Ar-C(=O)-CH3 变成 Ar-C(=CH2)-。

        缩合之后酮碳只剩 Ar 和 N 两根键，甲基如果还是 sp3 的 CH3
        （1.50 Å、109°），酮碳就只有三根单键 —— 少一根键的碳不是化学结构。
        正确的产物是 β-酮烯胺的烯醇式：Ar-C(=CH2)-NH-Ar'，
        所以这里把甲基碳挪到 1.34 Å 的 C=C 上、两个 H 摆回
        Ar-C-N 平面内的 120° 位置，第三个 H 计入消去名单。

        酮碳是 sp2：Ar、N、=CH2 三者互成 120° 且共面。
        取 u = C→Ar、o = C→O（将来 O 的位置就是 N），
        第三根键方向就是 -(u+o) 的角平分线。
        """
        for c1 in conn:
            nb = [n.GetIdx() for n in mol.GetAtomWithIdx(c1).GetNeighbors()]
            oxy = [j for j in nb if sym[j] == "O"]
            cs = [j for j in nb if sym[j] == "C"]
            # 甲基碳带 3 个 H，芳环碳只带 0–1 个 —— 只能靠 H 数区分，
            # 因为两者都不在 conn 里（conn 只放酮羰基碳）
            methyl = [j for j in cs
                      if sum(1 for k in mol.GetAtomWithIdx(j).GetNeighbors()
                             if k.GetSymbol() == "H") == 3]
            aryl = [j for j in cs if j not in methyl]
            if not oxy or not methyl or not aryl:
                continue
            c2 = methyl[0]

            u = pos[aryl[0]] - pos[c1]
            u /= np.linalg.norm(u)
            o = pos[oxy[0]] - pos[c1]
            o /= np.linalg.norm(o)
            w = -(u + o)                     # C1→CH2 方向，与 u、o 各 120°
            w /= np.linalg.norm(w)
            in_plane = u - (u @ w) * w       # 面内、垂直于 w
            in_plane /= np.linalg.norm(in_plane)

            pos[c2] = pos[c1] + 1.34 * w
            hs = [n.GetIdx() for n in mol.GetAtomWithIdx(c2).GetNeighbors()
                  if n.GetSymbol() == "H"]
            drop.extend(hs[2:])              # 消去一个 H，C 上留两个
            for k, h in enumerate(hs[:2]):
                d = -0.5 * w + (1 if k == 0 else -1) * (np.sqrt(3) / 2) * in_plane
                pos[h] = pos[c2] + 1.08 * d / np.linalg.norm(d)

    @property
    def formula(self):
        from collections import Counter
        c = Counter(self.elements)
        order = ["C", "N", "O", "B", "S", "H"]
        return "".join(f"{el}{c[el] if c[el] > 1 else ''}"
                       for el in order if el in c)

    def arm_angles(self):
        n = self.n_conn
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                c = np.clip(self.arm_dirs[i] @ self.arm_dirs[j], -1, 1)
                out.append(np.degrees(np.arccos(c)))
        return sorted(out)


# ============================================================ 拓扑几何
class Topology:
    def __init__(self, key):
        if key not in TOPOLOGIES:
            raise KeyError(f"未知拓扑 {key}，支持：{', '.join(TOPOLOGIES)}")
        self.key = key
        self.defn = TOPOLOGIES[key]
        self.sym = Symmetry(self.defn["spacegroup"])
        self.sites = self.defn["sites"]
        self.param = self.defn.get("param")

    # ---------- 位点展开 ----------
    def site_frac(self, site, x):
        out = []
        for v in site["frac"]:
            if isinstance(v, str):
                out.append(x)
            else:
                out.append(float(v))
        return np.array(out, float)

    def all_sites(self, x):
        """返回 [(coord, label, frac_array), ...]，已去重。"""
        out = []
        for s in self.sites:
            f = self.site_frac(s, x)
            orb = self.sym.orbit(f)
            for p in orb:
                out.append((s["coord"], s["label"], p))
        return out

    # ---------- 边 ----------
    def coord_classes(self, x):
        """把位点按连接数分组，返回 {coord: [frac,...]}（保持位点定义顺序）。"""
        out = {}
        for c, _l, p in self.all_sites(x):
            out.setdefault(c, []).append(p)
        return out

    def edge_length(self, cell, x):
        """两类位点之间的间距（第一类位点到最近的另一类）。

        ctn 是 4-c↔3-c，dia 是 4-c↔2-c，都按同一个定义取，不写死 4/3。
        """
        cls = self.coord_classes(x)
        keys = sorted(cls, reverse=True)
        if len(keys) < 2:
            return None
        A, B = cls[keys[0]], cls[keys[1]]
        best = [min(mic_dist(p, q, cell) for q in B)
                for p in A[:4]]
        return float(np.mean(best))

    def solve_param(self, cell, target):
        """解 x 使边跨度 = target。返回 (x, 实际跨度, 是否精确)。

        没有自由参数的拓扑（dia）直接返回 x=0：位点角度由立方对称锁死，
        跨度只跟 a 成正比，所以 a ← a·target/d 一次就收敛。
        """
        if self.param is None:
            d = self.edge_length(cell, 0.0)
            return 0.0, d, bool(d and abs(d - target) < 1e-3)
        lo, hi = self.param["lo"], self.param["hi"]
        xs = np.linspace(lo, hi, 400)
        ds = [self.edge_length(cell, float(x)) for x in xs]
        dmin = min(ds)
        if target < dmin - 1e-3:
            # 无解：晶胞太大
            xi = float(xs[int(np.argmin(ds))])
            return xi, dmin, False
        # 在单调段上二分（x 从 hi 往 lo，距离先减后增；我们取 hi 侧）
        i_best = None
        best_err = 1e9
        for i, d in enumerate(ds):
            e = abs(d - target)
            if e < best_err:
                best_err, i_best = e, i
        return float(xs[i_best]), float(ds[i_best]), best_err < 1e-3

    def best_param_for_shape(self, cell, m_acc, m_donor, n_scan=48):
        """按"臂角残差最小"挑参数 x。

        为什么不按"边跨度匹配"挑：跨度随 x 是 U 形的，同一跨度往往有两个解，
        靠近 lo 的那个几何极度扭曲（3-c 的邻居挤成一条线）。这里直接用分子
        的真实臂向量去打分，选几何最顺的那个。
        """
        if self.param is None:
            return 0.0, 0.0, 0.0
        lo, hi = self.param["lo"], self.param["hi"]
        best = None
        for x in np.linspace(lo, hi, n_scan):
            sites = self.all_sites(float(x))
            s4 = [p for c, _l, p in sites if c == 4]
            s3 = [p for c, _l, p in sites if c == 3]
            if not s4 or not s3:
                continue
            n4 = self.neighbours(s4[0], cell, m_acc.n_conn, sites_frac=sites)
            n3 = self.neighbours(s3[0], cell, m_donor.n_conn, sites_frac=sites)
            _, r4, _ = aligned_rotation(
                m_acc.arm_dirs, np.array([u for _d, u, _p in n4]))
            _, r3, _ = aligned_rotation(
                m_donor.arm_dirs, np.array([u for _d, u, _p in n3]))
            score = r4 + r3
            if best is None or score < best[0]:
                best = (score, float(x), r4, r3)
        _s, x, r4, r3 = best
        return x, r4, r3

    def neighbours(self, frac, cell, n, same_class=False, sites_frac=None):
        """找最近的 n 个（异类）节点，返回 [(dist, unit_vec, frac_of_neighbour)]。"""
        sites = sites_frac if sites_frac is not None else self.all_sites(0.0)
        my_coord = None
        # 判断自己的配位数：找距离为 0 的那个
        for c, _l, p in sites:
            if mic_dist(frac, p, cell) < 1e-6:
                my_coord = c
                break
        pool = []
        for c, _l, p in sites:
            if my_coord is not None and not same_class and c == my_coord:
                continue
            if my_coord is not None and same_class and c != my_coord:
                continue
            v = mic_vec(p, frac, cell)
            d = np.linalg.norm(v)
            if d > 1e-6:
                pool.append((d, v / d, p))
        pool.sort(key=lambda t: t[0])
        return pool[:n]


# ============================================================ 组装
def linkage_span(linkage: dict, node: Monomer, linker: Monomer):
    """按连接方式判定谁出 C、谁出 N，返回 (边跨度, 给 C 的单体, 接 N 的单体)。

    边跨度 = 出 C 那侧的臂长 + 目标键长 + 接 N 那侧的臂长。这个数就是
    拓扑必须提供的"邻居位点间距"，是后面定拓扑参数 x 和反推晶胞 a 的依据。

    谁出 C 看的是单体实际的连接基团（Monomer.kind），不是"node 还是 linker"
    这个角色 —— 否则把两个 SMILES 的位置写反就会搭出错误的缩合方向。
    """
    donor, acceptor = linkage["donor"], linkage["acceptor"]
    donors = [donor, *linkage.get("donor_alt", ())]
    if node.kind in donors and linker.kind == acceptor:
        m_donor, m_acc = node, linker
    elif node.kind == acceptor and linker.kind in donors:
        m_donor, m_acc = linker, node
    else:
        raise RuntimeError(
            f"连接方式 {linkage['name']} 需要 {' / '.join(donors)} + {acceptor}，"
            f"但你给的是 {node.kind} + {linker.kind}")
    span = m_donor.arm_len.mean() + m_acc.arm_len.mean() + linkage["bond"]
    return span, m_donor, m_acc


def estimate_cell(node: Monomer, linker: Monomer, topo: Topology, linkage: dict,
                  a0=25.0, iters=12, verbose=False):
    """只靠两个单体的几何反推晶胞边长，不需要文献值。

    思路：拓扑位点参数 x 的取值范围是固定的，能提供的最短边跨度有下限。
    先在 a0 上解一次 x；如果解不出目标跨度（x 撞到边界），说明晶胞偏大，
    按 span/d_edge 等比缩小 a 再解。通常两轮就收敛。

    返回 (a, x, d_edge, exact)。exact=True 表示边跨度和分子跨度完全对上，
    这时理论上不需要缝合，C=N 一放就是目标键长。
    """
    span, _md, _ma = linkage_span(linkage, node, linker)
    a, x, d, exact = a0, 0.0, 0.0, False
    for _ in range(iters):
        cell = np.diag([a, a, a])
        x, d, exact = topo.solve_param(cell, span)
        if verbose:
            print(f"    a={a:7.3f}  x={x:.5f}  间距={d:7.3f}  "
                  f"目标={span:7.3f}  {'对上了' if exact else '继续缩放'}")
        if exact or d <= 0:
            break
        a_new = a * span / d
        if abs(a_new - a) < 1e-3:
            a = a_new
            break
        a = a_new
    return a, x, d, exact


def _score_cell(node, linker, topo, linkage, a):
    """在给定 a 上跑一遍"摆放 + 缝合"，回报 (键长最大偏差, 最近间距, built)。"""
    b = build(node, linker, topo, linkage, a, verbose=False)
    groups = b["groups"]
    stitch(groups, b["m_acc"], b["m_donor"], b["cell"], linkage["bond"],
           verbose=False)
    edges = edge_table(groups, b["m_acc"], b["m_donor"], b["cell"])
    errs = []
    for i, k, j, l in edges:
        ci = groups[i]["mono"].conn[k]
        cj = groups[j]["mono"].conn[l]
        d = float(np.linalg.norm(
            mic_cart(groups[i]["cart"][ci], groups[j]["cart"][cj], b["cell"])))
        errs.append(abs(d - linkage["bond"]))
    return (max(errs) if errs else 99.0), min_pair_distance(groups, b["cell"]), b


def fit_cell(node: Monomer, linker: Monomer, topo: Topology, linkage: dict,
             clearance=1.0, bond_tol=0.06, max_scale=1.8, step=1.04,
             verbose=True):
    """在"能连上键"的基础上再挑一个**装得下分子**的晶胞。

    为什么光有 estimate_cell 不够：那个函数只保证"边跨度 = 分子跨度"，
    也就是拓扑给的邻居间距刚好等于两根臂加上键长。单体一变大，臂一长，
    这个"刚好"就变成了零余量 —— 分子被塞得互相穿插，缝合要么被限死
    （键长还是 5 Å），要么把原子压到 0.5 Å。

    所以这里沿"边跨度匹配"这条曲线往上扫 a，对每个候选做一次真实的
    摆放 + 缝合，取第一个同时满足「键长偏差 ≤ bond_tol」+「最近间距 ≥
    clearance」的。扫描上限 max_scale×a0，找不到就取评分最好的那个。

    返回 (a, 键长最大偏差, 最近间距, ok)。ok=False 说明扫遍了也找不到
    既能连上键、又不重叠的晶胞 —— 这多半是单体臂角和位点夹角差太多，
    不是调参能解决的，run() 会据此给出更具体的结论。
    """
    a0 = estimate_cell(node, linker, topo, linkage)[0]
    best = None
    a = a0
    n = 0
    while a <= a0 * max_scale + 1e-9 and n < 40:
        berr, dmin, _b = _score_cell(node, linker, topo, linkage, a)
        ok = (dmin >= clearance) and (berr <= bond_tol)
        if verbose:
            print(f"    a={a:7.3f}  键长偏差 {berr:5.3f}  最近间距 {dmin:5.3f}  "
                  f"{'✓ 可用' if ok else ''}")
        # 评分：先看是否达标，再看间距余量，键长偏差要罚
        score = (ok, dmin - 4.0 * berr)
        if best is None or score > best[0]:
            best = (score, a, berr, dmin, ok)
        if ok:
            return a, berr, dmin, True
        a *= step
        n += 1
    return best[1], best[2], best[3], best[4]


def build(node: Monomer, linker: Monomer, topo: Topology, linkage: dict,
          a: float, fit_param=True, verbose=True):
    """搭出不对称单元的分子坐标（分数坐标）。"""

    cell = np.diag([a, a, a])

    # ---------- 连接方式决定谁提供 C、谁提供 N ----------
    span, m_donor, m_acc = linkage_span(linkage, node, linker)

    # ---------- 求解拓扑参数 ----------
    # 用"边跨度 = 分子跨度"来定 x：这个解物理上唯一，而且不会因为 RDKit
    # 生成构象时连接原子的排序微差而在两个分支之间乱跳。
    # （按臂角打分虽然理论上更贴切，但 x 的评分曲线很平，argmin 不稳。）
    if fit_param:
        x, d_edge, exact = topo.solve_param(cell, span)
    else:
        x = topo.param["default"] if topo.param else 0.0
        d_edge = topo.edge_length(cell, x)
        exact = abs(d_edge - span) < 0.15

    if verbose:
        banner(f"拓扑 {topo.key}（{topo.defn['symbol']}）几何")
        print(f"  单体需要的节点间距 : {span:8.3f} Å"
              f"  (= {m_donor.arm_len.mean():.3f} + {linkage['bond']} + "
              f"{m_acc.arm_len.mean():.3f})")
        print(f"  晶胞 a             : {a:8.3f} Å")
        if topo.param:
            print(f"  位点参数 {topo.param['name']}         : {x:8.4f}"
                  f"  → 实际间距 {d_edge:8.3f} Å")
        if not exact:
            print(f"\n  [注意] 几何对不上：实际 {d_edge:.3f} Å vs 需要 {span:.3f} Å，"
                  f"差 {d_edge - span:+.3f} Å")
            if d_edge > span:
                print(f"         意思是这个晶胞偏大，或者单体臂偏短。")
                print(f"         把 a 改成 {a * span / d_edge:.2f} Å 左右就正合适。")

    sites = topo.all_sites(x)
    cls = {}
    for c, _l, p in sites:
        cls.setdefault(c, []).append(p)

    # 哪个分子进哪一类位点，看它实际的连接数 —— 不看它是"节点"还是"连接子"，
    # 也不看它是给 C 还是接 N。四醛基节点 + 三胺连接子这类体系都靠这条。
    want = sorted(cls, reverse=True)                 # 比如 [4, 3] 或 [4, 2]
    got = sorted({m_acc.n_conn, m_donor.n_conn}, reverse=True)
    if got != want:
        raise RuntimeError(
            f"拓扑 {topo.key} 要 "
            f"{' + '.join(f'{c}-c' for c in want)} 两个单体，"
            f"你给的是 {' + '.join(f'{c}-c' for c in got)}")

    pair = {}
    for c in want:
        # 该位点类要哪个分子：连接数对上就行；两个都对上时优先 m_acc
        who = [m for m in (m_acc, m_donor) if m.n_conn == c]
        pair[c] = who[0]
    m_hi, m_lo = pair[want[0]], pair[want[1]]

    if verbose:
        desc = " + ".join(f"{len(cls[c])} 个 {c}-c ({pair[c].label})" for c in want)
        print(f"  位点展开           : {desc}  (每晶胞)")

    # ---------- 逐节点定向摆放 ----------
    # 为什么不用"摆一个再用空间群复制"：分子对称性（Td）通常高于位点对称性
    # （ctn 的 12b 只有 C2），靠对称操作复制会让一部分分子翻到不相容的取向上。
    # 每个节点都按它自己的邻居方向单独对齐，最稳。
    groups = []

    def place(mono, frac):
        nbr = topo.neighbours(frac, cell, mono.n_conn, sites_frac=sites)
        if len(nbr) < mono.n_conn:
            raise RuntimeError(f"{mono.label}: 位点 {np.round(frac,3)} "
                               f"只找到 {len(nbr)} 个邻居")
        dirs = np.array([u for _d, u, _p in nbr])
        R, rms, perm = aligned_rotation(mono.arm_dirs, dirs)
        cart = (mono.positions - mono.center) @ R + frac @ cell
        return {"mono": mono, "cart": cart, "frac": frac, "rms": rms,
                "perm": perm, "dirs": dirs, "nbr": nbr}

    for c in want:
        for f in cls[c]:
            groups.append(place(pair[c], f))

    if verbose:
        rms4 = np.array([g["rms"] for g in groups if g["mono"] is m_hi])
        rms3 = np.array([g["rms"] for g in groups if g["mono"] is m_lo])
        banner("刚体对齐质量")
        print(f"  {m_hi.label:12s} {len(rms4):3d} 个分子  "
              f"臂角残差 平均 {rms4.mean():5.2f}°  最大 {rms4.max():5.2f}°")
        print(f"  {m_lo.label:12s} {len(rms3):3d} 个分子  "
              f"臂角残差 平均 {rms3.mean():5.2f}°  最大 {rms3.max():5.2f}°")
        print(f"  参考：< 5° 很理想，5–15° 可接受，> 20° 说明分子和拓扑不太匹配")
        worst = max(rms4.max(), rms3.max())
        if worst > 20:
            print(f"  [注意] 最大残差 {worst:.1f}°，是拓扑位点的固有夹角与单体臂角不匹配，"
                  f"不是 bug；")
            print(f"         结构会带内应力，靠后面的弛豫吸收。")

    return {
        "cell": cell, "x": x, "d_edge": d_edge, "span": span, "exact": exact,
        "groups": groups, "m_acc": m_acc, "m_donor": m_donor,
        "m_hi": m_hi, "m_lo": m_lo,
    }


def min_pair_distance(groups, cell):
    """所有原子（含分子间、跨晶胞镜像）里的最短距离。"""
    from scipy.spatial import cKDTree
    P = np.vstack([g["cart"] for g in groups])
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for k in (-1, 0, 1)], float) @ cell
    rep = (P[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tree = cKDTree(rep)
    dd, _ = tree.query(P, k=2)
    return float(dd[:, 1].min())


# ============================================================ 展开
# ============================================================ 缝合
def unwrap(groups, edges, cell):
    """把每个连接子平移到它三个节点邻居的中心附近，消除跨边界的镜像错位。

    不用 BFS 逐个展开：晶体里有环，BFS 沿不同路径累加会得到互相矛盾的结果
    （实测会把分子推到 30 Å 外）。这里直接按边表算，每个连接子取它所有
    邻居的相对位移平均值，一步到位、不会累积。
    """
    inv = np.linalg.inv(cell)
    n = len(groups)
    ctr = [g["cart"][g["mono"].conn].mean(0) for g in groups]

    for j in range(n):
        ds = []
        for (i, k, j2, l) in edges:
            if j2 != j:
                continue
            d = ctr[i] - ctr[j]
            f = d @ inv
            f -= np.round(f)
            ds.append(f @ cell)
        if ds:
            shift = np.mean(ds, axis=0)
            groups[j]["cart"] = groups[j]["cart"] + shift
            ctr[j] = ctr[j] + shift


def edge_table(groups, m_acc, m_donor, cell):
    """建立"哪一对 C/N 应该连成一根键"的边表。

    配对**不能靠距离猜**。同一个 4-c 的 4 个 N 在空间上是四面体分布，
    全晶胞找最近邻很容易有两个 N 抢到同一个 3-c 分子上（它有三根臂，
    总有一根离某个 N 最近），于是出现 7 Å 的假边。

    正确做法：对齐时 `aligned_rotation` 已经算出了"第 k 个臂对应哪一个
    邻居位点"（perm），这里直接照着这张拓扑表连，一根都不会错。
    """
    don_g = [(j, g) for j, g in enumerate(groups) if g["mono"] is m_donor]
    edges = []
    for i, g in enumerate(groups):
        if g["mono"] is not m_acc:
            continue
        nbr, perm = g.get("nbr"), g.get("perm")
        if nbr is None or perm is None:
            continue
        for k in range(g["mono"].n_conn):
            frac_j = nbr[perm[k]][2]
            best = None
            for j, gj in don_g:
                d = frac_j - gj["frac"]
                d -= np.round(d)
                dd = float(np.linalg.norm(d))
                if best is None or dd < best[0]:
                    best = (dd, j)
            j = best[1]
            gj = groups[j]
            ci = g["cart"][g["mono"].conn[k]]
            cc = gj["cart"][gj["mono"].conn]
            # 挑"离这个 N 最近的那个醛基 C"，同样要走最小镜像
            dd = np.array([np.linalg.norm(mic_cart(c, ci, cell)) for c in cc])
            l = int(np.argmin(dd))
            edges.append((i, k, j, l))
    return edges


def stitch(groups, m_acc, m_donor, cell, target, iters=600, relax=0.5,
           max_step=0.8, verbose=True, rotate=False, rotate_sweeps=8):
    """把每对"醛基 C / 胺基 N"拉到目标键长。

    为什么要单独做这一步：拓扑只规定节点中心，而单体的臂角（Td 的 109.5°、
    平面的 120°）和拓扑要求的邻居夹角（ctn 的 4-c 是 97.5°/137.6°，3-c 是
    ~107°）不可能完全一致。刚体对齐之后 C 和 N 会错开，散得很开。

    这里用最朴素的 Jacobi 松弛：每条边要求两端各让一半，把位移按每条分子
    参与边数取平均，迭代几百轮。分子只平移不转动，所以不会破坏单体内部几何。
    """
    acc = [i for i, g in enumerate(groups) if g["mono"] is m_acc]
    don = [i for i, g in enumerate(groups) if g["mono"] is m_donor]
    if not acc or not don:
        return np.zeros((len(groups), 3))

    edges = edge_table(groups, m_acc, m_donor, cell)

    shift = np.zeros((len(groups), 3))
    n_g = len(groups)
    for _it in range(iters):
        want = np.zeros_like(shift)
        cnt = np.zeros(n_g)
        for i, k, j, l in edges:
            ci = groups[i]["mono"].conn[k]
            cj = groups[j]["mono"].conn[l]
            Ni = groups[i]["cart"][ci] + shift[i]
            Cj = groups[j]["cart"][cj] + shift[j]
            # 必须走最小镜像：相邻分子经常被晶胞边界分开
            v = mic_cart(Cj, Ni, cell)
            d = float(np.linalg.norm(v))
            if d < 1e-6:
                continue
            u = v / d
            err = d - target
            want[i] += 0.5 * err * u
            want[j] -= 0.5 * err * u
            cnt[i] += 1
            cnt[j] += 1
        for g in range(n_g):
            if cnt[g] > 0:
                step = relax * want[g] / cnt[g]
                # 限制的是**单步**位移，不是累计位移。
                # 老写法把累计位移硬卡在 max_shift（1.5 Å），小单体够用，
                # 大单体需要挪 4–5 Å 就永远合不拢 —— 表现为"键连不上"，
                # 而脚本只会报一个 5 Å 的键长，看不出是位移被限死了。
                ns = float(np.linalg.norm(step))
                if ns > max_step:
                    step *= max_step / ns
                shift[g] += step

    for g in range(n_g):
        groups[g]["cart"] = groups[g]["cart"] + shift[g]
        groups[g]["shift"] = shift[g]

    # ---- 再用"旋转 + 平移"逐分子磨一遍 ----
    # 只平移是不够的：平面三角形（TFB）与三脚架位的夹角差 21°，臂末端会
    # 系统性错开。给每个分子 6 个自由度（3 转 + 3 移），固定邻居后轮流优化，
    # 几轮就能把大部分错位吃掉。
    if rotate:
        from scipy.optimize import minimize

        e_of = [[] for _ in range(n_g)]
        for idx, (i, k, j, l) in enumerate(edges):
            e_of[i].append(idx)
            e_of[j].append(idx)

        centers = np.array([g["cart"][g["mono"].conn].mean(0) for g in groups])

        def transform(gidx, t, w):
            th = float(np.linalg.norm(w))
            if th > 1e-10:
                kx = w / th
                K = np.array([[0, -kx[2], kx[1]],
                              [kx[2], 0, -kx[0]],
                              [-kx[1], kx[0], 0]])
                R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            else:
                R = np.eye(3)
            return (groups[gidx]["cart"] - centers[gidx]) @ R.T + centers[gidx] + t

        def make_obj(gidx):
            my = e_of[gidx]

            def obj(q):
                t, w = q[:3], q[3:]
                X = transform(gidx, t, w)
                E = 0.0
                for ei in my:
                    i, k, j, l = edges[ei]
                    ci = groups[i]["mono"].conn[k]
                    cj = groups[j]["mono"].conn[l]
                    if i == gidx:
                        p1 = X[ci]
                        p2 = groups[j]["cart"][cj]
                    else:
                        p1 = groups[i]["cart"][ci]
                        p2 = X[cj]
                    E += (np.linalg.norm(p2 - p1) - target) ** 2
                E += 0.008 * float(t @ t) + 0.008 * float(w @ w)
                return E
            return obj

        for _sweep in range(rotate_sweeps):
            for gidx in range(n_g):
                obj = make_obj(gidx)
                res = minimize(obj, np.zeros(6), method="Powell",
                               options={"maxiter": 400, "xtol": 1e-4, "ftol": 1e-6})
                t = res.x[:3]
                w = res.x[3:]
                groups[gidx]["cart"] = transform(gidx, t, w)
                groups[gidx]["shift"] = groups[gidx].get("shift", np.zeros(3)) + t

    if verbose:
        vals = []
        for i, k, j, l in edges:
            ci = groups[i]["mono"].conn[k]
            cj = groups[j]["mono"].conn[l]
            vals.append(np.linalg.norm(
                mic_cart(groups[j]["cart"][cj], groups[i]["cart"][ci], cell)))
        vals = np.array(vals)
        print(f"  缝合 {len(edges)} 条边：键长 {vals.mean():.3f} ± {vals.std():.3f} Å"
              f"（目标 {target:.2f}），分子最大位移 {np.linalg.norm(shift, axis=1).max():.2f} Å")
    return shift


# ============================================================ 连接位氢
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def repair_linkage_h(groups, m_acc, m_donor, cell, linkage,
                     blen=1.09, h_cut=1.25, heavy_cut=1.75, verbose=True):
    """把连接原子上的氢按**缝合后的**实际键方位重摆一遍（重原子一个都不动）。

    为什么必须单独做这一步：单体的羰基氢是 RDKit 按"sp2、两根键各 120°"
    摆的，但缝合只做刚体平移/旋转，跨分子的那根新键落在哪儿由拓扑决定，
    不会回到醛基氧原来那个方向。实测 BF-COF-1（ctn，亚胺）：

        ∠Ar–C–N = 131.8°（理想 120°）  ⇒  ∠H–C–N 只剩 54.4°
        H···C 1.103 Å + H···N 1.105 Å

    两根键几乎等长，VESTA / MS 就把它画成"一个氢同时连在碳和氮上"——
    化学上不成立，也因为这是弛豫的初始结构，MACE 一上来就在这个氢上
    吃到一个巨大的非键力。

    做法：氢只绕连接原子在"两根重键确定的平面"内旋转，扫一圈候选方向，
    按"离其它所有原子（含周期像）最远、同时偏离角平分线最少"打分择优。
    亚胺碳 ∠Ar–C–N 被拉直时，平分线正好给出 114°/114° 的替代解，
    H···N 回到 ~2.0 Å，价键图像恢复成 C–H 单键 + C=N 双键。

    重原子坐标、晶胞、拓扑都不碰，所以不会改掉已经缝合好的骨架。
    """
    edges = edge_table(groups, m_acc, m_donor, cell)
    if not edges:
        return 0

    sizes = [len(g["mono"].elements) for g in groups]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    P = np.vstack([g["cart"] for g in groups]).copy()
    cell = np.asarray(cell, float)
    Linv = np.linalg.inv(cell)

    def mic(dv):
        f = dv @ Linv
        return (f - np.round(f)) @ cell

    def dmin_to(q, skip):
        """q 到全结构（含周期像）的最近距离，跳过 skip 里的下标。"""
        f = (P - q) @ Linv
        f -= np.round(f)
        d = np.linalg.norm(f @ cell, axis=1)
        d[np.asarray(skip, int)] = 9.0
        return float(d.min())

    fixed, after = 0, []
    for (i, k, j, l) in edges:
        for gi, ki, gj, kj in ((i, k, j, l), (j, l, i, k)):
            mono = groups[gi]["mono"]
            c_loc = mono.conn[ki]
            c_glob = int(off[gi] + c_loc)
            cp = P[c_glob]
            own = groups[gi]["cart"]

            hs = [t for t in range(len(mono.elements))
                  if mono.elements[t] == "H"
                  and float(np.linalg.norm(own[t] - cp)) < h_cut]
            if not hs:
                continue          # 胺端缩合后已经没有氢了，正常
            h_glob = [int(off[gi] + t) for t in hs]

            # ---- 两根重键：分子内（Ar）与跨分子（N/O） ----
            vecs = []
            for t in range(len(mono.elements)):
                if t == c_loc or mono.elements[t] == "H":
                    continue
                dv = mic(own[t] - cp)
                if float(np.linalg.norm(dv)) < heavy_cut:
                    vecs.append(dv)
            other = groups[gj]["mono"]
            vecs.append(mic(groups[gj]["cart"][other.conn[kj]] - cp))

            vecs.sort(key=lambda v: float(np.linalg.norm(v)))
            uniq = []
            for v in vecs:
                u = _unit(v)
                # neighbor_list 会把同一根键的多个周期像都列出来，
                # 同向矢量必须去重，否则"角平分线"退化
                if any(abs(float(u @ _unit(w))) > 0.97 for w in uniq):
                    continue
                uniq.append(v)
                if len(uniq) == 2:
                    break

            if len(uniq) >= 2:
                u1, u2 = _unit(uniq[0]), _unit(uniq[1])
                nrm = np.cross(u1, u2)
                if float(np.linalg.norm(nrm)) < 0.15:      # 近共线，平面退化
                    for axis in (np.array([0., 0., 1.]),
                                 np.array([0., 1., 0.]),
                                 np.array([1., 0., 0.])):
                        nrm = np.cross(u1, axis)
                        if float(np.linalg.norm(nrm)) > 0.15:
                            break
                nrm = _unit(nrm)
                e2 = _unit(np.cross(nrm, u1))
                b = -_unit(u1 + u2)
                if float(np.linalg.norm(u1 + u2)) < 0.05:
                    b = e2                                # 180° 没有平分线
            else:
                b = -_unit(uniq[0]) if uniq else np.array([0., 0., 1.])
                e2 = _unit(np.cross(b, np.array([0., 0., 1.])))
                if float(np.linalg.norm(e2)) < 0.1:
                    e2 = np.array([1., 0., 0.])

            skip = h_glob + [c_glob]
            cands = []
            for dl in np.arange(-80.0, 80.01, 4.0):
                d = _unit(np.cos(np.radians(dl)) * b + np.sin(np.radians(dl)) * e2)
                dm = dmin_to(cp + blen * d, skip)
                cands.append((dm - 0.0020 * abs(dl), dm, float(dl), d))
            cands.sort(key=lambda t: -t[0])

            picked = [cands[0]]
            for extra in cands[1:]:          # CH2（烯醇式）才需要第二个
                if len(picked) >= len(hs):
                    break
                if abs(float(picked[0][3] @ extra[3])) < 0.5:
                    picked.append(extra)

            partner = groups[gj]["cart"][other.conn[kj]]
            for t, (_sc, _dm, _dl, d) in zip(h_glob, picked):
                P[t] = cp + blen * d
                fixed += 1
                after.append(float(np.linalg.norm(mic(P[t] - partner))))

    if fixed:
        for gi in range(len(groups)):
            groups[gi]["cart"] = P[off[gi]:off[gi + 1]].copy()
        if verbose:
            print(f"  重摆 {fixed} 个连接位氢（只转方位，重原子不动）："
                  f"H···{linkage['bond_name'][-1]} 现在 "
                  f"{min(after):.2f}–{max(after):.2f} Å，"
                  f"原来 ~1.10 Å（会被画成两个键）")
    return fixed


# ============================================================ 展开
def expand(built, topo: Topology):
    """把所有节点上的分子拼成一个大晶胞（已经是周期像，无需再做对称展开）。"""
    all_xyz = []
    all_el = []
    for g in built["groups"]:
        all_xyz.append(g["cart"])
        all_el.extend(g["mono"].elements)
        g["n_copies"] = 1
    return all_el, np.vstack(all_xyz)


# ============================================================ 诊断
def linkage_geometry(grp, edges, cell, linkage):
    """量连接键两端的局部几何：新键与单体骨架的夹角、以及跨键的最短原子间距。

    "直线边"理想化（分子臂对齐到两个位点中心的连线）必然把新生键摆在这条
    连线上，于是键角被拉直：亚胺碳上 N-C-C 会接近 180°，而化学上应该是
    ~120°。这不是配错原子，是这个理想化的固有代价；量出来是为了让人知道
    后面弛豫要做什么，也方便判断结果能不能直接用。

    注意：这里只量**重原子**的键角。连接位上那个氢在缝合后会被
    `repair_linkage_h()` 按最终键方位重摆一遍，所以这里量到的
    跨键 N···H 是重摆之后的值（~2.0 Å），不是"氢被顶到氮上"的原始值。
    """
    angs, near = [], []
    for i, k, j, l in edges:
        mi, mj = grp[i]["mono"], grp[j]["mono"]
        ci, cj = mi.conn[k], mj.conn[l]
        P, Q = grp[i]["cart"][ci], grp[j]["cart"][cj]
        for g, m, c in ((i, mi, ci), (j, mj, cj)):
            O = grp[g]["cart"][c]
            # 背离本分子的方向：分子 i 看 i→j，分子 j 看 j→i
            v = mic_cart(Q, P, cell) if g == i else mic_cart(P, Q, cell)
            vn = np.linalg.norm(v)
            if vn < 1e-6:
                continue
            # 骨架侧邻居（重原子 1.2–1.66 Å）：取最近的一个算夹角
            cand = [(np.linalg.norm(grp[g]["cart"][t] - O), t)
                    for t in range(len(m.elements))
                    if t != c and m.elements[t] != "H"]
            cand = [x for x in cand if 1.2 < x[0] < 1.66]
            if cand:
                t = min(cand)[1]
                u = grp[g]["cart"][t] - O
                angs.append(np.degrees(np.arccos(
                    np.clip(u @ v / (np.linalg.norm(u) * vn), -1, 1))))
            # 连接原子上还挂着的 H（亚胺的 =CH-）：看它离对侧连接原子多近
            other = Q if g == i else P
            for t in range(len(m.elements)):
                if m.elements[t] != "H":
                    continue
                if np.linalg.norm(grp[g]["cart"][t] - O) < 1.2:
                    near.append(np.linalg.norm(mic_cart(grp[g]["cart"][t],
                                                        other, cell)))
    return (float(np.mean(angs)) if angs else None,
            float(np.min(near)) if near else None)


def diagnose(elements, xyz, cell, built, linkage):
    banner("结构诊断")
    n = len(elements)
    volume = abs(np.linalg.det(cell))
    mass = sum(_mass(e) for e in elements)
    print(f"  原子数            : {n} / 晶胞")
    print(f"  晶胞体积          : {volume:.1f} Å³")
    print(f"  密度（含 H）      : {mass / 6.02214076e23 / (volume * 1e-24):.3f} g/cm³")

    frac = xyz @ np.linalg.inv(cell)
    from scipy.spatial import cKDTree
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    rep = np.vstack([frac + s for s in shifts]) @ cell
    tree = cKDTree(rep)
    dd, _ = tree.query(xyz, k=2)
    dmin = float(dd[:, 1].min())
    print(f"  最短原子间距      : {dmin:.3f} Å  " +
          ("[OK]" if dmin > 0.95 else "[警告：有原子重叠，需要弛豫]"))

    # 跨分子的连接键（用同一套边表，避免配错对象）
    grp = built["groups"]
    edges = edge_table(grp, built["m_acc"], built["m_donor"], cell)
    vals = np.array([
        np.linalg.norm(mic_cart(grp[j]["cart"][grp[j]["mono"].conn[l]],
                                grp[i]["cart"][grp[i]["mono"].conn[k]], cell))
        for i, k, j, l in edges])
    tag = "[OK]" if abs(vals.mean() - linkage["bond"]) < 0.12 else "[偏长，需弛豫]"
    print(f"  {linkage['bond_name']} 键长         : "
          f"{vals.mean():.3f} ± {vals.std():.3f} Å  "
          f"（{len(vals)} 条边，目标 {linkage['bond']:.2f}）{tag}")

    # 连接键的局部几何：直线边理想化会把键角拉直，这里量出来
    lk_ang, lk_near = linkage_geometry(grp, edges, cell, linkage)
    if lk_ang is not None:
        print(f"  键与骨架夹角      : {lk_ang:.1f}°  （理想 ~120°，"
              f">150° 说明被拉直）")
    if lk_near is not None:
        ok = lk_near > 1.75
        print(f"  跨键最短 N···H    : {lk_near:.2f} Å  正常 ~2.0  "
              f"{'[OK]' if ok else '[偏近]'}")
    if (lk_ang is not None and lk_ang > 150) or \
       (lk_near is not None and lk_near <= 1.75):
        print(f"  说明：新生键被摆在两个位点中心的连线上（分子臂就对齐到这条线），")
        print(f"        于是亚胺碳上的 H 只能留在 N 旁边 —— 这就是上面那个偏近的")
        print(f"        来源。拓扑、连接关系、周期性都是对的，这里只是几何预排；")
        print(f"        弛豫会把键角掰回 ~120°、把 H 推开，框架靠节点轻微倾斜")
        print(f"        吸收，量级约 1 Å（相对晶胞 ~2%）。")

    return {"natoms": n, "volume": volume, "dmin": dmin,
            "bond_mean": float(vals.mean()), "bond_std": float(vals.std()),
            "link_angle": lk_ang, "link_near": lk_near}


def _mass(el):
    return {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007,
            "O": 15.999, "S": 32.06}.get(el, 12.0)


# ============================================================ CIF
def write_cif(path, elements, xyz, cell, title="", extra=""):
    atoms = Atoms(symbols=elements, positions=xyz, cell=cell, pbc=True)
    atoms.wrap()
    ase_write(path, atoms, format="cif")
    # 在文件头补一段说明
    if title or extra:
        with open(path, "r", encoding="utf-8") as f:
            body = f.read()
        head = f"# {title}\n" if title else ""
        for line in extra.splitlines():
            head += f"# {line}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + body)


# ============================================================ 主流程
PRESETS = {
    "bf_cof_1": {
        "node": "NC1(C2)CC(C3)(N)CC2(N)CC3(N)C1",
        "linker": "O=CC1=CC(C=O)=CC(C=O)=C1",
        "topology": "ctn",
        "linkage": "imine",
        "a": 24.798,
        "name": "BF-COF-1",
        "note": "TAA + TFB，Fang et al. Angew. Chem. Int. Ed. 2014, 53, 2878",
    },
    # ---------- dia 家族（4-c + 2-c）----------
    "cof300": {
        "node": "Nc1ccc(cc1)C(c1ccc(N)cc1)(c1ccc(N)cc1)c1ccc(N)cc1",
        "linker": "O=Cc1ccc(C=O)cc1",
        "topology": "dia",
        "linkage": "imine",
        "a": "auto",
        "name": "COF300",
        "note": "TAM + 对苯二甲醛，COF-300 的单体组合（实验为七重穿插，这里是未穿插的理想 dia 网）",
    },
    "cof300_bpda": {
        "node": "Nc1ccc(cc1)C(c1ccc(N)cc1)(c1ccc(N)cc1)c1ccc(N)cc1",
        "linker": "O=Cc1ccc(cc1)-c1ccc(C=O)cc1",
        "topology": "dia",
        "linkage": "imine",
        "a": "auto",
        "name": "COF300-BPDA",
        "note": "TAM + 联苯二醛，把 dia 的边长再拉长 ~10 Å",
    },
    "taa_bda": {
        "node": "NC1(C2)CC(C3)(N)CC2(N)CC3(N)C1",
        "linker": "O=Cc1ccc(C=O)cc1",
        "topology": "dia",
        "linkage": "imine",
        "a": "auto",
        "name": "TAA-BDA",
        "note": "金刚烷四胺 + 对苯二甲醛，dia 里最紧凑的体系（a ≈ 33 Å）",
    },
    "taa_terphenyl": {
        "node": "NC1(C2)CC(C3)(N)CC2(N)CC3(N)C1",
        "linker": "O=Cc1ccc(cc1)-c1ccc(cc1)-c1ccc(C=O)cc1",
        "topology": "dia",
        "linkage": "imine",
        "a": "auto",
        "name": "TAA-TERPHENYL",
        "note": "金刚烷四胺 + 三联苯二醛，a ≈ 53 Å，孔道最开阔",
    },
}


def detect_group(smi, candidates):
    """看这个 SMILES 里含哪几种可连接官能团，返回 {kind: 匹配数}。"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise RuntimeError(f"SMILES 解析失败：{smi}")
    out = {}
    for kind in candidates:
        patt = Chem.MolFromSmarts(CONNECT_SMARTS[kind])
        hits = mol.GetSubstructMatches(patt)
        if hits:
            out[kind] = len({h[0] for h in hits})
    return out


def resolve_kinds(linkage, node_smi, linker_smi):
    """决定节点 / 连接子各用哪个官能团当连接点。

    不硬编码"节点一定是胺、连接子一定是醛"：那样把两个单体写反就直接报
    "找不到 aldehyde 连接点"。这里现场探测两边的官能团，谁含 donor 谁含
    acceptor 自动对号入座。两边都含同一种（罕见）时，按"连接数多的当节点"。
    """
    donor, acceptor = linkage["donor"], linkage["acceptor"]
    alts = list(linkage.get("donor_alt", ()))
    kinds = list(dict.fromkeys([donor, *alts, acceptor]))
    dn = detect_group(node_smi, kinds)
    dl = detect_group(linker_smi, kinds)

    # 主 donor 优先，找不到再试备选（如 β-酮烯胺的 Tp 路线走的是醛而不是甲基酮）
    for d in [donor, *alts]:
        if d in dn and acceptor in dl and d not in dl:
            return d, acceptor                      # 节点出 C，连接子接 N
        if acceptor in dn and d in dl and acceptor not in dl:
            return acceptor, d                      # 节点接 N，连接子出 C
    if any(d in dn for d in [donor, *alts]) and any(d in dl for d in [donor, *alts]):
        raise RuntimeError(
            f"两个单体里都找到 {donor}（{CONNECT_SMARTS[donor]}），没法配对。\n"
            f"  缩合反应需要一边给 {donor}、一边给 {acceptor}。")
    if acceptor in dn and acceptor in dl:
        raise RuntimeError(
            f"两个单体里都找到 {acceptor}（{CONNECT_SMARTS[acceptor]}），没法配对。\n"
            f"  缩合反应需要一边给 {donor}、一边给 {acceptor}。")
    raise RuntimeError(
        f"官能团对不上：节点找到 {list(dn) or '无'}，连接子找到 {list(dl) or '无'}，\n"
        f"  但 {linkage['name']} 需要 {' / '.join([donor, *alts])} + {acceptor}。")


def run(args):
    linkage = LINKAGES[args.linkage]
    node_kind, linker_kind = resolve_kinds(linkage, args.node, args.linker)

    banner("1/4  生成单体 3D 结构")
    node = Monomer(args.node, node_kind, label="节点", seed=0xF00D)
    linker = Monomer(args.linker, linker_kind, label="连接子", seed=0xBEEF)
    node.label = f"节点({node.n_conn}-c)"
    linker.label = f"连接子({linker.n_conn}-c)"
    for m in (node, linker):
        print(f"  {m.label:12s} {m.smi}")
        print(f"  {'':12s} 化学式 {m.formula}  连接点 {m.n_conn} 个  "
              f"臂长 {np.round(m.arm_len, 3)} Å")
        print(f"  {'':12s} 臂间夹角 {np.round(m.arm_angles(), 1)}°")

    topo = Topology(args.topology)

    # 这个拓扑要哪两种连接数，从拓扑定义里读，不写死 4-c + 3-c
    need = sorted({s["coord"] for s in topo.defn["sites"]}, reverse=True)
    have = sorted([node.n_conn, linker.n_conn], reverse=True)
    if have != need:
        print(f"\n  [单体与拓扑不匹配] {topo.key} 要 "
              f"{' + '.join(f'{c}-c' for c in need)} 两个单体，"
              f"你给的是 {node.n_conn}-c + {linker.n_conn}-c。")
        print(f"         按连接数能用的拓扑："
              + ("、".join(f"{k}（{' + '.join(f'{c}-c' for c in sorted({s['coord'] for s in v['sites']}, reverse=True))}）"
                          for k, v in TOPOLOGIES.items()
                          if sorted({s['coord'] for s in v['sites']}, reverse=True) == have)
                 or "（一个都没有，这两个单体自己配不成网）"))
        print(f"         换 --topology 后重跑即可，其它参数不用动。")
        raise SystemExit(2)

    # 没给 --cell 就按单体几何反推（拓扑能提供的最短边跨度 → 反算 a）
    if args.cell is None:
        banner("1.5  反推晶胞边长（没给 -a，按单体几何估算）")
        a_span, _x, _d, _ex = estimate_cell(node, linker, topo, linkage,
                                            verbose=False)
        print(f"  按「边跨度 = 分子跨度」算出 a = {a_span:.3f} Å"
              f"{'（跨度刚好对上）' if _ex else '（跨度对不上，靠缝合补）'}")
        print(f"  但臂越长，这个「刚好」就等于零余量，分子会互相穿插。")
        print(f"  所以再扫一遍晶胞，挑一个既连得上键、又装得下分子的：")
        args.cell, _be, _dm, cell_ok = fit_cell(node, linker, topo, linkage,
                                                verbose=True)
        if abs(args.cell - a_span) > 0.05:
            print(f"  → 实际用 a = {args.cell:.3f} Å"
                  f"（比纯几何值大 {args.cell - a_span:+.3f} Å，给分子留出间隙）")
        else:
            print(f"  → 实际用 a = {args.cell:.3f} Å")
        if not cell_ok:
            print(f"  [警告] 没有找到两全的晶胞：")
            print(f"         键长偏差 {_be:.2f} Å，最近间距 {_dm:.2f} Å。")
            print(f"         这不是调参数能解决的 —— 单体臂角与 {topo.key} 位点夹角差太多，")
            print(f"         臂越长，这个角度差放大成端点错位就越严重。")
            print(f"         可以试：① 换更短的臂（少一个苯环）② 换拓扑（dia/pts）")
            print(f"         ③ 就这样写出去，交给弛豫去释放内应力。")
        print(f"  有实验值就直接用 -a 覆盖，文献晶胞通常比几何估算准。")
    else:
        cell_ok = True

    banner("2/4  摆放分子")
    built = build(node, linker, topo, linkage, args.cell, fit_param=not args.fix_param)

    # 边表只用分数坐标建，跟分子摆在哪个镜像无关
    _edges = edge_table(built["groups"], built["m_acc"], built["m_donor"], built["cell"])

    if not args.no_stitch:
        banner("2.5  缝合连接键（刚体平移，不改单体内部几何）")
        backup = [g["cart"].copy() for g in built["groups"]]
        stitch(built["groups"], built["m_acc"], built["m_donor"],
               built["cell"], linkage["bond"])
        d_after = min_pair_distance(built["groups"], built["cell"])
        if d_after < 0.85 and not args.force_stitch:
            for g, b in zip(built["groups"], backup):
                g["cart"] = b
            print(f"  [已回退] 缝合后最近原子间距只有 {d_after:.2f} Å，"
                  f"分子被挤到一起了。")
            if cell_ok:
                print(f"  这不是缝合算法的问题：a = {args.cell:.2f} Å 装不下这两个单体。")
                try:
                    a_fit, _be2, _dm2, ok2 = fit_cell(node, linker, topo, linkage,
                                                      verbose=False)
                    if ok2 and abs(a_fit - args.cell) > 0.05:
                        print(f"  建议晶胞： -a {a_fit:.2f}    "
                              f"（已实测：键能连上、原子不重叠）")
                    else:
                        print(f"  换个晶胞也解决不了，原因见上面 1.5 节的警告。")
                except Exception:
                    print(f"  建议改成 -a auto，让脚本自己找合适的晶胞。")
            else:
                print(f"  原因见 1.5 节：「单体臂角和 ctn 位点夹角差太多」，"
                      f"没有两全的晶胞。")
            print(f"  当前写出的结构是纯拓扑位置（键长偏大），"
                  f"想强行缝合加 --force-stitch。")

    banner("2.6  连接位氢按最终键角重摆")
    repair_linkage_h(built["groups"], built["m_acc"], built["m_donor"],
                     built["cell"], linkage)

    banner("3/4  展开成周期晶胞")
    elements, xyz = expand(built, topo)
    for mono in (built["m_hi"], built["m_lo"]):
        n = sum(g["n_copies"] for g in built["groups"] if g["mono"] is mono)
        print(f"  {mono.label:12s} × {n:3d} 个分子/晶胞  ({n * len(mono.conn)} 根连接臂)")
    print(f"  合计 {len(elements)} 个原子/晶胞")
    print(f"  输出方式：P1 全晶胞（VESTA / MS / OVITO 直接可读）")

    banner("4/4  写出 CIF")
    extra = [
        f"built by build_cof_3d.py",
        f"topology = {topo.key} ({topo.defn['symbol']}, #{topo.defn['spacegroup']})",
        f"node     = {node.smi}",
        f"linker   = {linker.smi}",
        f"linkage  = {linkage['name']}  ({linkage['equation']})",
        f"a = {args.cell} A, x({topo.param['name']}) = {built['x']:.5f}"
        if topo.param else f"a = {args.cell} A",
        "P1 expanded cell",
    ]
    write_cif(args.out + ("" if args.out.lower().endswith(".cif") else ".cif"),
              elements, xyz, built["cell"],
              title=f"{topo.key} 3D COF", extra="\n".join(extra))
    out = args.out if args.out.lower().endswith(".cif") else args.out + ".cif"
    print(f"  已保存: {os.path.abspath(out)}")

    info = diagnose(elements, xyz, built["cell"], built, linkage)

    if linkage.get("caution"):
        print(f"\n  [注意] {linkage['caution']}")

    print("\n下一步：这个结构只是几何预排，节点处有内应力，必须弛豫。")
    print(f"  python opt_cof.py {os.path.basename(out)} -p mace-mp --relax-steps 500 --cell-steps 800")
    print("  三维骨架不能用 QCOF（只训过单层）；MACE-MP-0 + D3 是这里唯一稳妥的选择。")
    return out, info


def cmd_inspect(path):
    from ase.io import read
    a = read(path, format="cif")
    cp = a.cell.cellpar()
    banner(f"检查 {os.path.basename(path)}")
    print(f"  原子数   : {len(a)}")
    print(f"  化学式   : {a.get_chemical_formula()}")
    print(f"  晶胞     : a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
          f"α={cp[3]:.2f} β={cp[4]:.2f} γ={cp[5]:.2f}")
    print(f"  体积     : {a.get_volume():.1f} Å³")
    print(f"  密度     : {a.get_masses().sum() / 6.02214076e23 / (a.get_volume() * 1e-24):.3f} g/cm³")
    d = a.get_all_distances(mic=True)
    np.fill_diagonal(d, 99)
    print(f"  最短间距 : {d.min():.3f} Å")


def cmd_list():
    banner("可用拓扑")
    for k, v in TOPOLOGIES.items():
        print(f"  {k:6s} {v['symbol']:8s} #{v['spacegroup']:3d}  {v['desc']}")
        print(f"         位点: " + ", ".join(
            f"{s['coord']}-c @ {s['label']}" for s in v["sites"]))
    banner("可用连接方式")
    for k, v in LINKAGES.items():
        print(f"  {k:12s} {v['name']}")
        print(f"               {v['equation']}")
    banner("内置预设")
    for k, v in PRESETS.items():
        print(f"  {k:12s} {v['name']}  ({v['note']})")


# ============================================================ 交互向导
def _ask(prompt, default=None):
    """要一行输入。空了就用默认值；没有默认值就一直追问。"""
    tail = f" [{default}]" if default is not None else ""
    while True:
        try:
            s = input(f"  {prompt}{tail}: ").strip()
        except EOFError:
            raise SystemExit("\n  输入结束，退出。")
        except KeyboardInterrupt:
            raise SystemExit("\n  已取消。")
        if s:
            return s
        if default is not None:
            return str(default)
        print("    这一项不能空着。")


def _ask_choice(prompt, options, default):
    """从编号列表里选一个。返回 key。"""
    keys = list(options)
    print()
    for i, k in enumerate(keys, 1):
        mark = "*" if k == default else " "
        print(f"  {mark}[{i}] {k:12s} {options[k]}")
    while True:
        s = _ask(prompt, default=str(keys.index(default) + 1))
        if s.isdigit() and 1 <= int(s) <= len(keys):
            return keys[int(s) - 1]
        if s in options:
            return s
        print(f"    请输入 1–{len(keys)}，或者直接写名字。")


def _preview_monomer(smi, linkage, label):
    """在向导里立刻把单体读一遍，报化学式 / 连接数 / 臂长，避免跑到一半才炸。"""
    d = detect_group(smi, [linkage["donor"], linkage["acceptor"]])
    if not d:
        print(f"    [警告] 没找到 {linkage['donor']} 或 {linkage['acceptor']} 官能团")
        return d
    kind = next(iter(d))
    m = Monomer(smi, kind, label=label)
    print(f"    {m.formula}  |  {kind} 连接点 ×{m.n_conn}  |  "
          f"臂长 {np.round(m.arm_len, 2)} Å  |  夹角 {np.round(m.arm_angles(), 0)}°")
    return d


def wizard():
    banner("三维 COF 搭建向导")
    print("  直接回车 = 用 [ ] 里的默认值；Ctrl+C 随时退出。")

    # ---------- 1. 单体来源 ----------
    keys = list(PRESETS)
    print("\n  内置预设：")
    for i, k in enumerate(keys, 1):
        v = PRESETS[k]
        print(f"   [{i}] {v['name']:10s} {v['topology']:4s}/{v['linkage']:6s}  "
              f"a={v['a']:.3f}  {v['note']}")
    print("   [0] 自己输入两个单体的 SMILES")

    pick = _ask("选择", default="0")
    cfg = None
    if pick in keys:
        cfg = PRESETS[pick]
    elif pick.isdigit() and 1 <= int(pick) <= len(keys):
        cfg = PRESETS[keys[int(pick) - 1]]

    if cfg:
        node_smi, linker_smi = cfg["node"], cfg["linker"]
        topology, linkage = cfg["topology"], cfg["linkage"]
        a, out = cfg["a"], cfg["name"]
        print(f"\n  预设 {cfg['name']}")
        print(f"    节点   {node_smi}")
        print(f"    连接子 {linker_smi}")
    else:
        topology = _ask_choice("拓扑",
                               {k: v["desc"] for k, v in TOPOLOGIES.items()}, "ctn")
        linkage = _ask_choice("连接方式",
                              {k: f"{v['name']}  {v['equation']}"
                               for k, v in LINKAGES.items()}, "imine")
        L = LINKAGES[linkage]
        print(f"\n  需要两个单体分别提供 {L['donor']} 和 {L['acceptor']}。")
        print("  哪个当节点都行，脚本自动认（连接数多的自然当节点）。")

        while True:
            print("\n  节点（4 连接的星形/四面体单体）：")
            node_smi = _ask("SMILES")
            try:
                _preview_monomer(node_smi, L, "节点")
                break
            except Exception as e:
                print(f"    [错误] {e}")
        while True:
            print("\n  连接子（3 连接的三角单体）：")
            linker_smi = _ask("SMILES")
            try:
                _preview_monomer(linker_smi, L, "连接子")
                break
            except Exception as e:
                print(f"    [错误] {e}")
        a, out = None, f"{topology}_{linkage}"

    # ---------- 2. 晶胞 ----------
    print()
    if a is None:
        s = _ask("晶胞边长 a (Å)（留空 = 按单体几何反推）", default="auto")
    else:
        s = _ask("晶胞边长 a (Å)", default=f"{a:.3f}")
    if s.lower() in ("auto", "a", "自动", "算", "估算"):
        a = None
    else:
        try:
            a = float(s)
        except ValueError:
            print("    不是数字，改用按几何反推。")
            a = None

    # ---------- 3. 输出 ----------
    out = _ask("输出文件名", default=out)

    print("\n" + "-" * 74)
    print(f"  节点    {node_smi}")
    print(f"  连接子  {linker_smi}")
    print(f"  拓扑    {topology}   连接方式 {linkage}")
    print(f"  晶胞 a  {'自动反推' if a is None else f'{a:.3f} Å'}")
    print(f"  输出    {out}.cif")
    print("-" * 74)
    if _ask("开始搭建？(y/n)", default="y").lower() not in ("y", "yes", ""):
        print("  已取消。")
        return

    args = argparse.Namespace(
        preset=None, node=node_smi, linker=linker_smi,
        topology=topology, linkage=linkage, cell=a, out=out,
        fix_param=False, no_stitch=False, force_stitch=False,
        wizard=False, inspect=None, list_topologies=False,
    )
    run(args)


def main():
    p = argparse.ArgumentParser(
        description="三维 COF 搭建器：SMILES + 拓扑 + 连接方式 → CIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例子:
  python build_cof_3d.py --preset bf_cof_1 -o BF-COF-1
  python build_cof_3d.py -n "NC1(C2)CC(C3)(N)CC2(N)CC3(N)C1" \\
      -l "O=CC1=CC(C=O)=CC(C=O)=C1" --topology ctn --linkage imine -a 24.798

不给任何参数直接运行 → 进入交互向导，一步步问 SMILES / 拓扑 / 连接方式。
""")
    p.add_argument("--preset", choices=list(PRESETS), help="内置组合")
    p.add_argument("-n", "--node", help="节点 SMILES（4 连接）")
    p.add_argument("-l", "--linker", help="连接子 SMILES（3 连接）")
    p.add_argument("--topology", default="ctn", help="拓扑名（ctn / dia）")
    p.add_argument("--linkage", default="imine", help="连接方式（imine/boronate/ketoenamine）")
    p.add_argument("-a", "--cell", metavar="A",
                   help="立方晶胞边长 a (Å)；写 auto 让脚本反推并自动挑尺寸"
                        "（单体较大时务必用 auto，手填的 a 常常装不下）")
    p.add_argument("-o", "--out", default="COF3D", help="输出文件名")
    p.add_argument("--fix-param", action="store_true",
                   help="不调拓扑位点参数，直接用默认值")
    p.add_argument("--no-stitch", action="store_true",
                   help="跳过键长缝合（保留纯拓扑位置，分子会明显错开）")
    p.add_argument("--force-stitch", action="store_true",
                   help="缝合后即使出现原子重叠也不回退")
    p.add_argument("--inspect", metavar="CIF", help="只检查一个已有 CIF")
    p.add_argument("--list-topologies", action="store_true", help="列出支持的拓扑/连接方式")
    p.add_argument("--wizard", action="store_true",
                   help="强制进入交互向导")
    args = p.parse_args()

    if args.list_topologies:
        cmd_list()
        return
    if args.inspect:
        cmd_inspect(args.inspect)
        return

    # 裸跑 / 显式 --wizard → 向导。没有可交互终端（管道、重定向、
    # 自动化调用）时不进，否则会卡在 input() 上。
    bare = len(sys.argv) == 1
    if args.wizard or bare:
        if args.wizard or (sys.stdin is not None and sys.stdin.isatty()):
            wizard()
            return
        print("这条命令带参数才有意义；不给参数时会进入交互向导，"
              "但当前没有可交互的终端。\n")
        p.print_help()
        return

    if args.preset:
        cfg = PRESETS[args.preset]
        args.node = args.node or cfg["node"]
        args.linker = args.linker or cfg["linker"]
        args.topology = cfg["topology"]
        args.linkage = cfg["linkage"]
        args.cell = args.cell or cfg["a"]
        if args.out == "COF3D":
            args.out = cfg["name"]

    if not args.node or not args.linker:
        p.error("要么给 --preset，要么同时给 -n 和 -l，"
                "或者不带任何参数运行进入交互向导")

    if args.linkage not in LINKAGES:
        p.error(f"未知连接方式 {args.linkage}（可选：{', '.join(LINKAGES)}）")

    if args.cell is not None:
        s = str(args.cell).strip().lower()
        if s in ("auto", "a", "自动", "估算"):
            args.cell = None
        else:
            try:
                args.cell = float(s)
            except ValueError:
                p.error(f"-a 要一个数字或 auto，你给的是 {args.cell!r}")

    run(args)


if __name__ == "__main__":
    main()
