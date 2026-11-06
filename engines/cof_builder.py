# -*- coding: utf-8 -*-
"""
cof_builder.py —— 通用 COF 搭建器（命令行，不用改脚本里的配置区）
=================================================================

给两个单体的 SMILES，直接搭出二维 COF 单层的周期 CIF + 俯视图。

    python cof_builder.py -n "<节点 SMILES>" -l "<连接子 SMILES>"

最省事的写法（推荐）：只写分子本身，一个 * 都不用加，再说明用什么键连起来
------------------------------------------------------------------------
    python cof_builder.py -n "O=Cc1cc(C=O)cc(C=O)c1" -l "Nc1cc(N)cc(N)c1" -k imine

``-k`` 就是"连接键类型"。脚本据此：

  1. 自己去找单体上的官能团（-CHO、-NH2、-CH3、-COOH、-B(OH)2、-Br …）；
  2. 按该连接的化学、**真的删掉离去基团**（亚胺就摘掉醛基的 O、胺上 2 个 H，
     β-酮烯胺只摘 1 个 H，并把环上 OH 互变成 C=O），得到"装配态"单体；
  3. 连接数由官能团个数定（3 个 -CHO → hcb，4 个 → sql），不用再猜 --arms；
  4. 键长按连接键类型给默认值，也照样可以用 --bond 覆盖。

  ``-k auto``（默认）会先试着识别：两个单体上正好是互补的官能团时自动套用上面的
  逻辑；识别不出来才退回"按几何挑最对称 sp2 位点"的老路。

支持的连接键（``--list-links`` 可查）
------------------------------------
    imine        亚胺      -CHO + -NH2   → -CH=N-            C=N  1.28
    ketoenamine  β-酮烯胺  -CHO(邻位 -OH) + -NH2 → -NH-CH=C(-C=O)-  C–N 1.36
    enol         烯醇式    -CHO + -NH2   → -CH=N-，酚羟基保留     C=N  1.28
    azine        连氮      -CHO + 肼     → -CH=N-N=CH-          C=N  1.28
    hydrazone    腙        -CHO + 酰肼   → -CH=N-NH-            C=N  1.28
    vinylene     乙烯基    -CHO + 芳甲基 → -CH=CH-              C=C  1.34
    cyanovinylene 氰基乙烯 -CHO + 芳乙腈 → -C(CN)=CH-            C=C  1.35
    amide        酰胺      -COOH + -NH2  → -C(=O)-NH-           C–N  1.35
    suzuki       芳基单键  -B(OH)2 + -Br/-Cl/-I → Ar-Ar         C–C  1.47

能自动判断的东西
----------------
1. 拓扑     节点上有几个 ``*`` → 3 连接走 hcb（蜂窝）、4 连接走 sql（四方）
            （``--topology`` 可强制指定）
2. 装配方式 节点 + 线形连接子（连接子 2 个 ``*``）；
            或两个同连接数的节点直接成键（C3+C3 / C4+C4，vertex 模式）
3. 缩合形式 检测到 -NH2 + -CHO 会自动按亚胺缩合（-NH2 + O=CH- → -N=CH- + H2O）
4. 连接键长 按连接两端原子的化学环境自动猜（也可 ``--bond`` 覆盖）：
              N 上带 H         → C–N 单键 1.36（β-酮烯胺 / 烯胺，如 TFP）
              N 上不带 H       → C=N    1.28（亚胺）
              C 与 C 都带 H     → C=C    1.34（乙烯基）
              其余 C–C（sp1）  → 1.43   其余 C–C  → 1.46
              其他元素对        → 1.43（保守值，建议 --bond 覆盖）

单体 SMILES 怎么填
------------------
* 用 ``*`` 标出"这个位置要跟对方连起来"（``*[*]`` 这种写法会自动规整）；
* 方括号里显式写氢数：``[CH]`` / ``[NH]`` / ``[nH]``，否则 RDKit 按价态自动补满，
  常常多一个 H，装配后就会重叠；
* 单体可以写"缩合后"的样子，也可以写缩合前的 -NH2 / -CHO（脚本会自动缩合）。

  更省事的办法：**一个 * 都不写**，只给分子本身 + 用 ``-k`` 说明连接键类型
  （见上面那段），脚本自己认官能团、自己做缩合。只有 ``-k auto`` 又认不出
  官能团时，才会退回"按几何挑最对称 sp2 位点"。

内置组合（``--preset``）
------------------------
    tfp_ketoenamine  TFP 酮式 + 三(4-氨基苯基丁二炔基)苯 —— β-酮烯胺 hcb COF
    tfp_tapb         TFP 酮式 + 1,3,5-三(4-氨基苯基)苯  —— 经典 TpPa/TAPB 型
    triazine / lzul / c3c3_imine / porphine_sql（见 build_cof_smiles.py）
    ``--list-presets`` 看全部。

    注意：preset 只是 6 个快捷方式，不是能力的边界。任意单体只要能写成
    SMILES，就可以直接搭。preset 不参与任何判断逻辑。

能泛化到什么程度（以及前提）
----------------------------
只要满足下面三条，脚本对单体本身"零记忆"，全靠几何算出来：

1. 二维单层。这个脚本只搭 2D COF（不做 3D、不做互穿、不做多层堆叠）。
2. 节点是"平面、重对称"的 3 或 4 连接单元：
   · 3 连接 → 三条臂近似 120°，走 hcb（蜂窝）
   · 4 连接 → 四条臂近似 90°/90°/180°，走 sql（四方）
   跑之前会先量一遍"臂夹角"打出来，不合就提前告诉你（见「单体体检」）。
3. 第二个单体二选一：
   · 线形连接子（2 个 ``*``，两端共线）→ 夹在节点之间（edge 模式）
   · 与节点同连接数的另一个节点（3+3 或 4+4）→ 两个节点直接成键（vertex 模式）

连接化学不限：亚胺 / β-酮烯胺 / 烯醇 / 连氮 / 腙 / 乙烯基 / 酰胺 / 芳基偶联……
要么用 ``-k`` 说清楚（脚本自己删离去基团、自己定键长），要么按"连接两端原子的
化学环境"自动猜键长，还可以用 ``--bond`` 直接指定。

``--list-presets`` 之外的单体，一般这样喂：
    python cof_builder.py -n "<节点SMILES>" -l "<连接子SMILES>" -k imine -o 我的COF
或者直接运行 ``python cof_builder.py``（不带参数）进向导，选 ``0`` 自己输。

常用命令
--------
    python cof_builder.py --list-presets
    python cof_builder.py --list-links
    python cof_builder.py -n "O=Cc1cc(C=O)cc(C=O)c1" -l "Nc1cc(N)cc(N)c1" -k imine -o LZU1
    python cof_builder.py -n "*[CH]c1cc([CH]*)cc([CH]*)c1" -l "*[N]c1ccc([N]*)cc1"
    python cof_builder.py --preset tfp_ketoenamine -o Ref_COF
    python cof_builder.py -n "<node>" -l "<linker>" --bond 1.40 --c 3.8 --no-plot

产物：``<out>.cif``、``<out>_layer.xyz``（未折回晶胞，方便画图）、``<out>_layer.png``
"""

import argparse
import math
import sys

try:                      # rdkit 缺失时 _load_engine() 会给中文提示并退出
    from rdkit import Chem
except ImportError:
    Chem = None

# ---------------------------------------------------------------- 依赖 / 引擎
def _load_engine():
    """把搭建引擎和第三方依赖都载进来，缺东西时给中文提示而不是一堆 traceback。"""
    for pkg, name in (("rdkit", "rdkit"), ("ase", "ase")):
        try:
            __import__(pkg)
        except ImportError:
            sys.exit(f"\n缺少 Python 包 '{name}'，请先安装：\n"
                     f"    pip install rdkit ase numpy matplotlib\n")
    try:
        import build_cof_smiles as eng
    except ImportError:
        sys.exit("\n找不到 build_cof_smiles.py（搭建引擎）。\n"
                 "请把它和 cof_builder.py 放在同一个文件夹里。\n")
    return eng


def _load_lattice_sym():
    """层内对称化模块（可选：没有它也能建模，只是不会投影到 P4/P6）。"""
    try:
        import lattice_sym
        return lattice_sym
    except ImportError:
        return None


def _chem_formula(a):
    """带占位意识的化学式（4 个 0.5 的 H 要算成 2 个 H）。"""
    ls = _load_lattice_sym()
    if ls is not None and 'occupancy' in a.arrays:
        return ls.formula(a)
    return a.get_chemical_formula()


def _chem_count(a):
    """(位点数, 等效原子数)。--nh disorder 时两者不同：108 个位点 = 106 个等效原子。"""
    ls = _load_lattice_sym()
    if ls is not None and 'occupancy' in a.arrays:
        neff = ls.n_effective(a)
        if abs(neff - len(a)) > 0.01:
            return len(a), neff
    return len(a), None


# ---------------------------------------------------------------- 内置组合
EXTRA_PRESETS = {
    "tfp_ketoenamine": dict(
        topology="hcb",
        node="*[*]NC(C=C1)=CC=C1C#CC#CC2=CC(C#CC#CC3=CC=C(N[*])C=C3)"
             "=CC(C#CC#CC4=CC=C(N[*])C=C4)=C2",
        linker="*[CH]=C1C(=O)C(=[CH][*])C(=O)C(=[CH][*])C1=O",
        bond=1.36,
        note="三(4-氨基苯基丁二炔基)苯 + TFP 酮式 → β-酮烯胺 hcb COF"),
    "tfp_tapb": dict(
        topology="hcb",
        node="*Nc1ccc(-c2cc(-c3ccc(N*)cc3)cc(-c3ccc(N*)cc3)c2)cc1",
        linker="*[CH]=C1C(=O)C(=[CH][*])C(=O)C(=[CH][*])C1=O",
        bond=1.36,
        note="1,3,5-三(4-氨基苯基)苯 + TFP 酮式 → 经典 TpPa/TAPB 型 β-酮烯胺 COF"),
}


def all_presets(eng):
    out = {}
    for k, v in getattr(eng, "PRESETS", {}).items():
        out[k] = dict(v)
    out.update(EXTRA_PRESETS)
    return out


# ================================================================ 连接键化学
# 只给"分子本身"的 SMILES（一个 * 都不写）+ 一句"用什么键连"，剩下的都在这里：
#
#   ① 认官能团（醛、伯胺、肼、酰肼、芳甲基、羧酸、硼酸、卤代）
#   ② 按该连接化学删掉离去基团，并把该留的 H 留对
#      —— 亚胺：醛基 O + 胺上 2 个 H 走掉，醛碳保留 1 个 H、N 上 0 个 H
#      —— β-酮烯胺：醛基 O + 胺上 1 个 H 走掉，N 上留 1 个 H，环上 OH 互变成 C=O
#   ③ 挂上 *，交给后面的几何引擎
#
# 为什么不用字符串替换：芳香环、邻/间/对位、以及"哪个环碳带 OH"都只能从图上问，
# 文本层面猜不准；而且删原子以后原子序号会变，必须是纯粹的图操作。

LINK_TYPES = {
    "imine": dict(
        label="亚胺  -CH=N-",
        bond=1.28,
        pair=("aldehyde", "amine"),
        formula="-CHO + -NH2 → -CH=N- + H2O",
        h_a=1, h_b=0),
    "ketoenamine": dict(
        label="β-酮烯胺  -NH-CH=C(-C=O)-",
        bond=1.36,
        pair=("aldehyde", "amine"),
        formula="-CHO + -NH2 → -NH-CH=C(-C=O)- + H2O（酮式互变）",
        h_a=1, h_b=1, keto=True),
    "enol": dict(
        label="烯醇式  -CH=N-（酚羟基保留）",
        bond=1.28,
        pair=("aldehyde", "amine"),
        formula="-CHO + -NH2 → -CH=N- + H2O，邻位 -OH 原样保留",
        h_a=1, h_b=0),
    "azine": dict(
        label="连氮  -CH=N-N=CH-",
        bond=1.28,
        pair=("aldehyde", "hydrazine"),
        formula="-CHO + H2N-NH2 → -CH=N-N=CH- + 2 H2O",
        h_a=1, h_b=0),
    "hydrazone": dict(
        label="腙  -CH=N-NH-",
        bond=1.28,
        pair=("aldehyde", "hydrazide"),
        formula="-CHO + -NH-NH2 → -CH=N-NH- + H2O",
        h_a=1, h_b=0),
    "vinylene": dict(
        label="乙烯基  -CH=CH-",
        bond=1.34,
        pair=("aldehyde", "methyl"),
        formula="-CHO + Ar-CH3 → -CH=CH- + H2O（Knoevenagel / 羟醛缩合）",
        h_a=1, h_b=1, soft=0.55),
    "cyanovinylene": dict(
        label="氰基乙烯  -C(CN)=CH-",
        bond=1.35,
        pair=("aldehyde", "cyanomethyl"),
        formula="-CHO + Ar-CH2CN → -C(CN)=CH- + H2O"
                "（Knoevenagel，sp2-COF / sp2c-COF 的连接键）",
        # 氰基是接在双键碳上的刚性线形取代基，连接键若按 od 斜着接（soft 大），
        # 连接子苯环会被推到氰基上（实测 C···C 0.86 Å）。这里取 0：
        # 连接键沿臂轴走，氰基甩到侧向，实测最近间距回到 1.09 Å。
        h_a=1, h_b=0, soft=0.0),
    "amide": dict(
        label="酰胺  -C(=O)-NH-",
        bond=1.35,
        pair=("acid", "amine"),
        formula="-COOH + -NH2 → -C(=O)-NH- + H2O",
        h_a=0, h_b=1, soft=0.70),
    "imide": dict(
        label="酰亚胺  -C(=O)-N(-C(=O))-（聚酰亚胺）",
        bond=1.40,
        pair=("anhydride", "aniline"),
        formula="二酐 + 二胺 → 五元酰亚胺环 + H2O（PI-COF）",
        h_a=0, h_b=0,
        imide=True,
        # 注意：不要给 imide 指定 tilt。PMDA 的酐环是刚性的五元环，强制
        # 走"直线边"（tilt=0）会把整个酐环横推 ~1 Å，羰基氧直接插进
        # 三嗪节点，实测 N–O 0.90 Å。交给装配器在"几何锚定 / 逐臂 sp2"
        # 两个模型里按补氢后的真实质量择优即可（实测 N–O 恢复到 1.59 Å）。
        soft=0.70),
    "suzuki": dict(
        label="芳基–芳基单键  Ar-Ar",
        bond=1.47,
        pair=("boronic", "halide"),
        formula="Ar-B(OH)2 + Ar'-X → Ar-Ar'（Suzuki 偶联）",
        h_a=0, h_b=0),
}

# 中文 / 常见别名 → 规范名
LINK_ALIASES = {
    "亚胺": "imine", "c=n": "imine", "schiff": "imine",
    "酮烯胺": "ketoenamine", "β-酮烯胺": "ketoenamine", "beta-ketoenamine": "ketoenamine",
    "b-ketoenamine": "ketoenamine", "keto": "ketoenamine", "enamine": "ketoenamine",
    "烯胺": "ketoenamine", "tfp": "ketoenamine",
    "烯醇": "enol", "enol-oh": "enol",
    "连氮": "azine", "azine-c=n-n=c": "azine",
    "腙": "hydrazone", "hydrazide": "hydrazone",
    "乙烯基": "vinylene", "vinyl": "vinylene", "c=c": "vinylene", "knoevenagel": "vinylene",
    "氰基乙烯": "cyanovinylene", "氰基烯": "cyanovinylene", "cyano": "cyanovinylene",
    "cyanovinyl": "cyanovinylene", "cyano-vinylene": "cyanovinylene",
    "sp2": "cyanovinylene", "sp2c": "cyanovinylene", "sp2-cof": "cyanovinylene",
    "氰基": "cyanovinylene", "c(cn)=c": "cyanovinylene",
    "酰胺": "amide", "amide-bond": "amide",
    "聚酰亚胺": "imide", "酰亚胺": "imide", "pi": "imide", "polyimide": "imide",
    "pi-cof": "imide", "imide-bond": "imide", "酐": "imide", "anhydride": "imide",
    "芳基": "suzuki", "c-c": "suzuki", "aryl": "suzuki", "biaryl": "suzuki",
    "auto": "auto", "自动": "auto", "": "auto",
}

_GROUP_TEXT = {
    "aldehyde": "-CHO（醛基）", "amine": "-NH2（伯胺）",
    "hydrazine": "H2N-NH2（肼）", "hydrazide": "-NH-NH2（酰肼/单取代肼）",
    "methyl": "Ar-CH3（芳甲基）", "cyanomethyl": "Ar-CH2CN（芳乙腈/氰甲基）",
    "acid": "-COOH（羧酸）",
    "boronic": "-B(OH)2（硼酸）", "halide": "Ar-X（氯/溴/碘代）",
    "anhydride": "-C(=O)-O-C(=O)-（酸酐）", "aniline": "Ar-NH2（芳香伯胺）",
}


def normalize_link(name):
    """把用户写的连接键类型规整成规范名；认不出就抛 ValueError。"""
    if name is None:
        return "auto"
    k = str(name).strip().lower().replace(" ", "").replace("_", "-")
    if k in LINK_TYPES:
        return k
    if k in LINK_ALIASES:
        return LINK_ALIASES[k]
    for full, _spec in LINK_TYPES.items():         # 允许写标签里的英文前缀
        if len(k) >= 3 and k in full:
            return full
    raise ValueError(f"不认识的连接键类型 '{name}'。可用："
                     + "、".join(sorted(LINK_TYPES)) + "（或 auto）")


# ---------------------------------------------------------------- 官能团识别
def _find_aldehyde(mol):
    """醛基碳：C=O、只有 1 个 H、重原子配位 2（排除酮 / 羧酸 / 酯）。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetTotalNumHs() != 1 or a.GetDegree() != 2:
            continue
        if len([b for b in a.GetBonds()
                if b.GetBondTypeAsDouble() == 2.0
                and b.GetOtherAtom(a).GetAtomicNum() == 8]) == 1:
            out.append(a.GetIdx())
    return out


def _find_amine(mol):
    """伯胺：中性、非环、2 个 H、只有 1 个重原子邻居（-NH2）。"""
    return [a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 0
            and not a.IsInRing() and a.GetTotalNumHs() == 2 and a.GetDegree() == 1]


def _find_hydrazine(mol, terminal_only=False):
    """肼 / 单取代肼上的端位 -NH2。

    terminal_only=False → H2N-NH2（连氮，两个 N 都上）
    terminal_only=True  → R-NH-NH2（腙，只上端位那个 2H 的 N）
    """
    out = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() != 7 or a2.GetAtomicNum() != 7:
            continue
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        if terminal_only:
            for x, y in ((a1, a2), (a2, a1)):
                if (x.GetTotalNumHs() == 2 and x.GetDegree() == 1
                        and y.GetTotalNumHs() == 1):
                    out.append(x.GetIdx())
        elif all(x.GetTotalNumHs() == 2 and x.GetDegree() == 1 for x in (a1, a2)):
            out += [a1.GetIdx(), a2.GetIdx()]
    return out


def _find_methyl(mol):
    """芳甲基：Ar-CH3（连着芳环/sp2 碳的 CH3）。CH2 不算。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetTotalNumHs() != 3 or a.GetDegree() != 1:
            continue
        nb = a.GetNeighbors()[0]
        if nb.GetAtomicNum() == 6 and (nb.GetIsAromatic()
                                       or nb.GetHybridization() == Chem.HybridizationType.SP2):
            out.append(a.GetIdx())
    return out


def _find_cyanomethyl(mol):
    """氰甲基：Ar-CH2-C#N，返回那个亚甲基碳（-CH2CN 的 CH2）。

    这是 Knoevenagel 缩合的另一种写法：芳乙腈 + 醛 → Ar-C(CN)=CH-Ar'。
    跟芳甲基的区别只在亚甲基的另一头是 C#N 而不是 H，所以必须单独认，
    否则 _find_methyl 会因为"只有 2 个 H"把它漏掉。
    """
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetTotalNumHs() != 2 or a.GetDegree() != 2:
            continue
        nbs = list(a.GetNeighbors())
        ar = [n for n in nbs if n.GetAtomicNum() == 6
              and (n.GetIsAromatic() or n.GetHybridization() == Chem.HybridizationType.SP2)]
        cn = [n for n in nbs if n.GetAtomicNum() == 6
              and any(b.GetBondTypeAsDouble() == 3.0
                      and b.GetOtherAtom(n).GetAtomicNum() == 7 for b in n.GetBonds())]
        if ar and cn:
            out.append(a.GetIdx())
    return out


def _find_acid(mol):
    """羧基碳：C(=O)-OH。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetTotalNumHs() != 0:
            continue
        dbl = [b.GetOtherAtom(a) for b in a.GetBonds()
               if b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetAtomicNum() == 8]
        sgl = [b.GetOtherAtom(a) for b in a.GetBonds()
               if b.GetBondTypeAsDouble() == 1.0 and b.GetOtherAtom(a).GetAtomicNum() == 8
               and b.GetOtherAtom(a).GetTotalNumHs() >= 1]
        if len(dbl) == 1 and len(sgl) == 1:
            out.append(a.GetIdx())
    return out


def _find_boronic(mol):
    """硼酸：返回挂着 -B(OH)2 的芳环碳（真正成键的那个原子）。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 5:
            continue
        if len([n for n in a.GetNeighbors()
                if n.GetAtomicNum() == 8 and n.GetTotalNumHs() >= 1]) < 2:
            continue
        for c in a.GetNeighbors():
            if c.GetAtomicNum() == 6 and c.GetIsAromatic():
                out.append(c.GetIdx())
    return out


def _find_halide(mol):
    """卤代芳烃：返回挂卤素的芳环碳。"""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ("Cl", "Br", "I") or a.GetDegree() != 1:
            continue
        c = a.GetNeighbors()[0]
        if c.GetAtomicNum() == 6 and c.GetIsAromatic():
            out.append(c.GetIdx())
    return out


def _find_anhydride(mol):
    """酸酐：返回酐桥氧（环内 O，两个邻居都是羰基碳）的索引。

    这个氧在成环时会被替换成 N（见 _anhydride_to_imide），所以它既是
    "要改的原子"也是"连接位"——索引不变，不用重排。
    判据里要求两个邻居都在同一个环里，避免误伤酯基/羧酸酐链。
    """
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 8 or not a.IsInRing() or a.GetDegree() != 2:
            continue
        cs = a.GetNeighbors()
        if not all(c.GetAtomicNum() == 6 for c in cs):
            continue
        if not all(any(b.GetBondTypeAsDouble() == 2.0
                       and b.GetOtherAtom(c).GetAtomicNum() == 8
                       for b in c.GetBonds()) for c in cs):
            continue
        pair = {cs[0].GetIdx(), cs[1].GetIdx()}
        if any(pair <= r for r in rings):
            out.append(a.GetIdx())
    return out


def _find_aniline_c(mol):
    """芳香伯胺：返回挂 -NH2 的那个芳环碳（而不是 N 本身）。

    聚酰亚胺里 N 是"两个人共用"的：它既要连二酐的两个羰基，又要连芳环。
    所以装配时 N 由二酐一侧提供（酐氧 → N），二胺一侧只贡献芳环碳。
    """
    out = []
    for n in _find_amine(mol):
        for nb in mol.GetAtomWithIdx(n).GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIsAromatic():
                out.append(nb.GetIdx())
    return out



_FINDERS = {
    "aldehyde": _find_aldehyde,
    "amine": _find_amine,
    "hydrazine": lambda m: _find_hydrazine(m, terminal_only=False),
    "hydrazide": lambda m: _find_hydrazine(m, terminal_only=True),
    "methyl": _find_methyl,
    "cyanomethyl": _find_cyanomethyl,
    "acid": _find_acid,
    "boronic": _find_boronic,
    "halide": _find_halide,
    "anhydride": _find_anhydride,
    "aniline": _find_aniline_c,
}


# ---------------------------------------------------------------- 图操作工具
def _remove_and_cap(mol, drop, caps):
    """删原子 + 设定指定原子的显式氢数，返回 (RWMol, 旧序号→新序号)。"""
    drop = sorted({int(i) for i in drop})
    rw = Chem.RWMol(mol)
    for i in reversed(drop):                       # 倒着删，前面的序号才不跳
        rw.RemoveAtom(i)
    remap, k = {}, 0
    for o in range(mol.GetNumAtoms()):
        if o in drop:
            continue
        remap[o] = k
        k += 1
    for o, nh in caps.items():
        a = rw.GetAtomWithIdx(remap[o])
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(int(nh))
    return rw, remap


def _add_stars(rw, remap, sites):
    """在指定原子（旧序号）上各挂一个 *，返回 (分子, 新序号列表)。"""
    new = []
    for o in sites:
        d = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(remap[o], d, Chem.BondType.SINGLE)
        new.append(remap[o])
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m, new


def _keto_tautomer(mol, exo_sites):
    """亚胺式 → β-酮烯胺式（Keto）。

    做法：醛基碳与芳环碳之间改成双键（环外 =CH-），环上每个"非连接位"的
    环碳把 -OH 变成 =O，于是芳环去芳构化，成为环己二烯酮骨架。TFP 出来的
    就是 -C(=O)-C(=CH-)-C(=O)- 交替的那一圈。
    """
    rw = Chem.RWMol(mol)
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    pairs = []
    for c in exo_sites:
        ar = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
              if n.GetIsAromatic()]
        if len(ar) != 1:
            raise ValueError(
                f"β-酮烯胺要求醛基直接挂在芳环上（原子 {c} 不是），"
                f"做不了酮式互变。\n  改用 -k imine，或者给 -k enol。")
        pairs.append((ar[0], c))
    ca_set = {a for a, _ in pairs}
    touched = set()
    for r in rings:
        if r & ca_set:
            touched |= r

    for i in touched:                              # 去芳构化
        at = rw.GetAtomWithIdx(i)
        at.SetIsAromatic(False)
        at.SetNoImplicit(True)
        at.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        if b.GetBeginAtomIdx() in touched and b.GetEndAtomIdx() in touched:
            b.SetIsAromatic(False)
            b.SetBondType(Chem.BondType.SINGLE)

    for ca, cex in pairs:                          # 环外 C=C
        b = rw.GetBondBetweenAtoms(ca, cex)
        b.SetBondType(Chem.BondType.DOUBLE)
        b.SetIsAromatic(False)
        a = rw.GetAtomWithIdx(cex)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(1)

    for i in sorted(touched - ca_set):             # 环上 -OH → =O
        at = rw.GetAtomWithIdx(i)
        oh = [n for n in at.GetNeighbors()
              if n.GetAtomicNum() == 8 and n.GetTotalNumHs() >= 1]
        if len(oh) != 1:
            raise ValueError(
                f"β-酮烯胺需要每个醛基邻位的环碳上带 -OH（原子 {i} 上没有），"
                f"这个单体做不了酮式。\n  改用 -k imine（亚胺式）试试。")
        o = oh[0]
        rw.GetBondBetweenAtoms(i, o.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        o.SetNoImplicit(True)
        o.SetNumExplicitHs(0)

    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def _anhydride_to_imide(mol, o_sites):
    """把酸酐的桥氧就地换成 N，得到酰亚胺环（聚酰亚胺的成环步骤）。

    为什么是"换原子"而不是"删氧再加氮"：
    删原子会让后面所有原子序号往前跳，已经算好的连接位、显式氢数全部对不上；
    而 O→N 只改元素符号，原子数、键、序号一个不动，五元环也是现成的。

    化学上对应：-C(=O)-O-C(=O)- + H2N-Ar → -C(=O)-N(Ar)-C(=O)- + H2O
    （真实过程先开环成酰胺酸，再脱水闭环；这里直接给闭环产物）。
    """
    rw = Chem.RWMol(mol)
    for o in o_sites:
        a = rw.GetAtomWithIdx(o)
        if a.GetAtomicNum() != 8 or a.GetDegree() != 2:
            raise ValueError(f"原子 {o} 不是酸酐桥氧，做不了酰亚胺环。")
        a.SetAtomicNum(7)                    # O → N
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    # 这里**不能** sanitize：此刻 N 还是二配位（两个羰基），价态不满，
    # RDKit 会以 "Can't kekulize" 报错。等 _add_stars 把 * 挂上去、
    # N 凑够三根键之后再统一 sanitize（那一步在 _add_stars 里做）。
    return rw.GetMol()


def _n_ortho_oh(mol, sites):
    """醛基**邻位**的环碳上有几个带 -OH（判断能不能做 β-酮烯胺）。

    这里要往外看两层：醛基碳 → 挂它的环碳 → 该环碳的环上邻居 → 邻居身上的 O。
    只看第一层会数成 0（醛基那个环碳自己没有 OH），TFP 就会被误判成普通亚胺。
    """
    n = 0
    for c in sites:
        for ar in mol.GetAtomWithIdx(c).GetNeighbors():
            if ar.GetAtomicNum() != 6 or not ar.GetIsAromatic():
                continue
            for nb in ar.GetNeighbors():
                if (nb.GetIdx() == c or nb.GetAtomicNum() != 6
                        or not nb.GetIsAromatic()):
                    continue
                if any(o.GetAtomicNum() == 8 and o.GetTotalNumHs() >= 1
                       and not o.IsInRing() for o in nb.GetNeighbors()):
                    n += 1
    return n


def _pick_subset(eng, smiles, sites, want):
    """官能团比需要的多时（比如 6 个醛只要 4 条臂），按对称性挑 want 个。"""
    from itertools import combinations
    if want >= len(sites):
        return list(sites)
    try:
        pos, dirs = eng._site_geometry(smiles, sites)
        best = None
        for combo in combinations(sites, want):
            sc = eng._site_score(pos, dirs, combo)[0]
            if best is None or sc < best[0]:
                best = (sc, combo)
        return list(best[1]) if best else list(sites)[:want]
    except Exception:
        return list(sites)[:want]


def apply_linkage(eng, key, node_smiles, linker_smiles, arms=None, arms2=None,
                  verbose=True):
    """按连接键类型把两个"纯分子"变成"带 * 的装配态单体"。

    返回 (节点 SMILES, 第二体 SMILES, 节点连接数, 第二体连接数,
          默认键长, 说明文字)。
    """
    spec = LINK_TYPES[key]
    role_a, role_b = spec["pair"]
    ma, mb = Chem.MolFromSmiles(node_smiles), Chem.MolFromSmiles(linker_smiles)
    if ma is None or mb is None:
        raise ValueError(f"SMILES 无法解析：{node_smiles if ma is None else linker_smiles}")

    # 谁是"醛/酸/硼酸"那一侧，靠官能团判断，跟用户输入顺序无关
    sa, sb = _FINDERS[role_a](ma), _FINDERS[role_b](mb)
    if len(sa) < 2 or len(sb) < 2:
        sa2, sb2 = _FINDERS[role_a](mb), _FINDERS[role_b](ma)
        if len(sa2) >= 2 and len(sb2) >= 2:
            ma, mb, sa, sb = mb, ma, sa2, sb2
    if len(sa) < 2 or len(sb) < 2:
        raise ValueError(
            f"按 '{key}'（{spec['label']}）连不起来：\n"
            f"  需要「{_GROUP_TEXT[role_a]}」和「{_GROUP_TEXT[role_b]}」两边各至少 2 个，\n"
            f"  实际只找到 {len(sa)} 个和 {len(sb)} 个"
            f"（调换两个单体的顺序也一样）。\n"
            f"  用 --inspect \"<SMILES>\" 可以看清每个原子到底是几个氢、什么杂化。")

    # 谁是"节点"看臂数多少，跟它是醛还是胺无关：
    #   3 个 -NH2 的 TAPB + 2 个 -CHO 的对苯二甲醛，节点是 TAPB。
    # （之前这里按"醛那一侧当节点"来排，导致"胺在前、醛在后"时直接判错。）
    node_is_b = len(sb) > len(sa)

    # 官能团多于需要的臂数时，挑最对称的子集
    if arms and arms < (len(sb) if node_is_b else len(sa)):
        if node_is_b:
            sb = _pick_subset(eng, Chem.MolToSmiles(mb), sb, arms)
        else:
            sa = _pick_subset(eng, Chem.MolToSmiles(ma), sa, arms)
    if arms2 and arms2 < (len(sa) if node_is_b else len(sb)):
        if node_is_b:
            sa = _pick_subset(eng, Chem.MolToSmiles(ma), sa, arms2)
        else:
            sb = _pick_subset(eng, Chem.MolToSmiles(mb), sb, arms2)

    n_node, n_other = (len(sb), len(sa)) if node_is_b else (len(sa), len(sb))
    if n_node < 2 or n_node > 4:
        raise ValueError(f"节点那一侧有 {n_node} 个可反应官能团，"
                         f"目前只支持 3 连接（hcb）和 4 连接（sql）节点。")
    if n_other < 2 or n_other > 4:
        raise ValueError(f"另一个单体上有 {n_other} 个可反应官能团，"
                         f"做连接子要 2 个、做第二个节点要和节点一样多（{n_node} 个）。")
    if n_other != 2 and n_other != n_node:
        raise ValueError(
            f"两边官能团数对不上：一边 {n_node} 个、另一边 {n_other} 个。\n"
            f"  合法的搭配只有两种：\n"
            f"    · 节点 {n_node} + 线形连接子 2（edge 模式）\n"
            f"    · 节点 {n_node} + 另一个节点 {n_node}（vertex 模式）\n"
            f"  检查一下两个单体的取代位置，或者用 --arms/--arms2 指定臂数。")

    # 醛基邻位带 -OH 的单体（TFP 那类）实际是以酮式成键的：用亚胺式建出来
    # 酚羟基会和亚胺上的 H 挤在一起，这里提前说一句，别等它报重叠。
    if role_a == "aldehyde" and not spec.get("keto"):
        n_oh = _n_ortho_oh(ma, sa)
        if n_oh >= 2 and verbose:
            print(f"[提示] 这个醛基单体邻位有 {n_oh} 处 -OH（TFP 这类）："
                  f"真实 COF 里它多半以 β-酮烯胺成键，\n"
                  f"       要那个形态请用 -k ketoenamine；"
                  f"现在的 -k {key} 会保留酚羟基。")

    # ---------- 删离去基团 + 保氢 ----------
    # 聚酰亚胺：先把二酐的桥氧就地变成 N（闭环成酰亚胺），
    # 二胺那一侧则把 -NH2 整个摘掉，只留芳环碳当连接位。
    if spec.get("imide"):
        ma = _anhydride_to_imide(ma, sa)

    h_a, h_b = spec["h_a"], spec["h_b"]
    if role_a == "aldehyde":
        drop_a = [b.GetOtherAtom(ma.GetAtomWithIdx(c)).GetIdx()
                  for c in sa for b in ma.GetAtomWithIdx(c).GetBonds()
                  if b.GetBondTypeAsDouble() == 2.0
                  and b.GetOtherAtom(ma.GetAtomWithIdx(c)).GetAtomicNum() == 8]
    elif role_a == "acid":
        drop_a = [b.GetOtherAtom(ma.GetAtomWithIdx(c)).GetIdx()
                  for c in sa for b in ma.GetAtomWithIdx(c).GetBonds()
                  if b.GetBondTypeAsDouble() == 1.0
                  and b.GetOtherAtom(ma.GetAtomWithIdx(c)).GetAtomicNum() == 8]
    elif role_a == "boronic":
        drop_a = []
        for c in sa:
            for b in ma.GetAtomWithIdx(c).GetNeighbors():
                if b.GetAtomicNum() == 5:
                    drop_a.append(b.GetIdx())
                    drop_a += [n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != c]
    else:
        drop_a = []

    drop_b = []
    if role_b == "halide":
        for c in sb:
            drop_b += [n.GetIdx() for n in mb.GetAtomWithIdx(c).GetNeighbors()
                       if n.GetSymbol() in ("Cl", "Br", "I")]
    elif spec.get("imide"):
        # 酰亚胺的 N 已经由酸酐那一侧提供，二胺的 N 必须整个拿掉，
        # 否则会在 N–芳环之间多插一个氮，变成肼的价态。
        drop_b = _find_amine(mb)

    rw_a, map_a = _remove_and_cap(ma, drop_a, {c: h_a for c in sa})
    mol_a, new_a = _add_stars(rw_a, map_a, sa)
    if spec.get("keto"):
        mol_a = _keto_tautomer(mol_a, new_a)

    rw_b, map_b = _remove_and_cap(mb, drop_b, {c: h_b for c in sb})
    mol_b, _new_b = _add_stars(rw_b, map_b, sb)

    smi_a, smi_b = Chem.MolToSmiles(mol_a), Chem.MolToSmiles(mol_b)
    # 臂多的那个当节点输出（跟它是醛还是胺无关）
    if node_is_b:
        smi_node, smi_other = smi_b, smi_a
    else:
        smi_node, smi_other = smi_a, smi_b

    note = (f"{spec['label']}：{spec['formula']}"
            f"（节点 {n_node} 连接 / 第二体 {n_other} 连接）")
    if verbose:
        print(f"[连接键] {spec['label']}   {spec['formula']}")
        print(f"[连接键] 节点官能团 {n_node} 个 → {smi_node}")
        print(f"[连接键] 第二体官能团 {n_other} 个 → {smi_other}")
    return smi_node, smi_other, n_node, n_other, float(spec["bond"]), note


def detect_linkage(node_smiles, linker_smiles, verbose=True):
    """两个单体都不带 * 时，按官能团互补猜一个连接键类型；猜不出返回 None。

    判据的顺序很要紧（前面的优先）：

      腙   醛 + 单取代肼（-NH-NH2）  —— 酰肼的端位 -NH2 也长着"伯胺"的样子，
                                       不先认它就会被当成亚胺
      连氮 醛 + 肼（H2N-NH2）        —— 两个 N 同样是"2 个 H、1 个重原子邻居"
      亚胺/酮烯胺  醛 + 伯胺          —— 醛基邻位带 -OH 的（TFP 这类）给酮烯胺，
                                       那才是它在 COF 里的实际成键形态
      酰胺 羧酸 + 伯胺
      乙烯基 醛 + 芳甲基
      Suzuki 硼酸 + 卤代
    """
    ma, mb = Chem.MolFromSmiles(node_smiles), Chem.MolFromSmiles(linker_smiles)
    if ma is None or mb is None:
        return None

    ald_a, ald_b = _find_aldehyde(ma), _find_aldehyde(mb)
    nh2_a, nh2_b = _find_amine(ma), _find_amine(mb)

    if (len(ald_a) >= 2 and len(_find_hydrazine(mb, terminal_only=True)) >= 2) or \
       (len(ald_b) >= 2 and len(_find_hydrazine(ma, terminal_only=True)) >= 2):
        if verbose:
            print("[连接键] 自动识别：醛 + 酰肼/单取代肼 → 腙（-CH=N-NH-）")
        return "hydrazone"

    if (len(ald_a) >= 2 and len(_find_hydrazine(mb)) >= 2) or \
       (len(ald_b) >= 2 and len(_find_hydrazine(ma)) >= 2):
        if verbose:
            print("[连接键] 自动识别：醛 + 肼 → 连氮（-CH=N-N=CH-）")
        return "azine"

    for m_ald, s_ald, s_ami in ((ma, ald_a, nh2_b), (mb, ald_b, nh2_a)):
        if len(s_ald) >= 2 and len(s_ami) >= 2:
            key = "ketoenamine" if _n_ortho_oh(m_ald, s_ald) >= 2 else "imine"
            if verbose:
                print(f"[连接键] 自动识别：一边 {len(s_ald)} 个 -CHO、"
                      f"另一边 {len(s_ami)} 个 -NH2 → {LINK_TYPES[key]['label']}")
            return key

    if (len(_find_anhydride(ma)) >= 2 and len(_find_aniline_c(mb)) >= 2) or \
       (len(_find_anhydride(mb)) >= 2 and len(_find_aniline_c(ma)) >= 2):
        if verbose:
            print("[连接键] 自动识别：酸酐 + 芳香伯胺 → 酰亚胺环（聚酰亚胺）")
        return "imide"
    if (len(_find_acid(ma)) >= 2 and len(nh2_b) >= 2) or \
       (len(_find_acid(mb)) >= 2 and len(nh2_a) >= 2):
        if verbose:
            print("[连接键] 自动识别：羧酸 + 伯胺 → 酰胺（-C(=O)-NH-）")
        return "amide"
    if (len(ald_a) >= 2 and len(_find_cyanomethyl(mb)) >= 2) or \
       (len(ald_b) >= 2 and len(_find_cyanomethyl(ma)) >= 2):
        if verbose:
            print("[连接键] 自动识别：醛 + 氰甲基 → 氰基乙烯（-C(CN)=CH-）")
        return "cyanovinylene"
    if (len(ald_a) >= 2 and len(_find_methyl(mb)) >= 2) or \
       (len(ald_b) >= 2 and len(_find_methyl(ma)) >= 2):
        if verbose:
            print("[连接键] 自动识别：醛 + 芳甲基 → 乙烯基（-CH=CH-）")
        return "vinylene"
    if (len(_find_boronic(ma)) >= 2 and len(_find_halide(mb)) >= 2) or \
       (len(_find_boronic(mb)) >= 2 and len(_find_halide(ma)) >= 2):
        if verbose:
            print("[连接键] 自动识别：硼酸 + 卤代芳烃 → 芳基–芳基单键")
        return "suzuki"
    return None


# ---------------------------------------------------------------- 连接键长推断
def _is_carbonyl(atom):
    return atom.GetAtomicNum() == 6 and any(
        b.GetOtherAtom(atom).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2.0
        for b in atom.GetBonds())


def _has_carbonyl_near(atom, max_bonds=2):
    """连接原子附近（≤max_bonds 根键）有没有羰基碳。

    β-酮烯胺里 C=O 落在环外 =CH- 的"隔壁的隔壁"（=CH–C(环)–C=O），
    所以只看直接邻居不够，要往外看两层。
    """
    seen = {atom.GetIdx()}
    frontier = [atom]
    found = []
    for _ in range(max_bonds):
        nxt = []
        for a in frontier:
            for nb in a.GetNeighbors():
                if nb.GetIdx() in seen:
                    continue
                seen.add(nb.GetIdx())
                nxt.append(nb)
        found.extend(nxt)
        frontier = nxt
    return any(_is_carbonyl(a) for a in found)


def _conn_atoms_of(smiles, eng):
    """列出每个连接位（*）背后那个真实原子，并描述它的化学环境。"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles}")
    mol = eng.prune_star_chains(mol)
    info = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        c = a.GetNeighbors()[0]
        info.append(dict(idx=c.GetIdx(), elem=c.GetSymbol(),
                         n_h=int(c.GetTotalNumHs()),
                         hyb=str(c.GetHybridization()),
                         nbr_has_co=_has_carbonyl_near(c)))
    return info


def guess_bond(node, linker, eng):
    """按连接原子的环境猜新键的键长，返回 (键长, 依据说明)。"""
    try:
        ca = _conn_atoms_of(node, eng)
        cb = _conn_atoms_of(linker, eng)
    except ValueError:
        return 1.34, "SMILES 解析异常，回退到默认 1.34"
    elems = {ca[i]["elem"] for i in range(len(ca))} | {cb[i]["elem"] for i in range(len(cb))}
    na = sum(1 for x in ca if x["elem"] == "N")
    nc_alkyne = sum(1 for x in ca + cb if x["elem"] == "C" and x["hyb"] == "SP")
    nh_node = [x for x in ca if x["elem"] == "N" and x["n_h"] >= 1]
    nh_lk = [x for x in cb if x["elem"] == "N" and x["n_h"] >= 1]
    keto = any(x["nbr_has_co"] for x in ca + cb if x["elem"] == "C")

    if elems == {"N", "C"}:
        if nh_node or nh_lk:
            why = ("C–N 单键（连接 N 上保留 1 个 H）"
                   + ("，且相邻碳带 C=O → β-酮烯胺/烯胺" if keto else ""))
            return (1.36 if keto else 1.40), why
        return 1.28, "C=N（亚胺，N 上无 H）"
    if elems == {"N", "N"}:
        return 1.40, "N–N（腙/连氮类）"
    if elems == {"C", "C"}:
        if nc_alkyne:
            return 1.43, "C–C（至少一端是 sp 炔碳）"
        # 有一端挂着 2 个以上 H（CH2/CH3）就是 sp3，只能单键：
        # "两端都带 H" 这个判据单独用会把 CH2 误判成 C=C（实测 1.34 偏短 0.12 Å）
        if any(x["n_h"] >= 2 or x["hyb"] == "SP3"
               for x in ca + cb if x["elem"] == "C"):
            return 1.46, "C–C 单键（有一端是 CH2/CH3 的 sp3 碳）"
        both_h = all(x["n_h"] >= 1 for x in ca + cb if x["elem"] == "C")
        if both_h and not keto:
            return 1.34, "C=C（两端碳都带 H，乙烯基/烯基）"
        return 1.46, "C–C（芳基–芳基/烯基–芳基）"
    if elems == {"C", "O"}:
        return 1.43, "C–O（酯/醚/缩醛类）"
    return 1.34, f"无法判断（元素对 {sorted(elems)}），用默认 1.34，建议 --bond 覆盖"


# ---------------------------------------------------------------- 结构体检
# 配位数的合法区间。超出就说明成键是坏的（重叠、多打了原子、连接点没接上）
_VALENCE = {
    # N 的下限给 1：氰基（-C#N）里的 N 只有一根键，按距离判键时 C#N 也算 1 根，
    # 写成 2 会把所有 sp2-COF / 氰基乙烯体系全部误判成"不合价"（实测踩过）。
    "H": (1, 1), "C": (2, 4), "N": (1, 3), "O": (1, 2),
    "S": (1, 2), "B": (2, 3), "P": (3, 4), "F": (1, 1), "Cl": (1, 1),
}


def check_structure(atoms, verbose=True, allow_nonplanar=False):
    """成键体检：配位数 + 重叠 + 离面。返回 (致命问题, 说明列表)。

    为什么需要这一关：装配时如果两个单体的臂接不上（臂角不符、连接位写错、
    ``*`` 数量对不上），引擎不会抛异常，它会把第二个单体原样摞上去，
    于是得到一个"所有原子 z 都等于 0.5、看起来完全共面"、
    但原子间距 0.28 Å 的废结构。只查平面度是查不出来的 —— 必须查成键。
    """
    import numpy as np
    from ase.data import atomic_numbers, covalent_radii

    sym = atoms.get_chemical_symbols()
    n = len(atoms)
    if n == 0:
        return True, ["[致命] 结构里没有原子"]

    R = covalent_radii[[atomic_numbers[s] for s in sym]]
    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, 99.0)

    # 成键判据：0.4 ~ 1.15×(r_i+r_j)。下限挡掉"叠在一起"的假键
    cut = 1.15 * (R[:, None] + R[None, :])
    B = (D < cut) & (D > 0.4)
    nc = B.sum(axis=1)

    msgs = []
    fatal = []

    # ---- 1. 重叠 ----
    dmin = float(D.min())
    if dmin < 0.95:
        pair = np.unravel_index(D.argmin(), D.shape)
        i, j = int(pair[0]), int(pair[1])
        fatal.append(
            f"[致命] 最近原子间距 {dmin:.3f} Å（{sym[i]}{i + 1}–{sym[j]}{j + 1}），"
            f"原子叠在一起了")
        close = sum(1 for a in range(n) for b in range(a + 1, n) if D[a][b] < 0.95)
        if close > 1:
            fatal.append(f"       全晶胞共有 {close} 对原子距离 < 0.95 Å")
    elif verbose:
        msgs.append(f"  最近原子间距 {dmin:.3f} Å  [OK]")

    # ---- 2. 配位数 ----
    bad = []
    for i in range(n):
        lo, hi = _VALENCE.get(sym[i], (1, 6))
        if not (lo <= nc[i] <= hi):
            nb = [(f"{sym[j]}{j + 1}", round(float(D[i][j]), 2))
                  for j in np.nonzero(B[i])[0]]
            bad.append((i, int(nc[i]), nb))
    if bad:
        fatal.append(f"[致命] {len(bad)} 个原子的成键数不合价（正常应 碳≤4、氢=1、氮≤3）")
        for i, k, nb in bad[:6]:
            fatal.append(f"       {sym[i]}{i + 1} 有 {k} 根键：{nb}")
        if len(bad) > 6:
            fatal.append(f"       …… 另有 {len(bad) - 6} 个")
    elif verbose:
        msgs.append("  配位数  全部落在合法价态内  [OK]")

    # ---- 3. 离面 ----
    frame, d_all, d_heavy, dev, soft = frame_offplane(atoms, B=B)
    if frame > 0.80:
        if allow_nonplanar:
            msgs.append(f"  [放行] 骨架离面 {frame:.2f} Å —— 这一层是褶皱的"
                        f"（--allow-nonplanar）")
        else:
            fatal.append(f"[致命] 骨架离面最大 {frame:.2f} Å"
                         f"（重原子 {d_heavy:.2f} Å，全部含甲基氢 {d_all:.2f} Å），"
                         f"这一层不是平面")
    elif verbose:
        extra = ""
        if soft.any() and d_all > frame + 0.05:
            extra = f"（另有 {int(soft.sum())} 个甲基/亚甲基氢在面上/下，属正常）"
        msgs.append(f"  离面量  骨架 {frame:.3f} Å / 重原子 {d_heavy:.3f} Å / "
                    f"全部 {d_all:.3f} Å  "
                    f"{'[OK]' if frame < 0.30 else '[注意: 骨架翘出平面]'}{extra}")

        # 只翘了 sp2 上的氢 —— 常见于把 sp2 的连接碳写成了 sp3
        if frame > 0.30 and d_heavy < 0.15:
            out = [i for i in range(n)
                   if sym[i] == "H" and not soft[i] and abs(dev[i]) > 0.30]
            if out:
                msgs.append(f"       有 {len(out)} 个 sp2 上的氢垂直翘出层面"
                            f"（{', '.join(f'{sym[i]}{i + 1}' for i in out[:8])}）"
                            f"，多半是这些氢所连的碳被当成了 sp3")

    if verbose:
        for m in msgs:
            print(m)
    return bool(fatal), fatal


# ---------------------------------------------------------------- 结果输出
def plane_stats(atoms):
    """实测平面度：所有原子到最佳拟合平面的距离。

    返回 (最大离面, 重原子最大离面, 每个原子的离面量, 法向)。

    为什么要单独算：``info["thickness"]`` 是引擎自报的"生长厚度"，
    只统计它自己铺出去的那一层。一旦有原子翘出去（把 sp2 的碳写成 sp3、
    或者单体带四面体中心），它照样报 0.000。这里改用 SVD 拟合平面，
    谁翘出平面、翘了多少，一眼就能看到。
    """
    import numpy as np
    p = atoms.get_positions()
    q = p - p.mean(axis=0)
    nrm = np.linalg.svd(q)[2][2]
    dev = q @ nrm
    heavy = np.array([s != "H" for s in atoms.get_chemical_symbols()])
    d_all = float(np.abs(dev).max()) if len(dev) else 0.0
    d_heavy = float(np.abs(dev[heavy]).max()) if heavy.any() else 0.0
    return d_all, d_heavy, dev, nrm


def frame_offplane(atoms, B=None):
    """骨架离面：重原子 + 连在 sp2 碳上的氢，排除甲基/亚甲基的氢。

    为什么不能直接看"全部原子离面"：甲氧基（-OCH3）、甲基上的 3 个氢本来
    就是 1 个在面内、2 个在面外 ±0.9 Å，这是正确的 sp3 构型。用全部原子
    去判平面度，会把结构完全正常的 COF 判死（COF3 的 C78H66N12O6 实测
    重原子 0.02 Å、全部 0.92 Å，就是被甲氧基的氢卡住的）。

    规则：氢所连的碳如果还挂着 ≥2 个 H（CH2/CH3），这些氢算"软氢"，
    不计入骨架离面；其余原子（重原子 + 芳环/烯烃上的氢）都算骨架。

    返回 (骨架最大离面, 全部最大离面, 重原子最大离面, 逐原子离面, 软氢掩码)。
    """
    import numpy as np
    sym = atoms.get_chemical_symbols()
    n = len(sym)
    d_all, d_heavy, dev, nrm = plane_stats(atoms)
    if B is None:
        from ase.data import atomic_numbers, covalent_radii
        R = covalent_radii[[atomic_numbers[s] for s in sym]]
        D = atoms.get_all_distances(mic=True)
        np.fill_diagonal(D, 99.0)
        B = (D < 1.15 * (R[:, None] + R[None, :])) & (D > 0.4)
    h_cnt = {i: sum(1 for j in np.nonzero(B[i])[0] if sym[j] == "H")
             for i in range(n) if sym[i] != "H"}
    soft = np.zeros(n, dtype=bool)
    for i in range(n):
        if sym[i] != "H":
            continue
        nb = [int(j) for j in np.nonzero(B[i])[0] if sym[j] != "H"]
        if nb and h_cnt.get(nb[0], 0) >= 2:
            soft[i] = True
    hard = ~soft
    frame = float(np.abs(dev[hard]).max()) if hard.any() else 0.0
    return frame, d_all, d_heavy, dev, soft


_XH_IDEAL = {"C": 1.09, "N": 1.01, "O": 0.97, "S": 1.34, "B": 1.19, "P": 1.42}


def fix_xh_geometry(atoms, tol=0.03):
    """把 X–H 摆成理想构型：键长归位 + 酰胺/吡咯型 N–H 摆回层平面。返回修了几个。

    两个都是装配过程自带的几何小缺陷，不是物性：

      · ``project_planar`` / 折环把原子压到 z=0 时，O–H 这种短键会被压短
        0.05~0.08 Å（TFP 烯醇式实测 0.90 Å），于是体检把它当成"原子叠在一起"；
      · ETKDG 给的酰胺 N–H 常常歪出平面 0.3 Å，让『骨架离面』虚高。

    只修键长落在 0.7~1.35 Å（确实成键、只是差一点）的 H；
    真出问题的（1.5 Å 还挂着、或者 0.5 Å 糊在一起）一律不动，留给体检去报。
    """
    import numpy as np
    from ase.data import atomic_numbers, covalent_radii

    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    n = len(sym)
    if n == 0:
        return 0
    R = covalent_radii[[atomic_numbers[s] for s in sym]]
    D = atoms.get_all_distances(mic=False)
    np.fill_diagonal(D, 99.0)
    B = (D < 1.15 * (R[:, None] + R[None, :])) & (D > 0.4)

    def u(v):
        c = float(np.linalg.norm(v))
        return v / c if c > 1e-9 else v

    new = pos.copy()
    fixed = 0
    for i in range(n):
        if sym[i] != "H":
            continue
        nb = [int(j) for j in np.nonzero(B[i])[0]]
        if len(nb) != 1:
            continue
        p = nb[0]
        ideal = _XH_IDEAL.get(sym[p])
        d = float(D[i][p])
        if ideal is None or not (0.70 < d < 1.35):
            continue
        direction = u(pos[i] - pos[p])
        # 酰胺 / 腙 / 吡咯型 N–H：按两个重原子邻居的角平分反方向摆，回到层平面
        if sym[p] == "N":
            heavy = [int(j) for j in np.nonzero(B[p])[0] if sym[j] != "H"]
            if len(heavy) == 2:
                v = -sum(u(pos[j] - pos[p]) for j in heavy)
                if float(np.linalg.norm(v)) > 0.35:   # 两个邻居不共线才谈得上平面
                    direction = u(v)
        target = pos[p] + direction * ideal
        if float(np.linalg.norm(target - pos[i])) > tol:
            new[i] = target
            fixed += 1
    if fixed:
        atoms.set_positions(new)
    return fixed


def print_summary(a, info, name, topology, mode, bond, why, note, files):
    cp = a.cell.cellpar()
    print("\n" + "=" * 64)
    print(f"完成 → {name}")
    print(f"  拓扑      : {topology}（{ '两个节点直接成键' if mode=='vertex' else '节点 + 线形连接子' }）")
    if note:
        print(f"  缩合      : {note}")
    print(f"  连接键长  : {bond:.2f} Å   依据：{why}")
    print(f"  晶胞      : a={cp[0]:.3f}  b={cp[1]:.3f}  c={cp[2]:.3f} Å   γ={cp[5]:.1f}°")
    _n, _neff = _chem_count(a)
    _cnt = (f"{_n} 个位点，等效 {_neff:.0f} 个原子/晶胞" if _neff else f"{_n} 个原子/晶胞")
    print(f"  化学式    : {_chem_formula(a)}   ({_cnt})")
    if info.get("mode") == "vertex":
        spread = info["delta_spread"]
        print(f"  两节点半径: {info['r_a']:.3f} / {info['r_b']:.3f} Å"
              f"    各臂间距偏差 {spread:.3f} Å "
              f"{'[OK]' if spread < 0.15 else '[警告: 单体不是严格三重对称]'}")
    else:
        print(f"  节点半径  : {info['r_node']:.3f} Å    连接子端距 {info['len_linker']:.3f} Å")
    print(f"  单层厚度  : {info['thickness']:.3f} Å "
          f"{'[OK]' if info['thickness'] < 1.0 else '[警告: 层不平]'}")
    frame, d_all, d_heavy, _dev, soft = frame_offplane(a)
    tag = "[OK] 单层严格共面" if frame < 0.10 else "[警告: 骨架离开平面]"
    print(f"  离面（实测）: 骨架 {frame:.3f} Å / 重原子 {d_heavy:.3f} Å / "
          f"全部 {d_all:.3f} Å（含 {int(soft.sum())} 个甲基氢）{tag}")
    print(f"  连接方向偏差: {info['sym_err']:.2f}° "
          f"{'[OK]' if info['sym_err'] < 5 else '[警告: 单体不是平面重对称构型]'}")
    print(f"  文件      : {', '.join(files)}")
    print("=" * 64)


# ---------------------------------------------------------------- 主流程
HELP_HINT = """
需要两个单体的 SMILES，或者直接用内置组合。

  看有哪些内置组合   python cof_builder.py --list-presets
  看有哪些连接键     python cof_builder.py --list-links
  用内置组合         python cof_builder.py --preset tfp_ketoenamine -o Ref_COF
  
最省事：单体只写分子本身（不写 *），再用 -k 说连接键类型
  亚胺 COF           python cof_builder.py -n "O=Cc1cc(C=O)cc(C=O)c1" -l "Nc1cc(N)cc(N)c1" -k imine
  β-酮烯胺 COF       python cof_builder.py -n "O=Cc1c(O)c(C=O)c(O)c(C=O)c1O" -l "<三胺>" -k ketoenamine
  乙烯基 COF         python cof_builder.py -n "<三醛>" -l "Cc1cnc(C)cn1" -k vinylene
  -k 省略 = auto：能认出互补官能团就自动套用，认不出才退回几何选位

也可以自己用 * 标连接位（老写法，照样支持）：
  python cof_builder.py -n "节点SMILES" -l "连接子SMILES" -o 输出名
  python cof_builder.py "节点SMILES" "连接子SMILES"

不确定 SMILES 写得对不对，先体检：
  python cof_builder.py --inspect "CC1=NC=NC=N1"

星号挂在 CH2/CH3（sp3）上也照接，H 留给 MS 补：
  python cof_builder.py -n "...[*]..." -l "...[*]..." --allow-sp3
  （连接原子上的 H 全部留空，骨架连好；MS 里 Adjust Hydrogens 按成键数补齐）

节点骨架天生翘（四芳基 TTF、四苯基乙烯、螺环中心）也照写：
  python cof_builder.py ... --allow-nonplanar
  （苯环绕单键扭出 40~50° 是真实构象，压不平；照写后层是褶皱的）

  什么都不加直接运行，会进入一问一答的向导。
"""


def _arm_angles(eng, smiles, n):
    """量出单体几条"臂"在空间里的分布（两两夹角，度）。

    用连接原子相对单体中心的位置向量来量，而不是用"外伸键方向"——
    后者在 C3 对称的节点上会退化（TFP 的三条外伸键有两支候选方向是对称的，
    挑哪支都"对"，量出来的夹角没有意义）。位置向量没有这个歧义。
    """
    sym, pos, conn, n_h, cap = eng.prepare(smiles, n)
    p = pos[conn]
    c = p.mean(axis=0)
    vecs = [p[i] - c for i in range(len(conn))]
    out = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a, b = vecs[i], vecs[j]
            na = float(sum(float(x) * float(x) for x in a)) ** 0.5
            nb = float(sum(float(x) * float(x) for x in b)) ** 0.5
            d = float(sum(float(x) * float(y) for x, y in zip(a, b))) / (na * nb)
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, d)))))
    return out


def arm_report(eng, smiles, n, label):
    """单体体检：3 连接应当是三条 120°；4 连接应当是 90°/90°/180°。

    这一项比后面的"最近原子间距"更早能看出问题：单体本身是个 60/120° 的
    菱形或三角形构型时，硬塞进四方网必然重叠，改 SMILES 也没用。
    """
    try:
        angs = _arm_angles(eng, smiles, n)
    except Exception as exc:                     # 3D 嵌入失败等
        print(f"   {label:<5} {n} 连接   臂夹角 无法测量 —— {exc}")
        return True                              # 不算"臂角不合"，真正的一步会报错
    if not angs:
        return True
    if n == 4:
        ideal = [180.0 if a > 150 else 90.0 for a in angs]
    elif n == 3:
        ideal = [120.0] * len(angs)
    else:
        ideal = [180.0] * len(angs)
    dev = max(abs(a - i) for a, i in zip(angs, ideal))
    ok = dev <= 15.0
    txt = " / ".join("%.1f°" % a for a in angs)
    print(f"   {label:<5} {n} 连接   臂夹角 {txt}   "
          f"{'[OK]' if ok else '[偏差 %.0f°]' % dev}")
    return ok


def _clean_smiles(s):
    """去掉粘贴时带上的引号 / 空白，避免把 “...” 当成 SMILES 的一部分。"""
    if not s:
        return s
    return (s.strip()
             .strip('"').strip("'")
             .strip('“').strip('”').strip('‘').strip('’')
             .replace(" ", "").replace("\t", ""))


def _wizard(presets, eng=None):
    """没给任何命令行参数时的交互式向导，返回等价的 argv 列表。"""
    keys = sorted(presets)
    print("=" * 64)
    print(" COF 搭建器（交互式）—— SMILES → CIF + 俯视图")
    print("=" * 64)
    print("\n内置组合：")
    for i, k in enumerate(keys, 1):
        note = presets[k].get("note") or presets[k]["topology"] + " 拓扑"
        print(f"  {i}  {k:<16} {note}")
    print("  0  自己输入两个单体的 SMILES + 连接键类型（单体不用写 *）")
    print("  h  看完整命令行用法")

    while True:
        try:
            c = input("\n请输入编号（直接回车 = 1）: ").strip()
        except EOFError:
            # stdin 被关掉（PyCharm 默认运行窗口就是这样），input() 立刻抛
            # EOFError。不提示的话看起来就是“闪一下什么都没发生”。
            print()
            print("=" * 64)
            print(" [提示] 当前运行窗口收不到键盘输入（stdin 已关闭）")
            print("=" * 64)
            print("  原因：运行配置里没勾 \"Emulate terminal in output console\"，")
            print("        或者这个配置的 Parameters 里已经写了 -n/-l，根本不会进向导。")
            print()
            print("  三个办法，任选一个：")
            print("   1. 右上角下拉框选 \"00-交互向导（可输入）\"，直接点绿三角；")
            print("   2. 右键 cof_builder.py -> Modify Run Configuration ->")
            print("      勾上 \"Emulate terminal in output console\"，并清空 Parameters；")
            print("   3. 不用向导，一条命令把参数写全：")
            print('      python cof_builder.py -n "<节点SMILES>" \\')
            print('             -l "<第二体SMILES>" -k imine -o my_cof')
            print()
            print("  想看用法：python cof_builder.py --help")
            print("=" * 64)
            raise SystemExit(0)
        except KeyboardInterrupt:
            print()
            raise SystemExit(0)
        if c == "":
            c = "1"
        if c.lower() in ("h", "help", "?", "？"):
            print(HELP_HINT)
            raise SystemExit(0)

        if c == "0":
            print("\n★ 只写单体本身就行 —— 一个 * 都不用加，后面再选连接键类型。")
            print("  （想自己用 * 标连接位也可以，脚本会照你标的走）")
            print("  -k 可选：", "、".join(LINK_TYPES), "（直接回车 = auto 自动识别）")
            print("例：节点 O=Cc1cc(C=O)cc(C=O)c1"
                  "   第二体 Nc1cc(N)cc(N)c1   连接键 imine\n")
            vals = []
            for tip in ("节点单体 SMILES", "第二个单体 SMILES（连接子或另一个节点）"):
                while True:
                    try:
                        v = _clean_smiles(input(f"  {tip}: "))
                    except (EOFError, KeyboardInterrupt):
                        print()
                        raise SystemExit(0)
                    if v:
                        vals.append(v)
                        break
                    print("    不能为空，请再输一次。")

            # 连接键类型：这里问一句，就不用在 SMILES 里画 * 了
            print("\n  连接键类型（两个单体靠什么键连起来）：")
            keys = list(LINK_TYPES)
            print("    0  自动识别（默认）")
            for i, k in enumerate(keys, 1):
                print(f"    {i:<2} {k:<13} {LINK_TYPES[k]['label']:<28} "
                      f"{LINK_TYPES[k]['formula']}")
            link = ""
            while True:
                try:
                    pick = input("  请输入编号或名称 [直接回车 = 自动]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    pick = ""
                if pick == "":
                    link = "auto"
                    break
                if pick.isdigit() and 1 <= int(pick) <= len(keys):
                    link = keys[int(pick) - 1]
                    break
                try:
                    link = normalize_link(pick)
                    break
                except ValueError:
                    print("    没有这个类型，重输，或直接回车让脚本自动识别。")

            try:
                out = input("  输出文件名前缀 [直接回车 = my_cof]: ").strip() or "my_cof"
            except (EOFError, KeyboardInterrupt):
                out = "my_cof"
            argv = ["-n", vals[0], "-l", vals[1], "-k", link, "-o", out]

            # 星号挂在 sp3 碳上（CH2/CH3）时，正常接出来不共面；
            # 很多人就是想先搭个骨架、H 到 MS 里补，这里直接问一句。
            if eng is not None:
                bad = []
                for v in vals:
                    try:
                        bad += [r for r in eng.diagnose_stars(v) if r["nh"] >= 2]
                    except Exception:
                        pass
                if bad:
                    print(f"\n  注意：有 {len(bad)} 个星号挂在 sp3 碳上"
                          f"（CH2/CH3），接出来的层不会共面。")
                    try:
                        ans = input("  不管它、直接接（H 你自己到 MS 里补）？"
                                    "[y/N]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        ans = "n"
                    if ans in ("y", "yes", "是", "好"):
                        argv.append("--allow-sp3")
            return argv

        if c.isdigit() and 1 <= int(c) <= len(keys):
            k = keys[int(c) - 1]
            try:
                out = input(f"  输出文件名前缀 [直接回车 = {k}]: ").strip() or k
                plot = input("  画 3×3 俯视图 PNG 吗？[Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                out, plot = k, "y"
            argv = ["--preset", k, "-o", out]
            if plot in ("n", "no", "否"):
                argv.append("--no-plot")
            print()
            return argv

        print("  没有这个编号，请重新输入（0 = 自己输入，h = 看用法）。")


def main():
    eng = _load_engine()
    presets = all_presets(eng)

    ap = argparse.ArgumentParser(
        description="两个单体的 SMILES → 任意 2D COF 单层（CIF + 俯视图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="三种用法：\n"
               "  ① 什么都不加直接运行 → 一问一答的向导（最省事）\n"
               "       python cof_builder.py\n"
               "  ② 用内置组合：\n"
               "       python cof_builder.py --list-presets\n"
               "       python cof_builder.py --preset tfp_ketoenamine -o Ref_COF\n"
               "  ③ 自己给两个单体（最简单：不写 *，用 -k 说连接键类型）：\n"
               "       python cof_builder.py -n \"O=Cc1cc(C=O)cc(C=O)c1\" "
               "-l \"Nc1cc(N)cc(N)c1\" -k imine -o LZU1\n"
               "       python cof_builder.py -n \"O=Cc1c(O)c(C=O)c(O)c(C=O)c1O\" "
               "-l \"Nc1ccc(-c2cc(-c3ccc(N)cc3)cc(-c3ccc(N)cc3)c2)cc1\" -k ketoenamine\n"
               "       （-k 省略 = auto：能认出互补官能团就自动套用；"
               "单体写了 * 则按 * 走）\n"
               "       --list-links 可以看全部连接键类型（亚胺/酮烯胺/连氮/腙/乙烯基/"
               "酰胺/芳基单键…）\n"
               "\n"
               "  也可以自己用 * 标连接位：\n"
               "       python cof_builder.py -n \"[*][CH]c1cc([CH][*])cc([CH][*])c1\" "
               "-l \"[*][N]c1ccc([N][*])cc1\" -o LZU1b\n"
               "\n"
               "也可以完全不写星号、也不说连接键，纯按几何选位：\n"
               "       python cof_builder.py -n \"c1ncncn1\" "
               "-l \"c1ccc(C#Cc2ccc(C#Cc3ccccc3)cc2)cc1\" -o Triazine\n"
               "       （默认 3 连接；四连接加 --arms 4；先体检用 --inspect \"SMILES\"）\n"
               "\n"
               "注意：星号必须是独立的原子，写成 *、[*] 都行；\n"
               "      不要写成 [CH*] / [N*] —— 括号里的 * 会被当成非法记号。")
    ap.add_argument("-n", "--node", help="节点单体 SMILES。可以完全不写 *（脚本自动找 sp2 连接位）")
    ap.add_argument("-l", "--linker", help="第二个单体 SMILES：线形连接子（2 个 *）"
                                          "或与节点同连接数的另一个节点")
    ap.add_argument("-k", "--link", "--bond-type", dest="link", default="auto",
                    metavar="类型",
                    help="连接键类型（推荐用法：单体不写 *，只写分子本身）："
                         "imine/ketoenamine/enol/azine/hydrazone/vinylene/amide/suzuki，"
                         "auto=自动识别（默认）。用 --list-links 看全部")
    ap.add_argument("--list-links", action="store_true", help="列出支持的连接键类型")
    ap.add_argument("--arms", type=int, default=None,
                    help="节点要几个连接臂（3=蜂窝 hcb，4=四方 sql）。只在节点没写 * 时用")
    ap.add_argument("--arms2", type=int, default=None,
                    help="第二个单体几个连接臂（连接子=2）。默认 2")
    ap.add_argument("--fix-sites", action="store_true",
                    help="丢掉你自己写的 *，按分子本身重新自动选连接位")
    ap.add_argument("--allow-sp3", action="store_true",
                    help="连接位是 CH2/CH3 也照接：按平面三角摆放（H 放在层内）。"
                         "骨架能搭出来，键角是强行压平的，H 自己到 MS 里补")
    ap.add_argument("--inspect", metavar="SMILES", default=None,
                    help="只体检：列出每个原子的杂化/氢数/能否当连接位，不搭建")
    ap.add_argument("smiles", nargs="*", metavar="SMILES",
                    help="也可以按顺序直接给两个单体 SMILES（等于 -n 和 -l）")
    ap.add_argument("--preset", help="用内置组合（无需再写 --node/--linker）")
    ap.add_argument("--list-presets", action="store_true", help="列出所有内置组合")
    ap.add_argument("--topology", choices=["auto", "hcb", "sql"], default="auto",
                    help="拓扑：hcb=蜂窝(3 连接)，sql=四方(4 连接)，auto=自动判断（默认）")
    ap.add_argument("--bond", type=float, default=None,
                    help="新生成连接键的键长 (Å)。默认按化学环境自动推断")
    ap.add_argument("--c", type=float, default=3.60, help="层间距 (Å)，默认 3.60")
    ap.add_argument("--scale-a", type=float, default=1.0, help="a/b 缩放系数，默认 1.0")
    ap.add_argument("--tilt", type=float, default=None,
                    help="连接键相对节点连线的倾角 (度)。默认自动推算，一般不用给")
    ap.add_argument("--no-condense", action="store_true",
                    help="不自动做 -NH2 + -CHO 的亚胺缩合")
    ap.add_argument("--strict-planar", action="store_true",
                    help="强制单层严格共面：任一原子离面 > 0.05 Å 就拒绝写出")
    ap.add_argument("--allow-nonplanar", action="store_true",
                    help="骨架本身翘出平面也照写（四苯基 TTF、螺环这类节点天生是"
                         "螺旋桨形）。会提示褶皱幅度，结构仍然可用，只是层不平")
    ap.add_argument("--ideal-flat", action="store_true",
                    help="节点外围芳环绕 ipso–para 轴刚体转进平面（键长零变化）后再装配。"
                         "四芳基大环节点（卟啉 TAPP/TFPP/TCPP、酞菁）用它：meso-苯基在真实"
                         "构象里垂直大环，整分子投影压平会把环压坏，只有绕轴转环能无损落到平面")
    ap.add_argument("-o", "--out", default=None, help="输出文件名前缀，默认 cof")
    ap.add_argument("--check", metavar="CIF", default=None,
                    help="只体检一个已有的 CIF（成键/重叠/离面），不搭建")
    ap.add_argument("--no-plot", action="store_true", help="不画俯视图 PNG")
    ap.add_argument("--no-fix-xh", action="store_true",
                    help="不修 X–H 的键长/共面（默认会修：压平过程会把 O–H 压短到 0.90 Å，"
                         "酰胺 N–H 也会被甩出平面，都会让体检误报）")
    ap.add_argument("--no-sym", action="store_true",
                    help="不做层内对称化投影。默认会按晶格形状把结构投影到它该有的空间群"
                         "（a=b, γ=90° → P4；a=b, γ=120° → P6），使重原子严格对称")
    ap.add_argument("--nh", default="keep",
                    choices=["keep", "disorder", "metal", "bare"],
                    help="对称化时吡咯/酰胺型 X–H 怎么交代："
                         "keep=保留原位（默认，化学正确，可直接进 MACE）；"
                         "disorder=按 1/2 占位摊到 n 个 N 上（整个结构严格 P4/P6）；"
                         "metal=金属化（D4h，无占位问题）；bare=只删不补")
    ap.add_argument("--metal", default="Zn",
                    help="--nh metal 时插什么金属，默认 Zn")
    ap.add_argument("-q", "--quiet", action="store_true", help="少打印中间信息")

    # 什么都不给 → 交互式向导（比抛一大段帮助友好得多）
    argv = _wizard(presets, eng) if len(sys.argv) == 1 else None
    args = ap.parse_args(argv)

    # ---------------- 只体检一个分子 ----------------
    if args.inspect:
        eng.inspect_molecule(_clean_smiles(args.inspect))
        return

    # ---------------- 只体检一个已有的 CIF ----------------
    if args.check:
        from ase.io import read
        try:
            a = read(args.check)
        except Exception as exc:
            sys.exit(f"\n读不了 {args.check}：{exc}\n")
        print("=" * 64)
        print(f"结构体检：{args.check}")
        print("=" * 64)
        cp = a.cell.cellpar()
        print(f"晶胞: a={cp[0]:.3f} b={cp[1]:.3f} c={cp[2]:.3f} "
              f"γ={cp[5]:.2f}   原子 {len(a)}   {a.get_chemical_formula()}")
        print()
        fatal, why_bad = check_structure(a, verbose=True,
                                         allow_nonplanar=args.allow_nonplanar)
        print()
        if fatal:
            for m in why_bad:
                print(m)
            print("\n结论：这个结构不能用。")
            sys.exit(2)
        print("结论：没有发现硬伤，可以用。")
        return

    # 位置参数：python cof_builder.py "节点" "连接子"
    if args.smiles:
        if len(args.smiles) > 2:
            sys.exit("\n位置参数最多两个：节点 SMILES、第二个单体 SMILES。\n"
                     "（想调别的参数请用 -n / -l）\n")
        args.node = args.node or args.smiles[0]
        if len(args.smiles) == 2:
            args.linker = args.linker or args.smiles[1]

    # 去掉粘贴时可能带上的引号和空白
    args.node = _clean_smiles(args.node)
    args.linker = _clean_smiles(args.linker)

    # ---------------- 内置组合列表 ----------------
    if args.list_presets:
        print("内置组合（--preset 后面填左边的名字）：\n")
        for k in sorted(presets):
            note = presets[k].get("note", "")
            print(f"  {k:<16} {note if note else presets[k]['topology'] + ' 拓扑'}")
        return

    # ---------------- 连接键类型列表 ----------------
    if args.list_links:
        print("支持的连接键类型（-k 后面填左边的名字）：\n")
        for k, v in LINK_TYPES.items():
            print(f"  {k:<13} {v['label']}")
            print(f"  {'':<13} {v['formula']}    默认键长 {v['bond']:.2f} Å")
        print("\n用法（单体不用写 *，只写分子本身）：")
        print('  python cof_builder.py -n "O=Cc1cc(C=O)cc(C=O)c1" '
              '-l "Nc1cc(N)cc(N)c1" -k imine')
        print("不写 -k 时按 auto 处理：能认出互补官能团就自动套用，认不出才按几何选位。")
        return

    # ---------------- 取单体 ----------------
    if args.preset:
        if args.preset not in presets:
            sys.exit(f"\n没有内置组合 '{args.preset}'。用 --list-presets 看全部可用名字。\n")
        p = presets[args.preset]
        node = args.node or p.get("node")
        linker = args.linker or p.get("linker")
        if not args.quiet and p.get("note"):
            print(f"[组合] {args.preset}：{p['note']}")
    else:
        node, linker = args.node, args.linker

    if not (node and linker):
        print(HELP_HINT)
        raise SystemExit(2)

    # ---------------- 连接位：优先按"连接键类型"算，其次按几何挑 ----------------
    # 只给单体本身的 SMILES（如 "c1ncncn1"）也能搭：脚本自己找最对称的 sp2 位点。
    def _auto(smi, want, role):
        try:
            have = eng.count_stars(smi)
        except ValueError as e:
            sys.exit(f"\nSMILES 语法有问题：{e}\n"
                     f"请检查括号配对、芳香原子写法、* 的位置。\n")
        if have and not args.fix_sites:
            for r in eng.diagnose_stars(smi):
                if r["nh"] >= 2:
                    if args.allow_sp3:
                        print(f"[注意] {role}的 * 挂在 {r['sym']}{r['idx']} 上"
                              f"（{r['hyb']}，{r['nh']} 个 H）——--allow-sp3 已放行，"
                              f"按平面三角压平摆放")
                    else:
                        print(f"[注意] {role}的 * 挂在 {r['sym']}{r['idx']} 上，"
                              f"那个原子是 {r['hyb']}、挂着 {r['nh']} 个 H（CH2/CH3），"
                              f"四面体碳接不出平面层。\n"
                              f"       加 --allow-sp3 可以照接（H 自己到 MS 里补），"
                              f"加 --fix-sites 可让脚本重选位。")
            if want and have != want and not args.quiet:
                print(f"[提示] {role}写了 {have} 个 *，按 {have} 连接处理")
            return smi
        if have:
            print(f"[连接位] {role}原写了 {have} 个 *，--fix-sites 要求重选 {want} 个")
        else:
            print(f"[连接位] {role}没有 *，按分子本身自动找 {want} 个 sp2 位点")
        try:
            return eng.auto_star(smi, want, role, force=args.fix_sites)[0]
        except ValueError as e:
            sys.exit(f"\n{role} 无法确定连接位：\n  {e}\n")

    # 先把用户写的连接键类型规整好（写错的类型在这里就报错，别等到后面）
    try:
        link_key = normalize_link(args.link)
    except ValueError as e:
        sys.exit(f"\n{e}\n")

    def _star_count(smi):
        try:
            return eng.count_stars(smi)
        except ValueError as e:
            sys.exit(f"\nSMILES 语法有问题：{e}\n"
                     f"请检查括号配对、芳香原子写法、* 的位置。\n")

    has_star = _star_count(node) > 0 or _star_count(linker) > 0
    bond_by_link = None
    chem_note = ""

    if link_key == "auto" and not has_star:
        # 两个都是纯分子：先试试能不能从官能团认出连接化学
        detected = detect_linkage(node, linker, verbose=not args.quiet)
        if detected:
            link_key = detected

    chem_done = False
    if link_key != "auto":
        # ---------- 按连接键类型走：图变换 + 连接数由官能团个数定 ----------
        orig_node, orig_linker = node, linker
        if has_star:
            if not args.quiet:
                print("[连接键] SMILES 里已经有 *，先按 -k 的化学重算一遍连接位")
            try:
                node, linker = eng.strip_stars(node), eng.strip_stars(linker)
            except ValueError as e:
                sys.exit(f"\nSMILES 语法有问题：{e}\n")
        try:
            node, linker, nA, nB, bond_by_link, chem_note = apply_linkage(
                eng, link_key, node, linker,
                arms=args.arms, arms2=args.arms2, verbose=not args.quiet)
            node2, linker2 = node, linker
            mode = "vertex" if nB == nA else "edge"
            chem_done = True
        except ValueError as e:
            if not has_star:
                sys.exit(f"\n{e}\n")
            # 单体已经写成"缩合后"的样子了（没有 -CHO/-NH2 可反应）。
            # 这时候不要硬去重算化学，尊重用户写的 *，只借用 -k 给的默认键长。
            print("[连接键] 这些单体上已经没有可反应的官能团（多半是已经缩合后的写法），"
                  "保留你写的 *。")
            node, linker = orig_node, orig_linker
            bond_by_link = float(LINK_TYPES[link_key]["bond"])

    if not chem_done:
        # ---------- 没有连接键信息：沿用原来的几何选位 ----------
        want_A = args.arms or (4 if args.topology == "sql" else 3)
        node = _auto(node, want_A, "节点")
        linker = _auto(linker, args.arms2 or 2, "第二个单体")

        nA, nB = eng.count_stars(node), eng.count_stars(linker)
        if nB == 2 and nA >= 3 and args.arms2 is None and not args.quiet:
            print(f"[提示] 第二个单体按 2 连接（线形连接子）处理；"
                  f"若它其实是第二个节点（C3+C3 型），加 --arms2 {nA}")

        if nA not in (3, 4) and nB in (3, 4):
            if not args.quiet:
                print(f"[提示] 两个单体的连接位数是 {nA} / {nB}，已自动把 {nB} 连接的那个当节点")
            node, linker, nA, nB = linker, node, nB, nA

        mode = "vertex" if nB == nA else "edge"

        # ---------------- 缩合 ----------------
        node2, linker2, note = (node, linker, "")
        if not args.no_condense:
            node2, linker2, note = eng.condense_imine(node, linker)
            if note and not args.quiet:
                print(f"[提示] {note}")
            if note:
                chem_note = note

    if nA not in (3, 4):
        sys.exit(f"\n节点只有 {nA} 个连接位，无法搭建。\n"
                 f"目前支持 3 连接（hcb）和 4 连接（sql）节点：\n"
                 f"  3 连接例: *[CH]c1nc([CH]*)nc([CH]*)n1\n"
                 f"  4 连接例: *c1cc(*)c(*)cc1*\n"
                 f"  给的如果是单体本身（没有 *），可以用 -k 说明连接键类型，"
                 f"让脚本按官能团定连接数。\n")
    if mode == "edge" and nB != 2:
        sys.exit(f"\n第二个单体有 {nB} 个连接位，装不上。它只能是：\n"
                 f"  · 线形连接子（2 个 *）：*c1ccc(*)cc1\n"
                 f"  · 另一个节点（{nA} 个 *）：C3+C3 / C4+C4 型\n")

    topology = args.topology
    if topology == "auto":
        topology = {3: "hcb", 4: "sql"}[nA]
        if not args.quiet:
            print(f"[提示] 节点 {nA} 连接 → 自动选拓扑 '{topology}'")
    if eng.TOPOLOGIES[topology] != nA:
        sys.exit(f"\n拓扑和节点对不上：{topology} 需要 {eng.TOPOLOGIES[topology]} 连接节点，"
                 f"你的节点是 {nA} 连接。\n改成 --topology "
                 f"{ {3: 'hcb', 4: 'sql'}[nA] } 试试。\n")

    # ---------------- 连接键长 ----------------
    if args.bond is not None:
        bond, why = args.bond, "命令行 --bond 指定"
    elif bond_by_link is not None:
        bond, why = bond_by_link, f"按连接键类型 {LINK_TYPES[link_key]['label']} 取默认值"
        if not args.quiet:
            print(f"[提示] 连接键长 {bond:.2f} Å（{LINK_TYPES[link_key]['label']}）")
    elif args.preset and presets[args.preset].get("bond"):
        bond = float(presets[args.preset]["bond"])
        why = "内置组合自带"
    else:
        bond, why = guess_bond(node2, linker2, eng)
        if not args.quiet:
            print(f"[提示] 自动推断连接键长 {bond:.2f} Å（{why}）")

    note = chem_note

    # ---------------- 单体体检：臂夹角配不配得上这个网 ----------------
    if not args.quiet:
        print("=== 单体体检 ===")
    okA = arm_report(eng, node2, nA, "节点")
    okB = arm_report(eng, linker2, nB, "第二体" if mode == "vertex" else "连接子")
    if not (okA and okB):
        need = "三条 120°" if nA == 3 else "90° / 90° / 180°"
        print(f"   [注意] 上面标了偏差的单体，几条臂不是 {need}，硬塞进这个网会互相挤压；")
        print("          常见原因：① SMILES 里取代基的邻/间/对位写错；"
              "② 这个单体本来就是三角形或菱形构型。")
        print("          可以先用 --no-plot 看一眼晶胞和『最近原子间距』再决定。")

    # ---------------- 搭建 ----------------
    if not args.quiet:
        print("-" * 64)
        print(f"单体 A（节点）: {node2}")
        print(f"单体 B        : {linker2}")
        print("-" * 64)

    try:
        # 有些连接方式（聚酰亚胺）自带倾角要求，用户在命令行没写 --tilt 时用它
        _tilt_hint = LINK_TYPES.get(link_key, {}).get("tilt")
        eng.PAIR_SOFT = float(LINK_TYPES.get(link_key, {}).get("soft", 0.70))
        atoms, info = eng.build(node2, linker2, topology=topology, bond=bond,
                                c=args.c, scale_a=args.scale_a,
                                tilt=(args.tilt if args.tilt is not None else _tilt_hint),
                                condense=False, verbose=not args.quiet,
                                allow_sp3=args.allow_sp3,
                                ideal_flat=args.ideal_flat)
    except ValueError as e:
        # 引擎抛出带换行的消息时，说明它自己已经把原因和改法讲清楚了
        # （sp3 连接位、臂角不合、臂长配不上），不要再叠一句泛泛的提示。
        msg = str(e)
        hint = ("" if "\n" in msg else
                "\n常见原因：SMILES 语法错误、* 连在了不存在的位置、"
                "拓扑与节点连接数不匹配。")
        sys.exit(f"\n搭建失败：{msg}\n{hint}\n")
    except RuntimeError as e:
        sys.exit(f"\n3D 构象生成失败：{e}\n"
                 f"检查单体是否是合理的平面芳香体系，必要时换个共振式写法。\n")

    # ---------------- 输出 ----------------
    n_xh = 0 if args.no_fix_xh else fix_xh_geometry(atoms)
    if n_xh and not args.quiet:
        print(f"[提示] {n_xh} 个 X–H 按理想键长/共面归位"
              f"（压平过程会把 O–H 压短、把酰胺 N–H 甩出平面）")

    # ---------------- 层内对称化 ----------------
    # 建出来的结构其实已经很接近目标点群（重原子偏差 0.00~0.02 Å），但这点偏差
    # 足够让 spglib 判成 P1、让 MACE 弛豫顺着噪声歪掉。这里**主动**按晶格形状
    # 构造操作集合并投影 —— 而不是等 spglib 从结构里"找"出一个更低的群
    # （自由碱卟啉带两个 NH，spglib 永远只会给 P2）。
    lattice_sym = None if args.no_sym else _load_lattice_sym()
    sinfo = None
    if lattice_sym is not None:
        try:
            atoms, sinfo = lattice_sym.apply(
                atoms, nh=args.nh, metal=args.metal, verbose=not args.quiet)
            if not args.quiet:
                mark = "[OK]" if sinfo["applied"] else "[跳过]"
                print(f"[对称] {mark} {sinfo['note']}")
        except Exception as exc:                    # 对称化失败不该让建模整体失败
            print(f"[对称][跳过] {type(exc).__name__}: {exc}")

    name = args.out or "cof"
    if not name.lower().endswith(".cif"):
        cif = name + ".cif"
    else:
        cif = name
        name = name[:-4]
    xyz = name + "_layer.xyz"
    png = name + "_layer.png"

    plot_atoms = atoms.copy()                  # 未折回晶胞的副本，画图更完整
    # 把单层挪到晶胞中间（z = c/2）再折回。否则平面层的 z 恰好是 0，
    # 浮点误差为负的原子会被折到 z = c，读回来时 z 跨度显示成 3.6 Å（其实是同一层）。
    pos = atoms.get_positions()
    c_len = atoms.cell.cellpar()[2]
    if float(pos[:, 2].max() - pos[:, 2].min()) < 0.5 * c_len:
        pos[:, 2] -= pos[:, 2].mean() - 0.5 * c_len
        atoms.set_positions(pos)
    atoms.wrap()

    from ase.io import write

    # 先体检，再写盘。坏结构只留一份 .BAD.cif 备查，绝不生成正式 CIF ——
    # 否则你会在 Materials Studio / VESTA 里看到一个"原子糊成一团"的结构，
    # 还得自己猜是哪一步出的问题。
    dmin = eng.report(atoms, info)             # 晶胞/化学式/最近间距/孤立原子
    if not args.quiet:
        print("\n=== 成键体检 ===")
    fatal, why_bad = check_structure(atoms, verbose=not args.quiet,
                                     allow_nonplanar=args.allow_nonplanar)

    if fatal:
        bad = name + ".BAD.cif"
        write(bad, atoms)
        print("\n" + "!" * 64)
        for m in why_bad:
            print(m)
        print("!" * 64)
        print(f"\n这个结构不能用，已另存为 {bad} 供你排查；没有生成 {cif}。")
        print("最常见的原因，按出现频率排：")
        print("  1. 两个单体的臂根本接不上 —— 节点/连接子的 * 数量或位置不对。")
        print("     线形连接子的两个 * 必须反平行（对位取代）；邻位、间位取代，")
        print("     或顺式烯烃的两端夹角只有 60°，硬塞进 120° 的网就会折回来。")
        print("  2. 节点臂角不合 —— 上面『单体体检』里标了 [偏差 xx°] 的那一行就是它。")
        print("  3. 用 --scale-a 把晶胞压小了，长臂直接折回原点。")
        print("  4. 节点骨架天生是螺旋桨形（四芳基 TTF、四苯基乙烯、螺环中心），")
        print("     苯环绕单键扭出 40~50° 是它真实的构象，压不平。")
        print("     确认晶体结构本来就是褶皱的，就加 --allow-nonplanar 照写。")
        print()
        sys.exit(2)

    # ---- 平面度的硬闸（--strict-planar）----
    # 层状 COF 的一切后续处理（叠层、孔道、MACE 预优化）都假定单层是平的。
    # 有原子翘出去时，与其让你在 MS 里手工掰，不如在这里就拦下来：
    # 翘出平面的原因一定在单体上，改 SMILES 重搭比事后修补可靠得多。
    frame, d_all, d_heavy, dev, _soft = frame_offplane(atoms)
    if args.strict_planar and frame > 0.05:
        bad = name + ".BAD.cif"
        write(bad, atoms)
        sym = atoms.get_chemical_symbols()
        worst = sorted(range(len(atoms)), key=lambda i: -abs(dev[i]))[:8]
        print("\n" + "!" * 64)
        print(f"[致命] --strict-planar：单层不平，骨架最大离面 {frame:.3f} Å"
              f"（重原子 {d_heavy:.3f} Å）")
        print("       离面最多的原子：" + "、".join(
            f"{sym[i]}{i + 1} {dev[i]:+.2f}" for i in worst))
        print("!" * 64)
        print(f"\n已另存 {bad} 供你排查；没有生成 {cif}。")
        print("常见原因：")
        print("  · 单体里有 sp3 碳（写了 -CH2-、-CH3，或 [CH] 的 H 数不对）")
        print("  · 芳香环被生成成了船式/扭式构象 → 换个共振式重写 SMILES")
        print("  · 单体本身不是平面共轭体系（含四面体中心、螺环、大位阻取代基）")
        print()
        sys.exit(2)

    # 自定义 writer：ASE 的 CIF writer 会丢掉 occupancy，--nh disorder 时
    # 4 个 0.5 的 H 会变成 4 个整 H（化学式凭空多 2 个氢）。
    if lattice_sym is not None:
        lattice_sym.write_cif(cif, atoms)
    else:
        write(cif, atoms)
    write(xyz, plot_atoms)
    files = [cif, xyz]

    # ---- 附带一份"整个结构严格对称"的版本 ----
    # --nh keep（默认）保住化学上真实的两个吡咯 N–H，代价是 spglib 只能判 P2：
    # 那 2 个 H 确实只坐在 4 个 N 中的 2 个上。若需要"文件里 spglib 真判 P4/P6"
    # （投稿的结构表、拓扑分析），另存一份 H 按 1/2 占位摊平的版本，化学式不变。
    if lattice_sym is not None and sinfo and sinfo.get("applied") and \
            sinfo.get("nh_n") and args.nh == "keep":
        try:
            strict, s2 = lattice_sym.apply(atoms, nh="disorder", verbose=False)
            if s2.get("applied"):
                sp = f"{name}_strict.cif"
                lattice_sym.write_cif(sp, strict)
                files.append(sp)
                if not args.quiet:
                    print(f"[对称] 另存严格对称版 {sp}"
                          f"（{s2['nh_n']} 个吡咯 H 按 1/2 占位，化学式仍是 "
                          f"{_chem_formula(strict)}）")
        except Exception as exc:
            print(f"[对称][跳过 严格版] {type(exc).__name__}: {exc}")

    if not args.no_plot:
        try:
            # 画图函数原先住在 build_cof_image.py 里，那个文件还带着一套三嗪
            # 建模代码，清理时被整体挪进了 _trash_20261023/ —— 于是这句 import
            # 永远 ModuleNotFoundError，又被下面 except 吞成一行「跳过绘图」。
            from cof_view import plot_layer
            plot_layer(plot_atoms, out=png)
            files.append(png)
        except Exception as e:
            print(f"[跳过绘图] {type(e).__name__}: {e}")

    print_summary(atoms, info, name, topology, mode, bond, why, note, files)

    # ---------------- 体检提醒 ----------------
    if dmin < 0.95:
        print("[警告] 最近原子间距过小（结构有重叠），多半是：")
        print("  · 单体 SMILES 的氢数写错（多/少一个 H）→ 检查 [CH]/[N]/[nH]")
        print("  · 节点和连接子尺寸不匹配，孔被挤爆 → 换更长的连接子")
        print("  · 两个单体在共面层里本来就装不下 → 中间加炔撑/苯撑隔开")
        print()
    if info["sym_err"] > 5:
        print(f"[警告] 连接方向平均偏差 {info['sym_err']:.1f}°，单体可能不是平面重对称的，")
        print("       搭出来的孔会不是正六边形。检查单体的对称性与芳香性写法。\n")

    print(f"下一步：python run_opt.py  → 向导里选 '{cif}' 跑 MACE 预优化。")


if __name__ == "__main__":
    main()
