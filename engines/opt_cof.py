# -*- coding: utf-8 -*-
"""通用 COF 结构预优化（默认 QCOF 势）—— 任意 CIF/XYZ 进去，优化后的 CIF 出来。

流程照搬 run_opt.py（固定晶胞弛豫 → 放开晶胞 → a=b 规整），但把原来写死在
代码里的 read('4444.cif') / write('4444-Opt.cif') 换成了两种入口：

    ① 直接运行，一问一答（最省事，和 cof_builder.py 一样）
           python opt_cof.py

    ② 命令行带参数
           python opt_cof.py LongArm2_COF.cif
           python opt_cof.py my.cif -o my_opt.cif -p mace-off --model large
           python opt_cof.py my.cif --fix-c            # 只放开 a、b，c 固定
           python opt_cof.py my.cif --no-cell          # 连晶胞也不优化
           python opt_cof.py my.cif --keep-vacuum      # 输出保留真空（默认去掉）

    ③ 只删真空层，不做优化（子命令 strip）
           python opt_cof.py strip Triazine_QCOF.cif          # → *_novac.cif
           python opt_cof.py strip my.cif --gap 3.0 -o 紧凑.cif
           python opt_cof.py strip *.cif --dry-run            # 只看会改成什么样

真空层（重要）
--------------
MACE 类势是短程的，c 小于截断半径时每个原子都能看到自己在 z 方向的镜像，
周期镜像会被当成"原子贴太近"→ 巨大排斥 → 整层塌。实测 QCOF 在
c=3.60 Å / r_max=4.0 Å 的 AA 堆积上跑满 500 步，能量会从 -823 eV 飘到
+5.8e16 eV、c 压到 2.16 Å；即使是 vin2（69 原子）跑 40 步就已经到 +2757 eV。

所以：**真空是计算用的脚手架，不是交付物的一部分**。
    · 单层且 c < r_max + 1 → 计算前自动补 15 Å 真空（否则必炸）；
    · 计算完 → 输出自动还原成紧凑晶胞，c = 你输入里的层间距（如 vin2 的 3.448 Å）；
    · 想保留真空（比如还要接着跑）→ --keep-vacuum；
    · 多层（≥2 层/晶胞）→ 本来就不补真空，按原堆垛算。层间能不能算，看模型：
      MACE-MP-0（带 D3）和 MACE-OFF23 都能给出层间势能极小；QCOF 不能 ——
      它只训过带真空的单层，r_max=4 Å，**不能让它弛豫多层晶胞**（见下面
      「多层 + 单层势」）。只想弛豫 QCOF 多层的层内骨架，就加 --no-cell。

删真空（strip_vacuum_layer）
---------------------------
单层结构如果**输入本身**就带着 15 Å 真空，输出也会还原成紧凑晶胞
（c = 层厚 + --strip-gap，默认 3.45 Å）。也可以单独处理已有文件：
    python opt_cof.py strip 结构.cif [--gap 3.45] [--c 6.9] [--dry-run]
注意：紧凑晶胞的 c 小于截断，**不能直接拿去跑优化**（会算炸），它用于
交付、进 MS 堆垛、导图；要算就让它自己补真空。


势函数
------
    -p qcof      QCOF（Quantum COF，Szewczyk et al., arXiv:2604.21494）
                 COF 专用 MACE 势，48-0-4 架构，r_max=4 Å，只支持 C/H/N。
                 强度 RMSE 约 40 meV/Å，比通用 MACE-OFF 低一个量级，
                 但训练集是"带真空的单层碳氮纳米片"，对含 O/S 的 COF 不适用。
                 QCOF.model 找不到会自动找镜像下载（约 2.6 MB）。
    -p mace-mp   MACE-MP-0 + D3(BJ) 色散（89 元素，MPtrj 训练）。
                 算层间最可靠的选项：D3 提供长程吸引，缺了它层会飘开。
                 选它会**关闭**自动补真空（补了就等于把层间作用删掉）。
    -p mace-off  MACE-OFF23（small/medium/large），元素覆盖广，精度一般。
    -p file      自己给任意 MACE .model（--model-file 路径）。

换体系只需要重跑，不用再改代码。优化日志 / 轨迹也按体系命名
（``<输出名>_stage1.log``、``<输出名>_cell.traj`` …），
连跑好几个结构不会互相覆盖。

三个阶段
--------
    1. 固定晶胞，FIRE 弛豫内部坐标        —— 先消掉单体/SMILES 带来的局部畸变
    2. FrechetCellFilter(a,b,c 可动，角度锁死)，BFGS
    3. 若 a≠b 则取平均并再弛豫一轮        —— 蜂窝/四方网必须 a=b

相对 run_opt.py 修掉的两个坑
---------------------------
1. 阶段 3 原先把晶胞角硬写成 ``[90, 90, 120]``。这对六方（hcb）没问题，但对
   斜方胞是致命的——PyreneBBT_COF 这类四连接菱形节点实测 γ=62.35°，强行改成
   120° 会直接把整层拧散。现在原样沿用优化后的 α/β/γ。
2. 原写 ``device="cuda"``，没有 GPU 的机器一启动就崩。现在默认 auto 探测。
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

def _env_hint():
    """列一下本机可能装了 ase/torch 的虚拟环境，省得用户自己找路径。"""
    import glob as _g
    pats = [str(Path.home() / "WorkBuddy" / "*" / env / "Scripts" / "python.exe")
            for env in ("maceenv", "cofenv")]
    pats += [str(Path.home() / "WorkBuddy" / "*" / "cofenv" / "python.exe")]
    found, seen = [], set()
    for p in pats:
        for f in _g.glob(p):
            k = f.lower()
            if k not in seen:
                seen.add(k)
                found.append(f)
    return found


def _die_missing_dep(exc):
    """import 阶段缺包就直说，别甩一串 traceback。

    这一条是「为什么敲了 python 却一个提示都没有」的答案：缺 numpy/ase 时
    进程在第 84 行就退了，根本走不到向导的输入循环，看起来就像"不能输入"。
    """
    mod = getattr(exc, "name", "?") or "?"
    print("\n" + "=" * 66)
    print(" [启动失败] 这个 Python 里没有 %s —— 用错解释器了。" % mod)
    print("            %s" % sys.executable)
    print("            脚本在 import 阶段就退出，所以连输入提示都不会出现。")
    print(" 请用装了 ase / torch / mace 的环境。本机已验证可用的：")
    envs = _env_hint()
    if envs:
        for e in envs[:4]:
            print('    "%s" opt_cof.py' % e)
    else:
        print("    <你的虚拟环境>\\Scripts\\python.exe opt_cof.py")
    print(" 或者先激活环境（激活后 python 就指向它了）：")
    print("    <虚拟环境>\\Scripts\\activate")
    print("    python opt_cof.py")
    print(" 不想每次敲路径，也可以做个 run_opt.bat，内容一行：")
    print('    "<虚拟环境>\\Scripts\\python.exe" "%~dp0opt_cof.py" %*')
    print("=" * 66 + "\n")
    sys.exit(1)


try:
    import numpy as np
    from ase.io import read, write
    from ase.optimize import FIRE, BFGS
    from ase.filters import FrechetCellFilter
    from ase.cell import Cell
    from ase.geometry import cellpar_to_cell
except ModuleNotFoundError as _exc:
    _die_missing_dep(_exc)

# 成键判据（Å）：只用于体检，不影响优化
CUT = {("C", "C"): 1.80, ("C", "N"): 1.72, ("C", "O"): 1.66,
       ("C", "H"): 1.25, ("N", "N"): 1.70, ("N", "O"): 1.66,
       ("N", "H"): 1.20, ("O", "O"): 1.70, ("O", "H"): 1.15}


def _cut(e1, e2):
    return CUT.get(tuple(sorted((e1, e2))), 0.0)


def layer_normal(atoms):
    """层法向（原子位置 PCA 的最小方差方向，朝 +z）。

    不能假设"c 轴就是层法向"：MS 导出的晶胞经常把 c 写成斜的，
    vin2.cif 就是 α=83.73°、β=80.78°，c 偏离法向 14.87°。
    """
    P = atoms.get_positions()
    if len(P) < 3:
        return np.array([0.0, 0.0, 1.0])
    pc = P - P.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 1.0])
    nrm = float(np.linalg.norm(vt[2]))
    if nrm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    n = vt[2] / nrm
    return n if n[2] >= 0 else -n


def interlayer_spacing(atoms, n=None):
    """真正的层间距：晶胞沿层法向的周期长度 |c·n|（Å）。

    直接拿 c 当层间距只在 c 垂直层平面时才对。vin2.cif 里 c=3.5673，
    但 α=83.73°、β=80.78°，真值是 3.448 Å，差了 3%。
    """
    n = layer_normal(atoms) if n is None else n
    return abs(float(np.dot(np.asarray(atoms.cell)[2], n)))


def layer_count(atoms, n=None, gap=1.5):
    """晶胞里有几层：沿层法向排好后，数大于 gap 的空隙个数。

    单层 + 真空 → 1；AA 双层晶体（c=6.9）→ 2。
    分不清层数的话，"层间距"会把 2 层的周期误报成层间距离（差一倍）。
    """
    n = layer_normal(atoms) if n is None else n
    d = interlayer_spacing(atoms, n)
    if d <= 1e-6:
        return 1
    h = atoms.get_positions() @ n
    hs = np.sort((h - h.min()) % d)
    gaps = np.diff(np.concatenate([hs, [hs[0] + d]]))
    return max(1, int((gaps > gap).sum()))


def layer_thickness(atoms, n=None):
    """层厚：沿层法向覆盖所有原子的最短弧长（Å）。

    直接取 z.max()-z.min() 在周期边界上是错的——平面层的原子会被折到 z=0 和
    z=c 两侧，看起来像有 c 那么厚。最小圆弧才对。
    这里改成在层法向上做：投影 → 对法向周期取模 → 找最大空隙。
    """
    n = layer_normal(atoms) if n is None else n
    h = atoms.get_positions() @ n
    if len(h) == 0:
        return 0.0
    d = interlayer_spacing(atoms, n)
    if d <= 1e-6:
        return float(h.max() - h.min())
    h = (h - h.min()) % d
    hs = np.sort(h)
    gaps = np.diff(np.concatenate([hs, [hs[0] + d]]))
    return float(d - gaps.max())


def strip_vacuum_layer(atoms, gap=3.45, c_target=None):
    """删掉单层的真空层：把 c 压到「层厚 + gap」。

    只对 **单层 + 真空** 有意义。多层晶体（AA 双层 c=6.9）里那段空隙就是
    层间距本身，压掉等于把两层合并成一层，所以直接拒绝、原样返回。

    gap 默认 3.45 Å = 石墨/COF 常见 AA 堆垛距离：新晶胞就等于「把这一层按
    AA 直接摞起来」，进 MS 看是紧凑的，不是空荡荡一个大盒子。
    注意：压缩后不能再拿 QCOF 算（c < r_max 会让镜像成键），这只是一份
    "紧凑视图"，不是能继续做周期优化的结构。
    """
    n = layer_normal(atoms)
    nl = layer_count(atoms, n)
    th = layer_thickness(atoms, n)
    d_old = interlayer_spacing(atoms, n)
    info = {"n": n, "nl": nl, "th": th, "d_old": d_old, "d_new": d_old,
            "removed": 0.0}

    if c_target is None:
        if nl >= 2:
            return None, info                      # 多层：本来就没有真空
        c_target = th + gap
    else:
        c_target = float(c_target)
    if c_target <= th + 1e-3:
        raise ValueError(f"目标 c={c_target:.3f} Å 比层厚 {th:.3f} Å 还小")
    info["d_new"] = c_target
    info["removed"] = max(0.0, d_old - c_target)

    new = atoms.copy()
    C = np.array(new.cell)
    a, b, c0 = C[0], C[1], C[2]
    c_hat = c0 / np.linalg.norm(c0)
    # PCA 法向和 c 轴总会差个零点几度（vin2 就只差 0.3°），所以按角度判，
    # 不按 |a·n| 的绝对值判——否则 27 Å 的边长乘 0.005 就超过阈值了。
    ang = float(np.degrees(np.arccos(min(1.0, abs(float(c_hat @ n))))))
    ab_perp = max(abs(float(a @ n)), abs(float(b @ n))) < 0.02 * max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if ang < 2.0:
        new_c = c_hat * c_target                   # c 本来就垂直层平面，改长度即可
    elif ab_perp:
        new_c = n * c_target                       # c 是斜的但 a、b 在面内 → 扶正
        info["straighten"] = True
    else:                                          # 真·斜晶胞：沿原方向缩放
        new_c = c_hat * (c_target / max(np.cos(np.radians(ang)), 0.5))
        info["skew"] = True
    new.set_cell(np.array([a, b, new_c]), scale_atoms=False)

    P = new.get_positions()
    h = P @ n
    P = P - np.outer(np.full(len(P), float(h.min())), n)      # 最低点挪到 0
    h = P @ n
    P = P + n * (0.5 * (c_target - (float(h.max()) - float(h.min()))))  # 居中
    new.set_positions(P)
    new.wrap()
    return new, info



def bond_stats(atoms):
    """按距离判据统计关键键长区间，返回 (文本, 短接触列表)。"""
    s = atoms.get_chemical_symbols()
    n = len(atoms)
    D = atoms.get_all_distances(mic=True)
    nb = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] < _cut(s[i], s[j]):
                nb[i].append(j)
                nb[j].append(i)

    def rng(pair, lo, hi):
        v = [D[i][j] for i in range(n) for j in range(i + 1, n)
             if {s[i], s[j]} == set(pair) and lo <= D[i][j] < hi]
        if not v:
            return "  -   "
        return "%d根 %.3f~%.3f" % (len(v), min(v), max(v))

    txt = (f"C≡C {rng(('C', 'C'), 1.10, 1.26)} | "
           f"C-C(sp/sp2) {rng(('C', 'C'), 1.26, 1.62)} | "
           f"C-N {rng(('C', 'N'), 1.20, 1.50)} | "
           f"C=O {rng(('C', 'O'), 1.15, 1.30)}")
    short = [(round(float(D[i][j]), 3), s[i], s[j])
             for i in range(n) for j in range(i + 1, n) if D[i][j] < 0.85]
    return txt, short, nb


def report(atoms, tag, vacuum=None):
    p = atoms.cell.cellpar()
    txt, short, nb = bond_stats(atoms)
    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, 99.0)
    dmin = float(D.min())
    n = layer_normal(atoms)
    th = layer_thickness(atoms, n)
    d = interlayer_spacing(atoms, n)
    nl = layer_count(atoms, n)
    print(f"\n===== {tag} =====")
    print(f"  晶胞    a={p[0]:.3f}  b={p[1]:.3f}  c={p[2]:.3f}  "
          f"α={p[3]:.2f} β={p[4]:.2f} γ={p[5]:.2f}")
    print(f"  化学式  {atoms.get_chemical_formula()}   {len(atoms)} 原子")
    if nl >= 2:
        # 多层晶胞：法向周期是"n 层 + (n-1) 个层间隙"，层间距要除以 n
        print(f"  层数    {nl} 层/晶胞   层间距 {d / nl:.3f} Å（沿层法向）   "
              f"最近原子间距 {dmin:.3f} Å")
    elif vacuum:
        # 补过真空之后 c 已经不代表层间距了，必须写清楚，否则看着像层间被拉开了
        print(f"  层厚    {th:.3f} Å   层间距 {d:.3f} Å"
              f"（含 {vacuum:.1f} Å 真空）   最近原子间距 {dmin:.3f} Å")
    else:
        print(f"  层厚    {th:.3f} Å   层间距 {d:.3f} Å（沿层法向）   "
              f"最近原子间距 {dmin:.3f} Å")
    print(f"  键长    {txt}")
    if short:
        print(f"  [警告] 存在 <0.85 Å 的重叠: {short[:6]}")
    try:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        fmax = float(np.sqrt((f ** 2).sum(axis=1)).max())
        print(f"  能量    {e:.4f} eV  ({e / len(atoms):.4f} eV/atom)   最大力 {fmax:.4f} eV/Å")
    except Exception:
        pass
    return p


# ------------------------------------------------------------ 本地模型缓存
# MACE 默认从 raw.githubusercontent.com 拉模型，这个域名在国内基本连不上，
# 而 mace_off() 的下载超时是 120 秒 —— 选错一个尺寸就要干等两分钟然后崩。
# 所以加载前先扫一遍本地缓存：有就用，没有就当场说清楚，别让 uflib 去试。
OFF23 = ("small", "medium", "large")


def _cache_dir():
    """等价于 mace.tools.utils.get_cache_dir()。

    故意不 import mace —— 那会连带拉起 torch，光启动就几秒，
    而这里只是想看看目录里有哪些文件。
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(str(Path.home()), ".cache")
    return Path(base) / "mace"


def _local_models():
    """本地已缓存的 MACE-OFF23 模型 -> {尺寸: 路径}。小于 1 MB 视为下载残缺。"""
    d, got = _cache_dir(), {}
    for name in OFF23:
        f = d / ("MACE-OFF23_%s.model" % name)
        try:
            if f.is_file() and f.stat().st_size > 1_000_000:
                got[name] = f
        except OSError:
            pass
    return got


def model_url(name):
    """走 ghfast 镜像的模型直链（官方源国内不可达时的替代）。"""
    return ("https://ghfast.top/https://raw.githubusercontent.com/ACEsuit/"
            "mace-off/main/mace_off23/MACE-OFF23_%s.model" % name)


def _model_hint(name):
    """打印一条可以直接复制粘贴的下载命令。"""
    dst = _cache_dir() / ("MACE-OFF23_%s.model" % name)
    print("       网络不通时用镜像手动补（整行复制到命令行）：")
    print('       curl -L -o "%s" "%s"' % (dst, model_url(name)))


# ------------------------------------------------------------ QCOF（COF 专用势）
# QCOF: Quantum COF，Szewczyk et al., arXiv:2604.21494
#   基于 MACE 架构、48-0-4（num_channels=48, max_L=0, r_max=4 Å）在 36k 个
#   PBE+D3 非平衡碳氮纳米片构象上从零训练。力 RMSE ~40 meV/Å，比 MACE-OFF24
#   的 ~1000 meV/Å 低一个量级；140k 原子都能跑。
#   代价：只认识 C / H / N。带 O（β-酮烯胺）、S、F 的 COF 直接不能用。
#   仓库: https://github.com/Alo-koder/QCOF （MIT）
QCOF_FILE = "QCOF.model"
QCOF_RAW = ("https://raw.githubusercontent.com/Alo-koder/QCOF/"
            "master/model-weights/QCOF.model")
QCOF_MIRRORS = (
    "https://ghfast.top/" + QCOF_RAW,
    "https://gh-proxy.com/" + QCOF_RAW,
    QCOF_RAW,
)


def _qcof_search_dirs():
    """可能放 QCOF.model 的地方，按优先级。"""
    here = Path(__file__).resolve().parent
    dirs = [Path.cwd(), here, _cache_dir()]
    env = os.environ.get("QCOF_DIR") or os.environ.get("QCOF_MODEL")
    if env:
        p = Path(env)
        dirs.insert(0, p.parent if p.suffix else p)
    out, seen = [], set()
    for d in dirs:
        try:
            k = str(d.resolve()).lower()
        except OSError:
            k = str(d).lower()
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def find_qcof(explicit=None):
    """找 QCOF.model，返回 Path 或 None。小于 500 KB 视为下载残缺。"""
    cands = []
    if explicit:
        p = Path(explicit)
        cands += [p] if p.suffix else [p / QCOF_FILE, p]
    for d in _qcof_search_dirs():
        cands.append(d / QCOF_FILE)
    for f in cands:
        try:
            if f.is_file() and f.stat().st_size > 500_000:
                return f
        except OSError:
            pass
    return None


def download_qcof(dest=None, verbose=True):
    """从镜像拉 QCOF.model。成功返回 Path，失败返回 None。

    先写 .part 再改名 —— 中途断线不会留下一个"看起来存在但加载就崩"的半截文件。
    """
    import shutil
    import urllib.request

    dest = Path(dest or (_cache_dir() / QCOF_FILE))
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    tmp = dest.with_name(dest.name + ".part")
    for url in QCOF_MIRRORS:
        if verbose:
            print(f"       尝试 {url.split('/')[2]} …", end="", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            size = tmp.stat().st_size
            if size < 500_000:
                raise IOError(f"只下到 {size} 字节，不完整")
            tmp.replace(dest)
            if verbose:
                print(f" 好了（{size / 1048576:.1f} MB）")
            return dest
        except Exception as exc:
            if verbose:
                print(f" 失败（{type(exc).__name__}）")
            try:
                tmp.unlink()
            except OSError:
                pass
    return None


def _qcof_hint():
    dst = _cache_dir() / QCOF_FILE
    print("       手动下载（任选一行，整行复制）：")
    print('       curl -L -o "%s" "%s"' % (dst, QCOF_MIRRORS[0]))
    print("       或者从 https://github.com/Alo-koder/QCOF/tree/master/model-weights 下")
    print("       下来后放到当前目录，改名为 QCOF.model 即可。")


def model_info(calc):
    """从已加载的 MACE 模型里读元素表和截断半径（拿不到就返回空）。

    mace_mp(dispersion=True) 返回的是 SumCalculator（MACE + D3），
    真正的模型藏在 calc.mixer.calcs[0] 里，得先拆开。
    """
    zmap = {1: "H", 2: "He", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
            14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
    try:
        c = calc
        for _ in range(3):                     # 拆 SumCalculator 的壳
            if hasattr(c, "mixer") and getattr(c.mixer, "calcs", None):
                c = c.mixer.calcs[0]
            else:
                break
        m = c.models[0]
        zs = {zmap.get(int(z), str(int(z))) for z in m.atomic_numbers}
        r = float(m.r_max)
        return zs, r
    except Exception:
        return set(), 0.0



_EXT_OK = (".cif", ".xyz", ".vasp", ".extxyz", ".poscar", ".pdb")


def _ensure_ext(p, default=".cif"):
    """输出名缺扩展名就补上。

    ASE 的 write() 靠后缀猜格式，``write("44444", atoms)`` 直接抛
    ``UnknownFileTypeError: Could not guess file type`` —— 而且是在三个阶段
    全部跑完之后才抛，几分钟白跑。所以在这里先补齐。

    注意不能用 os.path.splitext 判断：``"2.2"`` 会被拆成 ("2", ".2")，
    补完就变成 "2.cif"，把名字吃掉一截。只认末尾是不是已知后缀。
    """
    if not p:
        return p
    if p.lower().endswith(_EXT_OK):
        return p
    return p.rstrip(".") + default


def _clean_path(s):
    """去掉粘贴路径时带上的引号和首尾空白。

    注意只剥两端的引号，不动中间的空格——Windows 路径里带空格是常事，
    像 ``"C:\\My Docs\\a.cif"`` 这样粘进来要能直接用。
    """
    if not s:
        return s
    for q in ('"', "'", "“", "”", "‘", "’", "「", "」", "《", "》"):
        s = s.strip().strip(q)
    return s.strip()


_STRUCT_EXT = (".cif", ".xyz", ".extxyz", ".vasp", ".poscar")


def _list_structures(limit=20):
    """列出当前目录下可读的结构文件。

    排序：CIF 优先 → 没优化过的优先 → 字母序。
    目录里常常堆着一堆 ``xxx-Opt.cif``，那是上一轮的产物，不该占着前排。
    """
    found, seen = [], set()
    for ext in _STRUCT_EXT:
        for pat in ("*" + ext, "*" + ext.upper()):
            for f in glob.glob(pat):
                k = f.lower()
                if k in seen:
                    continue
                seen.add(k)
                found.append(f)

    def key(f):
        low = f.lower()
        stem = os.path.splitext(low)[0]
        done = stem.endswith("-opt") or stem.endswith("_opt") or "opt" in stem.split("-")
        return (not low.endswith(".cif"), done, low)

    found.sort(key=key)
    return found[:limit], max(0, len(found) - limit)


HELP_HINT = """
命令行用法（不想进向导时）:

    python opt_cof.py 结构.cif                       # 默认 QCOF 势
    python opt_cof.py 结构.cif -o 输出.cif -p mace-off --model large
    python opt_cof.py 结构.cif --fix-c               # 只放开 a、b，c 固定
    python opt_cof.py 结构.cif --no-cell             # 完全不动晶胞，只弛豫原子
    python opt_cof.py 结构.cif --vacuum 15           # 补 15 Å 真空按单层优化
    python opt_cof.py 结构.cif -p mace-mp            # 带 D3 色散，算层间作用
    python opt_cof.py 结构.cif --allow-download      # 缺模型时允许联网下
    python opt_cof.py 结构.cif --keep-vacuum         # 输出保留真空（默认会去掉）
    python opt_cof.py 结构.cif --show                # 优化完弹窗看俯视 + 侧视
    python opt_cof.py strip 结构.cif                 # 只删真空，不优化
    python opt_cof.py -h                             # 看全部参数

真空层: 单层 c 太小时只在计算中临时补，输出会自动去掉（还原成你的层间距）。

势函数（-p）:
    qcof      QCOF，COF 专用 MACE 势（arXiv:2604.21494），只支持 C/H/N，
              单层几何最准，但完全不会层间作用
    mace-mp   MACE-MP-0 + D3(BJ) 色散（89 元素）—— 要算层间就用这个
    mace-off  通用 MACE-OFF23，small / medium / large
    file      自己的 .model，配 --model-file 路径

产物: <输出名>.cif、<输出名>_layer.png，以及
      <输出名>_stage1.log / _cell.traj / _stage2.log / _stage3.log
"""


_EOF_SEEN = [False]


def _ask(prompt, default=""):
    """统一的输入入口：Ctrl+C / 管道结束都当成「用默认值」。

    但「输入流已经结束」必须说出来。老版本静默返回 default，于是在没有终端
    的环境里（IDE 运行面板、输出重定向、被别的程序拉起来）整个向导会一路
    默认值冲过去 —— 用户看到的是"提示一闪而过、打的字一个都没进去"。
    """
    try:
        v = input(prompt).strip()
    except KeyboardInterrupt:
        print()
        return default
    except EOFError:
        if not _EOF_SEEN[0]:
            _EOF_SEEN[0] = True
            print("\n[注意] 标准输入已经结束（当前没有可交互的终端），"
                  "后面全部取默认值。")
            print("       要手动指定请改用命令行参数，例如：")
            print("       python opt_cof.py 结构.cif -p qcof --show")
        else:
            print()
        return default
    return v or default


def _wizard(ask_opts=True):
    """不带命令行参数时的交互式向导，返回等价的 argv 列表。

    和 cof_builder.py 用的是同一套套路：先列出目录里现成的结构文件让你选
    编号，选 0 才手打路径；粘贴进来的引号会自动清掉。
    """
    print("=" * 64)
    print(" COF 结构预优化（交互式）—— QCOF / MACE 势")
    print("=" * 64)

    # stdin 不是终端时，input() 要么立刻 EOF、要么按键进不来。提前说清楚，
    # 免得用户对着"提示一闪而过"猜半天。
    try:
        _tty = sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        _tty = False
    if not _tty:
        print("\n[提示] 标准输入不是交互终端（IDE 运行面板 / 输出被重定向 /")
        print("       向导被别的程序拉起来），在这里打字可能没反应。")
        print("       要么在真正的 cmd / PowerShell 窗口里跑，要么直接用参数：")
        print("         python opt_cof.py 结构.cif -p qcof --show")

    files, more = _list_structures()
    if files:
        print("\n当前目录下的结构文件：")
        for i, f in enumerate(files, 1):
            try:
                size = "%7.1f KB" % (os.path.getsize(f) / 1024.0)
            except OSError:
                size = " " * 10
            print("  %2d  %-36s %s" % (i, f, size))
        if more:
            print("      …… 还有 %d 个，请选 0 手动输路径" % more)
        print("   0  手动输入路径")
        print("   h  看完整命令行用法")
        tip = "\n请输入编号（直接回车 = 1）: "
    else:
        print("\n  当前目录没找到 .cif/.xyz 文件，直接输路径吧。")
        print("   h  看完整命令行用法")
        tip = "\n路径（直接回车 = 退出）: "

    while True:
        c = _ask(tip)
        if c.lower() in ("h", "help", "?", "？"):
            print(HELP_HINT)
            raise SystemExit(0)

        if files and c == "":
            pick = files[0]
            break
        if files and c.isdigit() and 1 <= int(c) <= len(files):
            pick = files[int(c) - 1]
            break
        if c == "0" or not files:
            pick = _clean_path(_ask("  结构文件路径: "))
            if pick:
                break
            if not files and c == "":
                raise SystemExit(0)
            print("    不能为空，请再输一次。")
            continue
        print("  没有这个编号，请重新输入（0 = 手动输路径，h = 看用法）。")

    pick = _clean_path(pick)
    # 顺手报一下选中的是什么，免得编号看错行了
    try:
        _a = read(pick)
        print(f"  选中: {pick}   {len(_a)} 原子   {_a.get_chemical_formula()}")
    except Exception:
        print(f"  选中: {pick}")

    stem = os.path.splitext(os.path.basename(pick))[0]
    dflt_out = stem + "_opt.cif"
    out = _clean_path(_ask("  输出 CIF [直接回车 = %s]: " % dflt_out, dflt_out))
    out = _ensure_ext(out)

    argv = [pick, "-o", out]
    if not ask_opts:
        print()
        return argv

    qc = find_qcof()
    print("  势函数  1 QCOF（COF 专用 MACE，只支持 C/H/N，单层最准）"
          + ("  ← 本地已找到" if qc else "  ← 本地没有，要联网下 2.6 MB"))
    print("          2 MACE-MP-0 + D3（通用材料势，算层间最稳）")
    print("          3 MACE-OFF23（通用有机势，元素覆盖广，精度一般）")
    ps = _ask("  [直接回车 = 1 QCOF]: ", "1").lower()
    if ps in ("2", "mp", "mace-mp", "macemp"):
        m = _ask("  MACE-MP 模型  1 small / 2 medium / 3 large "
                 "[直接回车 = 2 medium]: ", "2").lower()
        model = {"1": "small", "s": "small", "small": "small",
                 "3": "large", "l": "large", "large": "large"}.get(m, "medium")
        argv += ["-p", "mace-mp", "--model", model]
    elif ps in ("3", "m", "mace", "mace-off", "off"):
        have = _local_models()
        if have:
            print("  本地已缓存: " + "、".join(n for n in OFF23 if n in have)
                  + "（选没缓存的尺寸要联网下载）")
        m = _ask("  MACE 模型  1 small / 2 medium / 3 large "
                 "[直接回车 = 2 medium]: ", "2").lower()
        model = {"1": "small", "s": "small", "small": "small",
                 "3": "large", "l": "large", "large": "large"}.get(m, "medium")
        argv += ["-p", "mace-off", "--model", model]
    else:
        argv += ["-p", "qcof"]
        if not qc:
            argv.append("--allow-download")

    d = _ask("  计算设备  auto / cuda / cpu [直接回车 = auto]: ", "auto").lower()
    device = d if d in ("auto", "cuda", "cpu") else "auto"
    p = _ask("  优化后画图并开窗看一眼吗？[Y/n]: ", "y").lower()

    argv += ["--device", device]
    if p in ("n", "no", "否", "不"):
        argv.append("--no-plot")
    else:
        argv.append("--show")
    print("  说明: 如果 c 比模型的截断半径还小（AA 堆积的 3.6 Å 就是），")
    print("        会自动补 15 Å 真空再优化，免得周期镜像被当成成键算炸。")
    print()
    return argv


def make_calculator(args, atoms):
    """按 --potential 建 ASE 计算器，返回 (calc, 名字)。

    qcof      QCOF（COF 专用，只认 C/H/N）
    mace-off  通用 MACE-OFF23（small/medium/large）
    file      任意 MACE .model
    """
    import torch
    from mace.calculators import MACECalculator

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[提示] 没有可用 CUDA，退回 CPU")
        device = "cpu"
    print(f"设备: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    if args.potential == "qcof":
        path = find_qcof(args.model_file)
        if path is None:
            print(f"\n[提示] 没找到 {QCOF_FILE}（约 2.6 MB）")
            print(f"       找过这些位置：{[str(d) for d in _qcof_search_dirs()]}")
            if args.allow_download:
                path = download_qcof()
            else:
                print("       加 --allow-download 可以让它自动下；或者手动：")
            if path is None:
                _qcof_hint()
                sys.exit(1)
        print(f"模型: QCOF（COF 专用 MACE，48-0-4，r_max=4 Å）")
        print(f"      {path}  dtype={args.dtype}")
        try:
            calc = MACECalculator(model_paths=[str(path)], device=device,
                                  default_dtype=args.dtype)
        except Exception as exc:
            print(f"\n[失败] QCOF 加载不了：{type(exc).__name__}: {exc}")
            print("       QCOF 是在 mace-torch 0.3.13 下训练的。本机版本：")
            try:
                import importlib.metadata as _md
                print(f"       mace-torch {_md.version('mace-torch')}")
            except Exception:
                pass
            print("       实在不行先退回通用势：-p mace-off --model large")
            sys.exit(1)
        label = "QCOF"

    elif args.potential == "file":
        if not args.model_file:
            sys.exit("\n--potential file 需要同时给 --model-file 路径.model\n")
        p = Path(args.model_file)
        if not p.is_file():
            sys.exit(f"\n找不到模型文件: {p}\n")
        print(f"模型: 自定义 {p.name}  dtype={args.dtype}")
        calc = MACECalculator(model_paths=[str(p)], device=device,
                              default_dtype=args.dtype)
        label = p.stem

    elif args.potential == "mace-mp":
        # MACE-MP-0：MPtrj（1.6M 无机晶体）上训的通用材料势，89 元素。
        # 自带 D3(BJ) 色散校正 —— 长程吸引齐全，算层间最可靠。
        from mace.calculators import mace_mp

        disp = not args.no_dispersion
        print(f"模型: MACE-MP-0-{args.model}"
              + (" + D3(BJ) 色散校正（能算层间）" if disp else "（无显式色散，层间作用很弱）"))
        try:
            calc = mace_mp(model=args.model, device=device,
                           default_dtype=args.dtype,
                           dispersion=disp, damping="bj", dispersion_xc="pbe")
        except Exception as exc:
            print(f"\n[失败] MACE-MP-0-{args.model} 加载失败：{type(exc).__name__}: {exc}")
            if disp:
                print("       可能是 torch-dftd 没装： pip install torch-dftd")
                print("       或者先不加色散试： --no-dispersion")
            print("       模型从 github.com/ACEsuit/mace-foundations 下载，"
                  "国内可能超时，建议挂代理。")
            sys.exit(1)
        label = "MACE-MP-0-%s%s" % (args.model, "+D3(BJ)" if disp else "")

    elif args.potential == "mace-off":
        from mace.calculators import mace_off

        # 模型缓存预检：本地没有的就别让 MACE 去撞 120 秒的下载超时
        local = _local_models()
        if args.model not in local:
            have = "、".join(n for n in OFF23 if n in local) or "一个都没有"
            print(f"\n[提示] 本地没有 MACE-OFF23-{args.model}（已有: {have}）")
            print(f"       缓存目录: {_cache_dir()}")
            if local and not args.allow_download:
                fb = "medium" if "medium" in local else next(iter(local))
                print(f"       官方源 raw.githubusercontent.com 在国内经常连不上，"
                      f"直接改用本地已有的 {fb}。")
                print(f"       想真正下载请加 --allow-download，或手动补：")
                _model_hint(args.model)
                args.model = fb
            else:
                print("       尝试下载（失败会自动回退到本地已有的版本）…")
        print(f"模型: MACE-OFF23-{args.model}  dtype={args.dtype}")
        try:
            calc = mace_off(model=args.model, default_dtype=args.dtype, device=device)
        except Exception as exc:
            local = _local_models()
            if not local:
                print(f"\n[失败] MACE-OFF23-{args.model} 加载不了，"
                      f"本地也没有任何可用模型。")
                print(f"       原始错误: {type(exc).__name__}: {exc}")
                _model_hint("medium")
                sys.exit(1)
            fb = "medium" if "medium" in local else next(iter(local))
            print(f"\n[警告] MACE-OFF23-{args.model} 加载失败"
                  f"（{type(exc).__name__}），改用本地已有的 {fb}。")
            args.model = fb
            calc = mace_off(model=fb, default_dtype=args.dtype, device=device)
        label = "MACE-OFF23-%s" % args.model

    # ---- 元素覆盖 + 截断半径体检（从模型本身读，不靠写死） ----
    elems, r_max = model_info(calc)
    have = set(atoms.get_chemical_symbols())
    if elems:
        bad = sorted(have - elems)
        if bad:
            msg = (f"这个模型只支持 {'/'.join(sorted(elems))}，"
                   f"结构里有 {'/'.join(bad)}")
            if args.potential == "qcof":
                print(f"\n[致命] {msg}。")
                print("       QCOF 是在碳氮纳米片上训练的（36k 构象，PBE+D3），")
                print("       含 O/S/F 的 COF（β-酮烯胺、磺酸、氟代）它没见过。")
                print("       改用通用势： python opt_cof.py 结构.cif -p mace-off --model large")
                sys.exit(1)
            print(f"[警告] {msg} —— 模型会把这些元素当不存在，结果不可信。")

    # 截断半径体检：放在元素检查外面，即使模型信息读不到也不影响主流程
    # 用"沿层法向的层间距"判断，不用 c —— c 斜着写的时候两者能差 3%（vin2.cif）
    # 注意：只有"短程、无显式色散"的势才靠真空层避免假镜像成键。
    # mace-mp + D3 是拿来算层间作用的，补真空等于把层间作用删掉，所以不补。
    d_lay = interlayer_spacing(atoms) / layer_count(atoms)
    nl = layer_count(atoms)
    if nl >= 2:
        if args.potential == "mace-mp" and args.no_dispersion:
            print(f"[提示] {nl} 层/晶胞，但你关了 D3 —— 层间几乎没有吸引，"
                  f"c 会飘，别拿这个算层间作用。")
        elif args.potential == "mace-mp":
            print(f"[提示] {nl} 层/晶胞，层间 {d_lay:.2f} Å —— D3 负责层间吸引，"
                  f"按周期正常算，不补真空。")
        elif args.potential == "qcof":
            print(f"[提示] {nl} 层/晶胞，层间 {d_lay:.2f} Å —— 按原样算，不补真空。")
            print("       QCOF 只训过带真空的单层、r_max=4 Å：它没有层间作用，"
                  "别让它弛豫晶胞。")
        else:
            print(f"[提示] {nl} 层/晶胞，层间 {d_lay:.2f} Å —— 按原样算，不补真空"
                  f"（{args.potential.upper()} 能描述层间堆垛，层间距会略微变）。")
    elif r_max and d_lay < r_max + 1.0:
        print(f"[提示] 单层，周期 {d_lay:.2f} Å < 截断 {r_max:.1f} Å —— 计算时"
              f"会临时补真空（不然镜像被当成成键，必炸）。")
        if args.no_vacuum:
            print("       你加了 --no-vacuum，就按原样算（能量/受力很可能不可信）。")
        else:
            print("       这层真空只在计算里用，输出会还原成你的紧凑晶胞。")

    return calc, label, r_max


def strip_main(argv=None):
    """`opt_cof.py strip` 子命令：只删真空，不做任何优化。

    原来这是个独立脚本（strip_vacuum.py），合并进来后只有这一个入口，
    strip_vacuum_layer() 也就不用跨文件 import 了。
    """
    ap = argparse.ArgumentParser(
        prog="opt_cof.py strip",
        description="删掉单层结构的真空层：c 压到「层厚 + gap」，不做优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例:\n"
               "  python opt_cof.py strip Triazine_QCOF.cif\n"
               "  python opt_cof.py strip in.cif --gap 3.0 -o 紧凑.cif\n"
               "  python opt_cof.py strip 双层.cif --c 6.90    # 多层强制压间距\n")
    ap.add_argument("inputs", nargs="+",
                    help="输入 CIF/xyz（可多个，各自存 <原名>_novac.cif）")
    ap.add_argument("-o", "--out", default=None,
                    help="输出名（只在单个输入时有效，默认 <原名>_novac.cif）")
    ap.add_argument("--gap", type=float, default=3.45, metavar="Å",
                    help="保留的层间距，默认 3.45 Å（石墨/COF 的 AA 堆垛距离）")
    ap.add_argument("--c", type=float, default=None, metavar="Å",
                    help="直接指定新 c（多层晶体压间距时用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args(argv)

    if args.out and len(args.inputs) > 1:
        sys.exit("[-o] 只能在单个输入时用；多个输入会各自生成 <原名>_novac.cif")

    for src in args.inputs:
        path = Path(_clean_path(src))
        if not path.exists():
            print(f"[跳过] {src} 不存在")
            continue
        try:
            atoms = read(str(path))
        except Exception as exc:
            print(f"[跳过] {src} 读不了: {type(exc).__name__}: {exc}")
            continue

        n = layer_normal(atoms)
        nl = layer_count(atoms, n)
        th = layer_thickness(atoms, n)
        d = interlayer_spacing(atoms, n)
        print(f"\n{path.name}  {len(atoms)} 原子  {atoms.get_chemical_formula()}")
        print(f"  层数 {nl}   沿法向周期 {d:.3f} Å   层厚 {th:.3f} Å")

        try:
            new, info = strip_vacuum_layer(atoms, gap=args.gap, c_target=args.c)
        except ValueError as exc:
            print(f"  [跳过] {exc}")
            continue
        if new is None:
            print(f"  [跳过] {nl} 层/晶胞，那段空隙就是层间距本身，没有多余真空。"
                  f"要压间距请给 --c（例如 --c {2 * d / nl:.2f}）")
            continue

        out = Path(_clean_path(args.out)) if args.out else path.with_name(
            path.stem + "_novac.cif")
        print(f"  c {d:.3f} → {info['d_new']:.3f} Å"
              f"（删掉 {info['removed']:.3f} Å 真空，层厚 {th:.3f} + 间距 "
              f"{info['d_new'] - th:.2f}）")
        if info.get("straighten"):
            print("  [提示] c 原来是斜的，已扶正（a、b 在面内，α=β=90）")
        if info.get("skew"):
            print("  [提示] a/b 不在层平面内，c 沿原方向缩放，角度保持不变")
        if args.dry_run:
            print(f"  [dry-run] 会写到 {out}")
        else:
            write(str(out), new)
            print(f"  已保存: {out}  （紧凑视图，别再拿它跑优化）")


def main():
    # 子命令要在主 parser 之前拦下来：否则 "strip" 会被当成输入文件名。
    _argv = sys.argv[1:]
    if _argv and _argv[0].lower() in ("strip", "strip-vacuum", "strip_vacuum",
                                      "novac"):
        strip_main(_argv[1:])
        return

    ap = argparse.ArgumentParser(
        description="任意 COF 结构的预优化（QCOF / MACE 势，三步：原子→晶胞→a=b 规整）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="两种用法：\n"
               "  ① 什么都不加直接运行 → 一问一答的向导（最省事）\n"
               "       python opt_cof.py\n"
               "  ② 命令行给参数：\n"
               "       python opt_cof.py LongArm2_COF.cif\n"
               "       python opt_cof.py my.cif -o my_opt.cif -p mace-off --model large\n"
               "  ③ 只删真空层，不优化：\n"
               "       python opt_cof.py strip my.cif --gap 3.45\n"
               "\n产物：<out>.cif、<out>_layer.png、<out>_stage1.log / _cell.traj /\n"
               "      _stage2.log / _stage3.log（按体系命名，互不覆盖）")
    ap.add_argument("input", nargs="?", help="输入结构（.cif / .xyz / .vasp ...），"
                                            "不给则进向导")
    ap.add_argument("-o", "--out", default=None,
                    help="输出 CIF（默认 <输入名>_opt.cif）")
    ap.add_argument("-p", "--potential", default="qcof",
                    choices=["qcof", "mace-off", "mace-mp", "file"],
                    help="势函数：qcof=COF 专用 MACE（默认，单层最准）/ "
                         "mace-mp=通用材料势 +D3 色散（算层间最稳）/ "
                         "mace-off=通用有机势 / file=自定义 .model")
    ap.add_argument("--model-file", default=None,
                    help="指定 .model 路径。-p file 时必给；-p qcof 时用来指定 "
                         "QCOF.model 的位置（默认在 当前目录/脚本目录/~/.cache/mace 找）")
    ap.add_argument("--model", default="medium", choices=["small", "medium", "large"],
                    help="模型尺寸，-p mace-off / mace-mp 时有效（默认 medium）")
    ap.add_argument("--no-dispersion", action="store_true",
                    help="mace-mp 不加 D3 色散。默认加（不加就没有层间吸引，"
                         "层会飘开或压塌）")
    ap.add_argument("--allow-download", action="store_true",
                    help="允许联网下载缺失的模型（QCOF 约 2.6 MB，走 ghfast 镜像）")
    ap.add_argument("--vacuum", type=float, default=None, metavar="Å",
                    help="强制补这么多真空再算（一般不用给：单层 c 太小时会自动补，"
                         "而且计算完会从输出里去掉）")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="连计算用的真空也不补。单层 c < 截断时一定会算炸，只用于排查")
    ap.add_argument("--keep-vacuum", action="store_true",
                    help="输出保留真空层（默认去掉，c 还原成输入的层间距）")
    ap.add_argument("--strip-vacuum", action="store_true",
                    help="【已默认开启，保留参数只为兼容旧命令】")
    ap.add_argument("--strip-gap", type=float, default=3.45, metavar="Å",
                    help="输入本身带真空时，还原用的堆垛间距（默认 3.45 Å）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="计算设备（默认 auto 探测）")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--relax-fmax", type=float, default=0.05, help="阶段1 收敛阈值")
    ap.add_argument("--relax-steps", type=int, default=200, help="阶段1 最多步数")
    ap.add_argument("--cell-fmax", type=float, default=0.01, help="阶段2 收敛阈值")
    ap.add_argument("--cell-steps", type=int, default=500, help="阶段2 最多步数")
    ap.add_argument("--fix-c", action="store_true",
                    help="阶段2 只放开 a、b，c 固定（想保层间距时用）")
    ap.add_argument("--no-cell", action="store_true",
                    help="完全不优化晶胞，只弛豫原子")
    ap.add_argument("--no-plot", action="store_true", help="不画俯视图")
    ap.add_argument("--show", action="store_true",
                    help="画完直接开窗看。当前解释器没有 tkinter/PySide 时"
                         "（matplotlib 只能用 Agg），自动改用系统看图程序打开 PNG")

    # 什么都不给 → 交互式向导（比抛一大段帮助友好得多）
    args = ap.parse_args(_wizard() if len(sys.argv) == 1 else None)
    if not args.input:
        # 给了别的参数但漏了结构文件，比如 python opt_cof.py --model large
        w = _wizard(ask_opts=False)
        args.input = w[0]
        if args.out is None and "-o" in w:
            args.out = w[w.index("-o") + 1]
    args.input = _clean_path(args.input)
    args.out = _clean_path(args.out) if args.out else None

    # 先确认文件在了再加载模型 —— 否则打错一个字母要白等模型初始化
    if not os.path.exists(args.input):
        for ext in _EXT_OK:                 # 也许只是漏了后缀：4444 → 4444.cif
            if os.path.exists(args.input + ext):
                args.input += ext
                print(f"[提示] 按 {args.input} 处理")
                break
        else:
            sys.exit(f"\n找不到文件: {args.input}\n"
                     f"当前目录: {os.getcwd()}\n"
                     f"（直接运行 `python opt_cof.py` 可以进向导挑文件）\n")
    try:
        atoms = read(args.input)
    except Exception as exc:
        sys.exit(f"\n读不了 {args.input}: {exc}\n")
    if atoms.cell.volume < 1e-6:
        sys.exit("输入结构没有晶胞（是个分子），这个脚本需要周期性结构")
    print(f"读入: {args.input}  {len(atoms)} 原子  {atoms.get_chemical_formula()}")
    # 记住输入的晶胞角度：阶段 2 用应变张量 mask=[1,1,0,0,0,0] 放开 a、b，
    # 对六方胞（a 沿 x、b 与 x 成 120°）xx/yy 两种应变不相等时会顺带把 γ 拽走
    # ——实测 CN 从 120 跑到 121.81、NH2 跑到 123.11，阶段 3 只把 a=b 取平均，
    # γ 却再也回不来，晶胞就不是六方了。所以阶段 3 要用输入角度把角度拧回来。
    par_in = atoms.cell.cellpar().copy()

    # 先把势函数准备好：后面补不补真空要看它的截断半径 r_max
    calc, pot_label, r_max = make_calculator(args, atoms)

    # 输出名要先定下来：后面三阶段的日志 / 轨迹都按它的前缀命名，
    # 这样连着优化好几个体系时不会互相覆盖（老版本固定叫 stage1_atoms.log）。
    out = _ensure_ext(args.out or (args.input.rsplit(".", 1)[0] + "_opt.cif"))
    out_path = Path(out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    pre = out_path.stem
    log1 = str(out_path.with_name(pre + "_stage1.log"))
    traj = str(out_path.with_name(pre + "_cell.traj"))
    log2 = str(out_path.with_name(pre + "_stage2.log"))
    log3 = str(out_path.with_name(pre + "_stage3.log"))

    # 平面层若正好压在 z=0/z=c 边界，先挪到中间，避免周期镜像干扰体检
    pos = atoms.get_positions()
    c_len = atoms.cell.cellpar()[2]
    if float(pos[:, 2].max() - pos[:, 2].min()) < 0.5 * c_len:
        pos[:, 2] -= pos[:, 2].mean() - 0.5 * c_len
        atoms.set_positions(pos)
        atoms.wrap()

    # ---- 补真空层：把 AA 堆积的 c=3.6 Å 变成"单层 + 真空" ----
    # MACE 类短程势在 c < r_max 时，每个原子都能看到自己在 z 方向的镜像，
    # 会被当成"原子贴太近" → 巨大排斥 → 整层塌。
    # 实测 QCOF 在 c=3.60 / r_max=4.0 时：能量 -823 → +5.8e16 eV，c 压到 2.16 Å。
    # 所以只要 c < r_max + 1 就自动拉开，除非明确 --no-vacuum。
    c_len = atoms.cell.cellpar()[2]
    n_lay = layer_normal(atoms)
    d_in = interlayer_spacing(atoms, n_lay)     # 进脚本时的真实层间距
    th_in = layer_thickness(atoms, n_lay)
    nl_in = layer_count(atoms, n_lay)
    vac_applied = None
    # 多层（≥2 层/晶胞）本身就是真堆垛，绝不能补真空 —— 补了等于把层间作用删掉。
    # 实测 QCOF 在 c=6.9 的 AA 双层上不补真空也能正常收敛（3.450→3.424，fmax 0.33）。
    keep_stack = nl_in >= 2
    # 只有单层才需要"临时"真空：c < r_max 时每个原子都会看到自己在法向的镜像，
    # 被当成成键 → 巨大排斥 → 整层塌。这段真空只用于计算，输出前一定去掉。
    #
    # 判据要看**层间空隙**（层间距 − 层厚），不是层间距本身：
    # 平面层（th=0）两者相等，但带甲基/支链的层厚度能到 1.8 Å —— 层间距 5.09 Å
    # 看着"大于截断 4 Å"，实际空隙只有 3.30 Å，镜像照样落在截断内。
    # 实测 LZU-1-Me（c=5.089、层厚 1.79）就因为这个差 0.09 Å 的判据没补真空，
    # 阶段 2 一路塌到 500 步、力 2882 eV/Å。
    gap_in = max(0.0, d_in - th_in)
    auto_vac = (args.vacuum is None and not args.no_vacuum
                and not keep_stack
                and r_max and gap_in < r_max + 1.0)
    if nl_in >= 2:
        print(f"\n多层结构：{nl_in} 层/晶胞，沿法向周期 {d_in:.3f} Å —— 按原样算，不补真空。")

    # ---- 多层 + QCOF：提前拦一下 ----
    # 注意这里只针对 QCOF，不针对"MACE 势"整体：MACE-MP-0(+D3) 和 MACE-OFF23
    # 都能给出正常的层间势能极小（实测 vin2 AA 双层：两者都是 3.40 Å 附近有极小，
    # 全弛豫后分别落到 3.592 / 3.532 Å，fmax 都收敛到 0.01 以下）。
    # QCOF 不一样：它只在"带真空的单层碳氮纳米片"上训练，r_max 恰好 4.0 Å。
    # 刚性扫描显示 d ≥ 4 Å 时层间交互严格为零（曲线完全平），一进 4 Å 就单调
    # 下降 57 meV/atom —— 没有极小值，等于把层间接触当成成键。
    # 于是阶段 1（固定晶胞只动坐标）通常正常，一放开晶胞，优化器就一路往下压，
    # 直到两层互穿才炸：实测 c 6.90→5.72、a 27.6→26.5、dmin 0.25 Å、E +9e17 eV；
    # 改 --fix-c 也只是换个崩法（a 27.6→23.5，E +1.5e14）。
    # 这里只警告、不改变任何计算行为，跑不跑由使用者决定。
    if nl_in >= 2 and args.potential == "qcof" and not args.no_cell:
        print("\n" + "!" * 66)
        print(f"[警告] 多层 + QCOF：{pot_label} 只在「带真空的单层」上训练过")
        print(f"       层间 {d_in / nl_in:.2f} Å 落在截断 {(r_max or 0):.1f} Å 以内，"
              f"但训练集里没有任何层间构型。")
        print("       刚性扫描实测：d ≥ 4 Å 层间交互严格为零，进 4 Å 后能量单调下降")
        print("       （−57 meV/atom），没有极小值 —— 一放开晶胞，优化器就会一路")
        print("       往下压，直到两层互穿、能量飙到 1e17 eV 量级。")
        print("       实测 vin2 AA 双层：c 6.90→5.72、a 27.6→26.5、最近原子 0.25 Å；")
        print("       换成 --fix-c 并不解决问题，只是改崩 a（27.6→23.5）。")
        print("       三条可行路线：")
        print("         · 要层间作用（有长程色散）  →  -p mace-mp   （最稳）")
        print("         · 要层间作用（纯短程 ML）    →  -p mace-off  （实测 3.53 Å 收敛）")
        print("         · 只想固定堆垛、弛豫层内骨架  →  --no-cell")
        print("       现在继续跑，大概率还是以『这次优化崩了』收场。")
        print("!" * 66 + "\n")
    if args.vacuum or auto_vac:
        vac = args.vacuum if args.vacuum else 15.0
        vac_applied = vac
        need = th_in + vac            # 目标：沿法向的周期 = 层厚 + 真空
        cell = np.array(atoms.cell)
        c_old = cell[2].copy()
        par_old = atoms.cell.cellpar()
        in_xy = abs(cell[0][2]) < 1e-3 and abs(cell[1][2]) < 1e-3
        if in_xy:
            # a、b 都在 xy 面内（MS 导出的常见情形）：c 直接扶正到 ⟂ ab。
            # 老版本只改 c_z、留着 c_x/c_y，结果 α、β 从 83.7/80.8 被拽到
            # 88.7/88.1，c 还得乘 1/cos(倾角)，数字全是乱的。
            cell[2] = np.array([0.0, 0.0, need])
        else:
            perp_old = max(abs(float(c_old @ n_lay)), 1e-6)
            cell[2] = c_old * (need / perp_old)
        atoms.set_cell(Cell(cell), scale_atoms=False)
        atoms.center(axis=2)
        atoms.wrap()
        print(f"\n真空层（仅计算用）: 层厚 {th_in:.2f} Å + 真空 {vac:.1f} Å "
              f"→ 沿法向周期 {need:.2f} Å")
        print(f"        层间距 {d_in:.3f} Å → {need:.3f} Å"
              f"（变大是因为加了 {vac:.1f} Å 真空，不是层被拉开了）")
        ang = np.degrees(np.arccos(min(1.0, abs(float(
            c_old @ n_lay)) / max(float(np.linalg.norm(c_old)), 1e-9))))
        if in_xy and (abs(par_old[3] - 90) > 0.05 or abs(par_old[4] - 90) > 0.05):
            print(f"        顺带把斜着的 c 扶正: α {par_old[3]:.2f}→90.00, "
                  f"β {par_old[4]:.2f}→90.00（原 c 与层法向差 {ang:.1f}°）")
        if auto_vac:
            print(f"        （自动：层厚 {th_in:.2f} Å，层间空隙 {gap_in:.2f} Å "
                  f"< 截断 {r_max:.1f} Å + 1，镜像会被当成成键）")
        if not args.keep_vacuum:
            print("        输出时会把真空去掉，还原成你的紧凑晶胞")
        if not args.no_cell:
            args.fix_c = True                      # 真空层不该被 BFGS 压掉
            print("        阶段 2 自动改为只放开 a、b（--fix-c）")

    atoms.calc = calc

    p0 = report(atoms, "优化前", vacuum=vac_applied)
    x0 = atoms.get_positions().copy()
    e0 = float(atoms.get_potential_energy())

    t0 = time.time()
    made = []                       # 实际写出来的日志，收尾时只报这些
    # ---------------- 阶段 1：固定晶胞，弛豫原子 ----------------
    print(f"\n>>> 阶段 1：固定晶胞，FIRE 弛豫原子坐标（fmax={args.relax_fmax}, ≤{args.relax_steps} 步）")
    stage1 = FIRE(atoms, logfile=log1)
    stage1.run(fmax=args.relax_fmax, steps=args.relax_steps)
    made.append(log1)
    print(f"    完成 {stage1.nsteps} 步，t={time.time() - t0:.1f}s")

    # ---------------- 阶段 2：放开晶胞尺寸 ----------------
    if not args.no_cell:
        mask = [1, 1, 0, 0, 0, 0] if args.fix_c else [1, 1, 1, 0, 0, 0]
        print(f"\n>>> 阶段 2：FrechetCellFilter 同步优化晶胞与坐标"
              f"（mask={mask}, fmax={args.cell_fmax}, ≤{args.cell_steps} 步）")
        opt_cell = BFGS(FrechetCellFilter(atoms, mask=mask),
                        trajectory=traj, logfile=log2)
        opt_cell.run(fmax=args.cell_fmax, steps=args.cell_steps)
        made.append(log2)
        print(f"    完成 {opt_cell.nsteps} 步，t={time.time() - t0:.1f}s")

        # ---------------- 阶段 3：a = b 规整 ----------------
        par = atoms.cell.cellpar()
        # 角度用**输入**的（par_in），不用阶段 2 跑出来的。
        # 阶段 2 的 xx/yy 独立应变会顺带改变 γ（六方胞实测偏 0.4~3.1°），
        # 只把 a、b 取平均并不能把晶胞还原成六方 —— 角度必须显式拧回去。
        ang_fix = par_in[3:6]
        d_ang = max(abs(par[3] - ang_fix[0]), abs(par[4] - ang_fix[1]),
                    abs(par[5] - ang_fix[2]))
        need_ab = abs(par[0] - par[1]) > 1e-3
        need_ang = d_ang > 0.02
        if need_ab or need_ang:
            ab = 0.5 * (par[0] + par[1])
            msg = []
            if need_ab:
                msg.append(f"a={par[0]:.3f} ≠ b={par[1]:.3f}，取平均 {ab:.3f}")
            if need_ang:
                msg.append(f"角度 {par[3]:.2f}/{par[4]:.2f}/{par[5]:.2f} → "
                           f"{ang_fix[0]:.2f}/{ang_fix[1]:.2f}/{ang_fix[2]:.2f}")
            print(f"\n>>> 阶段 3：{'；'.join(msg)}，规整后再弛豫")
            atoms.set_cell(cellpar_to_cell([ab, ab, par[2], *ang_fix]),
                           scale_atoms=True)
            opt_final = BFGS(atoms, logfile=log3)
            opt_final.run(fmax=args.cell_fmax, steps=200)
            made.append(log3)
        else:
            print("\n>>> 阶段 3：a = b 且角度未漂移，跳过")
    else:
        print("\n>>> 跳过晶胞优化（--no-cell）")

    p1 = report(atoms, "优化后", vacuum=vac_applied)
    e1 = float(atoms.get_potential_energy())
    dx = atoms.get_positions() - x0
    rms = float(np.sqrt((dx ** 2).sum(axis=1).mean()))

    # ---- 崩没崩？短程势一旦塌陷，能量会飞到 1e13 eV 这种天文数字。 ----
    # 这种结果不能当"优化完成"报给用户，否则 CIF 拿进 MS 才发现是垃圾。
    _D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(_D, 99.0)
    dmin = float(_D.min())
    broken = []
    if not np.isfinite(e1) or e1 / len(atoms) > 0:
        broken.append(f"能量为 {'非有限' if not np.isfinite(e1) else '正数'}"
                      f"（{e1:.3g} eV）")
    if dmin < 0.75:
        broken.append(f"最近原子间距只剩 {dmin:.2f} Å")
    if p1[2] < 3.0 and p1[2] < 0.95 * p0[2]:
        broken.append(f"c 从 {p0[2]:.2f} 塌到 {p1[2]:.2f} Å")
    if broken:
        print("\n" + "!" * 66)
        print("[警告] 这次优化崩了：" + "；".join(broken))
        print("       结果不可信，别直接拿去用。常见原因和办法：")
        print("       1) c 小于截断半径 → 周期镜像被当成成键。单层会自动补真空，"
              "若你加了 --no-vacuum 请去掉。")
        print("       2) 输入结构本身有原子重叠 → 先跑 "
              "python build_cof_smiles.py --inspect 看一眼。")
        print("       3) 元素不在训练集里 → 换 -p mace-off --model large。")
        print("       4) 局部太离谱 → 降低步长：--relax-fmax 0.1 先粗弛豫。")
        print("       5) 想连层间一起算 → -p mace-mp（QCOF 只训过单层，没有层间作用）")
        print("!" * 66)

    # ---- 输出前把计算用的真空去掉 ----
    # 真空只是短程势的"脚手架"：单层 c < r_max 时不补就必炸（实测 vin2 跑到 40 步
    # 能量 +2757 eV、力 2.5e5、原子间距 0.655 Å）。但它是内部的，不该留在交付文件里。
    # 默认输出紧凑晶胞：c = 输入的真实层间距（输入本身带真空就用 --strip-gap）。
    out_atoms = atoms
    c_out = None
    c_from_vac = False
    if not args.keep_vacuum:
        nl_out = layer_count(atoms)
        if nl_out == 1:
            th_out = layer_thickness(atoms)
            gap_in = d_in / max(nl_in, 1)
            # 输入的"层间距"若大到像真空（> 6 Å），就退回默认堆垛距离
            c_from_vac = not (2.0 <= gap_in <= 6.0)
            c_out = th_out + args.strip_gap if c_from_vac else gap_in
            # 单层在真空里弛豫时没有层间 π-堆积压着，大孔或带取代基的层会自己起皱
            # （实测 LZU-1-Me 层厚 1.79→4.72 Å、TAPB-BPDA 0→3.13 Å）。
            # 这时若还沿用输入的 c（3.6 Å），输出就是把皱掉的层硬塞回原来的层间距，
            # 相邻层的原子直接互相穿插。这里按"新的层厚 + 堆垛距离"兜底。
            if th_out + args.strip_gap > c_out + 0.05:
                print(f"\n[提示] 优化后层厚 {th_in:.2f} → {th_out:.2f} Å"
                      f"（单层无层间作用，层自己起了皱），"
                      f"输出的 c 由 {c_out:.2f} 放大到 {th_out + args.strip_gap:.2f} Å")
                c_out = th_out + args.strip_gap
            try:
                cand, _inf = strip_vacuum_layer(atoms, c_target=c_out)
                if cand is not None:
                    out_atoms = cand
            except ValueError:
                c_out = None

    # 先把结构落盘，再做格式化输出。
    # 老顺序是先打印一大堆、最后才 write，一旦写出错（比如输出名没带 .cif）
    # 就是几分钟的优化白跑，而且只丢一个 ASE 的 UnknownFileTypeError。
    try:
        write(out, out_atoms)
    except Exception as exc:
        sys.exit(f"\n[失败] 结构写不进 {out}\n"
                 f"       {type(exc).__name__}: {exc}\n"
                 f"       优化本身是成功的，日志在 {'、'.join(made)}。\n"
                 f"       换个输出名（形如 xxx_opt.cif）再跑一次即可。\n")

    print("\n===== 前后对比 =====")
    print(f"  a      {p0[0]:8.3f} → {p1[0]:8.3f} Å   ({(p1[0] - p0[0]) / p0[0] * 100:+.2f}%)")
    print(f"  b      {p0[1]:8.3f} → {p1[1]:8.3f} Å   ({(p1[1] - p0[1]) / p0[1] * 100:+.2f}%)")
    ctag = "c(计算用)" if vac_applied else "c      "
    print(f"  {ctag} {p0[2]:8.3f} → {p1[2]:8.3f} Å   ({(p1[2] - p0[2]) / p0[2] * 100:+.2f}%)")
    nl_out = layer_count(atoms)
    if c_out:
        tail = ("输入本身带真空，用默认堆垛间距" if c_from_vac
                else "还原成你输入的层间距")
        print(f"  输出的 c  {c_out:8.3f} Å   ← 已去掉真空，{tail}")
    elif vac_applied:
        print(f"  层间距   原 {d_in:6.3f} Å → 单层 + {vac_applied:.1f} Å 真空"
              f"（--keep-vacuum 保留了真空）")
    elif nl_out >= 2:
        print(f"  层间距   {d_in / nl_in:6.3f} → {interlayer_spacing(atoms) / nl_out:6.3f} Å"
              f"   （{nl_out} 层/晶胞，沿层法向）")
    print(f"  能量   {e0:9.4f} → {e1:9.4f} eV   (Δ={e1 - e0:+.4f} eV, "
          f"{(e1 - e0) / len(atoms) * 1000:+.2f} meV/atom)")
    print(f"  原子位移 RMS {rms:.3f} Å，最大 {float(np.sqrt((dx ** 2).sum(axis=1)).max()):.3f} Å")
    print(f"  势函数 {pot_label}      总耗时 {time.time() - t0:.1f}s")

    print(f"\n已保存: {out}" + ("   ← 但结果不可信，见上面的警告" if broken else ""))
    print("  日志: " + "、".join(made))
    if not args.no_cell:
        print(f"  轨迹: {traj}")

    if args.keep_vacuum and vac_applied:
        print(f"  真空   --keep-vacuum 生效：c 保留 {p1[2]:.3f} Å（含 {vac_applied:.1f} Å 真空）")

    if not args.no_plot:
        # 画图模块只负责出图，不依赖建模代码。原来这里 import 的是
        # build_cof_image（那个文件还带着一整套三嗪建模代码，清理时被挪进
        # _trash_20261023/ 了），于是每次都 ModuleNotFoundError，又被 except
        # 吞成一句「跳过画图」—— 看起来就像这个脚本根本不做可视化。
        plotter = None
        for mod in ("cof_view", "build_cof_image"):
            try:
                plotter = __import__(mod, fromlist=["plot_layer"]).plot_layer
                break
            except ImportError:
                continue
        if plotter is None:
            print("[提示] 找不到画图模块 cof_view.py（应在脚本同目录），跳过画图")
        else:
            png = str(out_path.with_name(pre + "_layer.png"))
            try:
                try:
                    plotter(atoms, out=png, rep=2, show=args.show)
                except TypeError:
                    # 老版 build_cof_image.plot_layer 没有 show 参数
                    plotter(atoms, out=png, rep=2)
            except Exception as exc:
                # 这里要报出真实原因，别只写「跳过画图」
                print(f"[提示] 画图失败: {type(exc).__name__}: {exc}")
            else:
                print(f"  俯视图: {png}")


if __name__ == "__main__":
    main()
