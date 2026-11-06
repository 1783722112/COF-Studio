# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：COF Studio 桌面版

产出：dist/COF Studio/COFStudio.exe（onedir，双击即用，无需装 Python）

两条关键点：
  1. ui.html 和 engines/ 作为“数据文件”塞进包里，路径由 cof_studio.py 的 RES 取；
  2. 引擎脚本不在打包时执行，而是运行时由 exe 以 --pyexec 把自己当解释器拉起，
     所以 ase.io / rdkit 这些“动态 import”的模块必须显式列进 hiddenimports。
"""
from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs,
                                     collect_submodules)

datas = [
    ("ui.html", "."),          # 界面
    ("engines", "engines"),    # 四个引擎脚本
]
datas += collect_data_files("rdkit")
datas += collect_dynamic_libs("rdkit")
datas += collect_data_files("ase")

hiddenimports = [
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.AllChem",
    "rdkit.Chem.Descriptors",
    "rdkit.Chem.rdMolTransforms",
    "rdkit.Chem.rdDistGeom",
    "rdkit.Chem.rdForceFieldHelpers",
    "rdkit.Chem.rdchem",
    "rdkit.Geometry",
    "scipy.spatial",
    "scipy.spatial.cKDTree",
    "scipy.sparse",
]
hiddenimports += collect_submodules("ase.io")      # ase 按扩展名动态找格式模块
hiddenimports += collect_submodules("ase.geometry")

# 这些只在源码模式（自带解释器）里用到，打进去纯属浪费几十 MB
excludes = [
    "tkinter", "matplotlib", "pandas", "PIL", "IPython", "pytest",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "notebook", "nbformat",
    "sympy", "numba", "llvmlite", "torch", "mace", "setuptools", "pip",
    "distutils", "docutils", "jinja2", "sqlite3",
    # jax / jaxlib：rdkit 新版依赖链会把 jaxlib 拉进来，光 jax_common.dll
    # 就 240 MB，而 COF Studio 的 rdkit 用法（嵌构象、MMFF、子结构匹配）
    # 全在 C++ 侧，根本不调 jax。不排除的话整包从 40 MB 涨到 148 MB。
    "jax", "jaxlib", "jaxlib.mlir",
]

a = Analysis(
    ["cof_studio.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="COFStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # 窗口程序：双击不弹黑框
    disable_windowed_traceback=False,
    icon="cof_icon.ico",
    version="version_info.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="COF Studio",
)
