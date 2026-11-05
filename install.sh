#!/usr/bin/env bash
# COF Studio —— macOS / Linux 环境安装
set -e
cd "$(dirname "$0")"

echo
echo "============================================================"
echo "   COF Studio  环境安装"
echo "============================================================"
echo

PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "  [错误] 没找到 python3，请先安装 Python 3.10+"
  echo "         macOS:  brew install python@3.12"
  echo "         Ubuntu: sudo apt install python3 python3-venv python3-pip"
  exit 1
fi
echo "  使用解释器: $($PY --version)"

echo
echo "------------------------------------------------------------"
echo "  [1/3] 创建虚拟环境 .venv"
echo "------------------------------------------------------------"
if [ -x ".venv/bin/python" ]; then
  echo "  已存在，跳过"
else
  $PY -m venv .venv
  echo "  完成"
fi
VPY=".venv/bin/python"

echo
echo "------------------------------------------------------------"
echo "  [2/3] 建模依赖 (numpy ase scipy matplotlib rdkit)"
echo "------------------------------------------------------------"
"$VPY" -m pip install --upgrade pip -q
if ! "$VPY" -m pip install numpy ase scipy matplotlib rdkit; then
  echo
  echo "  [错误] 安装失败，可试国内镜像："
  echo "    $VPY -m pip install numpy ase scipy matplotlib rdkit -i https://pypi.tuna.tsinghua.edu.cn/simple"
  exit 1
fi
echo "  建模环境就绪"

echo
echo "------------------------------------------------------------"
echo "  [3/3] 优化依赖 (torch + mace-torch，约 1~2 GB)"
echo "------------------------------------------------------------"
printf "  现在安装吗？不装也能建模 [y/N]: "
read -r ANS
if [ "$ANS" = "y" ] || [ "$ANS" = "Y" ]; then
  echo "  安装 torch ..."
  "$VPY" -m pip install torch || echo "  [警告] torch 安装失败"
  echo "  安装 mace-torch ..."
  "$VPY" -m pip install mace-torch torch-dftd || echo "  [警告] mace-torch 安装失败"
  echo "  优化环境就绪"
else
  echo "  跳过（之后可重新运行本脚本安装）"
fi

echo
echo "------------------------------------------------------------"
echo "  自检"
echo "------------------------------------------------------------"
"$VPY" -c "import numpy,ase,scipy,matplotlib;print('  建模: numpy',numpy.__version__,'ase',ase.__version__,' OK')" || echo "  建模: 有问题"
"$VPY" -c "import torch,mace;print('  优化: torch',torch.__version__,'mace OK')" 2>/dev/null || echo "  优化: 未安装（不影响建模）"

echo
echo "============================================================"
echo "   安装完成。运行 ./run.sh 启动界面。"
echo "============================================================"
echo
