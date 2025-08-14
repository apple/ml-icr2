#!/bin/bash
source /miniconda/etc/profile.d/conda.sh

# # # Prepare icr env
conda activate easycontext

# # Install flash attention
# conda install cuda -c nvidia
# conda create -n easycontext python=3.10 -y && conda activate easycontext

# pip install --pre torch==2.4.0.dev20240519+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
# apt-get update && apt-get install -y g++
# pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
# pip install -r requirements.txt

cd src/DeepSpeed
DS_BUILD_CPU_ADAM=1 python3 -m pip install -e .
cd ..
cd ..

git config --global --add safe.directory /mnt/task_runtime

apt-get install -y screen

pip install prettytable
pip install seaborn
