from __future__ import annotations
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import glob
from datetime import timedelta

import pandas as pd
import seaborn as sns

import sys

# 添加 gnnp_driver 路径
sys.path.insert(0, '/AI4S/Users/zyf/tensornet/lammps/src/ML-GNNP')

# 设置 PYTHONPATH 环境变量（让 LAMMPS 能读到）
os.environ["PYTHONPATH"] = f"/AI4S/Users/zyf/tensornet/lammps/src/ML-GNNP:{os.environ.get('PYTHONPATH', '')}"

n_cpus = 4

HOME_DIR = os.environ["HOME"]

import subprocess
import os
from datetime import timedelta

HOME_DIR = os.path.expanduser("~")
n_cpus = 4

# 设置库路径
os.environ["LD_LIBRARY_PATH"] = f"{HOME_DIR}/.local/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["OMP_NUM_THREADS"] = f"{n_cpus}"

# 使用新编译的 LAMMPS
LMP_EXE = "/AI4S/Users/zyf/tensornet/lammps/build/lmp"
outfile = "lammps.out"  
# 运行 LAMMPS
lammps_command = f"{LMP_EXE} -in lammps.in > {outfile}"

import time
start = time.time()
result = subprocess.run(lammps_command, shell=True)
end = time.time()
runtime = end - start
