from __future__ import annotations

import os
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write, read
from ase.mep import NEB
from ase.optimize import BFGS, QuasiNewton
from ase.utils.forcecurve import fit_images
import matplotlib.pyplot as plt
import time
import warnings
import matgl
matgl.config.BACKEND = "DGL"
from matgl.ext.ase import PESCalculator, Relaxer
from matgl.apps._pes_dgl import Potential

warnings.simplefilter("ignore")

MODEL_PATH = "/AI4S/Users/zyf/tensornet/SSE-tune/all_lpsc_freeze/save/epoch_990"
DATA_DIR = "lpsc_neb_eb_stable"
N_IMAGES = 7
FMAX_TARGET = 0.05
MAX_STEPS = 1000
RELAX_FMAX = 0.02

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("NEB 计算（参考版逻辑：method='eb' + 独立计算器 + QuasiNewton）")
print("出处：https://github.com/sai-mat-group/mlips-migration-barriers/blob/main/codes/M3GNet.ipynb")
print("=" * 60)

#load_model
my_potential = Potential.load(MODEL_PATH)

#read structure
start_structure = Poscar.from_file("POSCAR_start").structure
end_structure = Poscar.from_file("POSCAR_end").structure
print(f"起始: {len(start_structure)} 原子, 终点: {len(end_structure)} 原子")

#ASE Atoms
start_atoms = AseAtomsAdaptor.get_atoms(start_structure)
end_atoms = AseAtomsAdaptor.get_atoms(end_structure)

start_atoms.set_constraint([])
end_atoms.set_constraint([])

calc_start = PESCalculator(
    potential=my_potential,
    compute_stress=False,
)
start_atoms.calc = calc_start
qn = QuasiNewton(start_atoms, trajectory=f"{DATA_DIR}/initial_relaxed.traj")
qn.run(fmax=RELAX_FMAX, steps=1000)
start_energy = start_atoms.get_potential_energy()
print(f"initial energy: {float(start_energy):.6f} eV")

calc_end = PESCalculator(
    potential=my_potential,
    compute_stress=False,
)
end_atoms.calc = calc_end
qn = QuasiNewton(end_atoms, trajectory=f"{DATA_DIR}/final_relaxed.traj")
qn.run(fmax=RELAX_FMAX, steps=1000)
end_energy = end_atoms.get_potential_energy()
print(f"final energy: {float(end_energy):.6f} eV")


start_atoms = read(f"{DATA_DIR}/initial_relaxed.traj")
end_atoms = read(f"{DATA_DIR}/final_relaxed.traj")

Poscar(AseAtomsAdaptor.get_structure(start_atoms)).write_file(f"{DATA_DIR}/start_relaxed.vasp")
Poscar(AseAtomsAdaptor.get_structure(end_atoms)).write_file(f"{DATA_DIR}/end_relaxed.vasp")
print("save structure..")

print("\n" + "=" * 60)
print("NEB step")
print("=" * 60)

images = [start_atoms.copy()]
for _ in range(N_IMAGES):
    img = start_atoms.copy()
    img.set_constraint([])
    images.append(img)
images.append(end_atoms.copy())

print("为每个图像分配独立的计算器.........")
for i, img in enumerate(images):
    # 每个图像独立加载模型，避免共享问题
    pot_i = Potential.load(MODEL_PATH)
    calc_i = PESCalculator(
        potential=pot_i,
        compute_stress=False,
    )
    img.calc = calc_i

neb = NEB(images, climb=False, method="eb", allow_shared_calculator=False)
neb.interpolate(method='idpp')

for i, img in enumerate(images):
    write(f"{DATA_DIR}/initial_image_{i:02d}.vasp", img, vasp5=True)
print(f"✓ create {len(images)} images (method='eb')")

print("\n" + "=" * 60)
print("NEB calculation (BFGS, method='eb')")
print("=" * 60)

start_time = time.time()

neb = NEB(images, climb=False, k=0.05, method="eb", allow_shared_calculator=False)
optimizer = BFGS(neb, trajectory=f"{DATA_DIR}/neb.traj", logfile=f"{DATA_DIR}/neb.log")


optimizer.run(fmax=FMAX_TARGET, steps=MAX_STEPS)

total_time = time.time() - start_time
print(f"\n total time: {total_time:.1f}s")

for i, img in enumerate(images):
    write(f"{DATA_DIR}/final_image_{i:02d}.vasp", img, vasp5=True)

print("\n" + "=" * 60)
print("single point calculation")
print("=" * 60)

single_point_energies = []
for i, img in enumerate(images):
    pot_sp = Potential.load(MODEL_PATH)
    calc_sp = PESCalculator(
        potential=pot_sp,
        compute_stress=False,
    )
    atoms_copy = img.copy()
    atoms_copy.calc = calc_sp
    
    try:
        energy = atoms_copy.get_potential_energy()
        single_point_energies.append(energy)
        print(f"  Image {i:2d}: {energy:.6f} eV")
    except Exception as e:
        print(f"  Image {i:2d}: false - {e}")
        single_point_energies.append(0.0)

single_point_energies = np.array(single_point_energies)


barrier = max(single_point_energies) - min(single_point_energies)
dE = single_point_energies[-1] - single_point_energies[0]
ts_idx = np.argmax(single_point_energies)

fmax = 0.0
for img in images:
    try:
        forces = img.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        fmax = max(fmax, max_force)
    except:
        pass

converged = fmax < FMAX_TARGET

#results
print(f"\n=== 最终结果 ===")
print(f"反应能垒: {barrier:.6f} eV")
print(f"反应能: {dE:.6f} eV")
print(f"最大力: {fmax:.6f} eV/Å")
print(f"鞍点在图像: {ts_idx}")
print(f"收敛: {'✅ 已收敛' if converged else '❌ 未收敛'}")

print("\n能量路径:")
for i, e in enumerate(single_point_energies):
    rel_e = e - single_point_energies[0]
    print(f"  Image {i}: {e:.6f} eV  (相对: {rel_e:+.6f} eV)")

with open(f"{DATA_DIR}/neb_results.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("NEB 计算结果 (method='eb')\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"起始结构优化后能量: {float(start_energy):.6f} eV\n")
    f.write(f"终点结构优化后能量: {float(end_energy):.6f} eV\n")
    f.write(f"反应能垒: {barrier:.6f} eV\n")
    f.write("能量路径:\n")
    for i, e in enumerate(single_point_energies):
        rel_e = e - single_point_energies[0]
        f.write(f"  Image {i}: {e:.6f} eV  (相对: {rel_e:+.6f} eV)\n")

print(f"\n✓ 已保存: {DATA_DIR}/neb_results.txt")

print("\n绘图...")
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(single_point_energies))
ax.plot(x, single_point_energies - single_point_energies[0], 'bo-', linewidth=2, markersize=8)
ax.plot(ts_idx, single_point_energies[ts_idx] - single_point_energies[0], 'ro', markersize=12,
        label=f'鞍点: {barrier:.4f} eV')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Image Number')
ax.set_ylabel('Relative Energy (eV)')
ax.set_title(f'NEB Energy Profile (method="eb")\nBarrier: {barrier:.4f} eV')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/energy_profile.png", dpi=300, bbox_inches='tight')
print(f"✓ 已保存: {DATA_DIR}/energy_profile.png")
plt.show()

print("\n" + "=" * 60)
print("NEB 计算完成！")
print("=" * 60)
print(f"\n结果目录: {DATA_DIR}/")
print(f"  - 弛豫后起始: start_relaxed.vasp")
print(f"  - 弛豫后终点: end_relaxed.vasp")
print(f"  - 初始路径: initial_image_*.vasp")
print(f"  - 最终路径: final_image_*.vasp")
print(f"  - 能垒图: energy_profile.png")
print(f"  - 结果: neb_results.txt")
print(f"  - 轨迹: neb.traj")
print(f"  - 日志: neb.log")
