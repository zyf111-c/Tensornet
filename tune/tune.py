from __future__ import annotations

import json
import warnings
import numpy as np
from functools import partial
import os
import glob
import torch
import lightning as L
from dgl.data.utils import split_dataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback

from matgl.ext._pymatgen_dgl import Structure2Graph
from matgl.graph._data_dgl import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.config import DEFAULT_ELEMENTS

import matgl
from matgl.utils._training_dgl import PotentialLightningModule

warnings.simplefilter("ignore")

# ========== 1. 加载数据 ==========
print("=" * 60)
print("Step 1: Loading data...")
print("=" * 60)

data = []
for f in glob.glob("./chunk_batches/chunk_*.json"):
    with open(f, 'r') as fp:
        data.extend(json.load(fp))

print(f"读取完成！共 {len(data)} 条数据")

# 提取数据
structures = []
energies = []
forces = []
stresses = []

for item in data:
    from pymatgen.core import Structure
    
    structure = Structure.from_dict(item['structure'])
    structures.append(structure)
    energies.append(float(item['energy']))
    
    force = item.get('forces')
    if force is not None:
        forces.append(force)
    else:
        forces.append(np.zeros((len(structure), 3)).tolist())
    
    stress = item.get('stress')
    if stress is not None:
        stresses.append(stress)
    else:
        stresses.append(np.zeros((3, 3)).tolist())

labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses,
}

print(f"Extracted: {len(structures)} structures, {len(energies)} energies")
print(f"  - With stress data: {sum(1 for s in stresses if np.any(np.array(s)))}")

# ========== 2. 创建数据集 ==========
print("\n" + "=" * 60)
print("Step 2: Creating dataset...")
print("=" * 60)

# 使用 DEFAULT_ELEMENTS（与预训练模型一致）
element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
    include_line_graph=True,
)

# ========== 3. 划分数据集 ==========
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# ========== 4. 创建数据加载器 ==========
my_collate_fn = partial(
    collate_fn_pes, 
    include_line_graph=True, 
    include_stress=True    
)

batch_size = 8
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=batch_size,
    num_workers=0,
)

# ========== 5. 加载预训练模型 ==========
print("\n" + "=" * 60)
print("Step 3: Loading pretrained model...")
print("=" * 60)

model_path = "/AI4S/Users/zyf/tensornet/pretrained_models/TensorNetDGL-MatPES-PBE-v2025.1-PES"
m3gnet_nnp = matgl.load_model(model_path)
model_pretrained = m3gnet_nnp.model

print(f"Model loaded from: {model_path}")

# ========== 6. 获取元素参考能 ==========
print("\n" + "=" * 60)
print("Step 4: Getting element reference energies...")
print("=" * 60)

if hasattr(m3gnet_nnp, 'element_refs') and m3gnet_nnp.element_refs is not None:
    property_offset = m3gnet_nnp.element_refs.property_offset
    print(f"✅ Element reference energies found! Shape: {property_offset.shape}")
else:
    property_offset = None
    print("⚠️  No element reference energies found")


# ========== 8. 创建 Lightning 模块 ==========
print("\n" + "=" * 60)
print("Step 6: Creating Lightning module...")
print("=" * 60)

learning_rate = 1e-5

lit_module_finetune = PotentialLightningModule(
    model=model_pretrained,
    element_refs=property_offset, 
    lr=learning_rate,
    include_line_graph=True,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.1,     
)


# ========== 9. 自定义 Callback ==========
class SaveModelEveryNEpochs(Callback):
    def __init__(self, save_dir="./save", save_every_n_epochs=10):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.save_dir, exist_ok=True)
        self.saved_epochs = set()
        
    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        if current_epoch % self.save_every_n_epochs == 0:
            if current_epoch not in self.saved_epochs:
                self.saved_epochs.add(current_epoch)
                save_path = os.path.join(self.save_dir, f"epoch_{current_epoch}")
                pl_module.model.save(save_path)
                
                val_mae = trainer.callback_metrics.get('val_Energy_MAE', float('inf'))
                print(f"\n  💾 Model saved at epoch {current_epoch} (val_Energy_MAE: {val_mae:.4f})")

save_callback = SaveModelEveryNEpochs(
    save_dir="./save",
    save_every_n_epochs=10
)

# ========== 10. 设置训练器 ==========
logger = CSVLogger("logs", name="lic.log")

trainer = L.Trainer(
    max_epochs=2000,
    accelerator="gpu",
    logger=logger,
    callbacks=[save_callback],
    inference_mode=False,
    log_every_n_steps=10,
)


# ========== 11. 开始训练 ==========
print("\n开始训练...")
trainer.fit(
    model=lit_module_finetune,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

print("\n✅ Training completed!")

# ========== 12. 保存最终模型 ==========
print("\n" + "=" * 60)
print("Step 8: Saving final model...")
print("=" * 60)

model_save_path = "./finetuned_model_lpsc/"
lit_module_finetune.model.save(model_save_path)
print(f"✅ Final model saved to: {model_save_path}")

# ========== 13. 测试模型 ==========
print("\n" + "=" * 60)
print("Step 9: Testing model...")
print("=" * 60)

test_results = trainer.test(
    model=lit_module_finetune,
    dataloaders=test_loader,
    verbose=True
)

if test_results:
    print("\nTest Results:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")

print("\n" + "=" * 60)
print("✅ All done!")
print("=" * 60)
