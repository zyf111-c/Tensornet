from emmet.core.vasp.calculation import Calculation
from pymatgen.io.vasp import Vasprun
from tqdm import tqdm
import glob
import json
import os
import gc
import torch
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据提取函数 ====================

def load_single_point(dirname):
    """使用emmet提取单点能数据"""
    try:
        calculation = Calculation.from_vasp_files(
            dir_name=dirname, task_name=dirname,
            vasprun_file="vasprun.xml", outcar_file="OUTCAR", contcar_file="CONTCAR"
        )
        dump = calculation[0].model_dump()
        if not dump.get('has_vasp_completed', False):
            return None
        
        structure = dump['output']['ionic_steps'][-1]['structure']
        structure.set_charge(0.0)
        
        output = {
            "structure": structure.as_dict(),
            "energy": dump['output']['energy'],
            "forces": dump['output']['ionic_steps'][-1]['forces'],
            "stress": [[-0.1 * x for x in row] for row in dump['output']['ionic_steps'][-1]['stress']],
            "step": -1,
            "total_steps": len(dump['output']['ionic_steps']),
            "source_type": "single_point"
        }
        return output
    except Exception as e:
        return None

def load_aimd_trajectory(dirname):
    """使用Vasprun提取AIMD轨迹 - 提取所有步数"""
    try:
        vasprun_file = os.path.join(dirname, "vasprun.xml")
        if not os.path.exists(vasprun_file):
            return None
        
        vr = Vasprun(vasprun_file, parse_potcar_file=False, 
                     exception_on_bad_xml=False)
        if not vr.ionic_steps:
            return None
        
        results = []
        total_steps = len(vr.ionic_steps)
        
        for step_idx, step in enumerate(vr.ionic_steps):
            try:
                structure = step.get('structure')
                if structure is None:
                    continue
                structure.set_charge(0.0)
                
                energy = step.get('e_0_energy')
                if energy is None:
                    energy = step.get('e_fr_energy', 0.0)
                
                forces = step.get('forces')
                if forces is None:
                    if hasattr(vr, 'ionic_step_forces'):
                        try:
                            forces = vr.ionic_step_forces(step_idx)
                        except:
                            forces = None
                if forces is None:
                    forces = [[0.0, 0.0, 0.0] for _ in structure.sites]
                
                stress = step.get('stress')
                if stress is not None:
                    stress = [[-0.1 * x for x in row] for row in stress]
                
                if len(structure.sites) > 0:
                    result = {
                        "structure": structure.as_dict(),
                        "energy": float(energy),
                        "forces": forces.tolist() if hasattr(forces, 'tolist') else forces,
                        "stress": stress,
                        "step": step_idx,
                        "total_steps": total_steps,
                        "source_type": "aimd_trajectory"
                    }
                    results.append(result)
            except Exception as e:
                continue
        
        return results if results else None
    except Exception as e:
        return None

def detect_data_type(dirname):
    """检测目录是单点能还是AIMD"""
    vasprun_file = os.path.join(dirname, "vasprun.xml")
    if not os.path.exists(vasprun_file):
        return "single_point"
    try:
        file_size = os.path.getsize(vasprun_file) / (1024 * 1024)
        if file_size > 10:
            return "aimd"
        vr = Vasprun(vasprun_file, parse_potcar_file=False, exception_on_bad_xml=False)
        if len(vr.ionic_steps) > 10:
            return "aimd"
        return "single_point"
    except:
        return "single_point"

def load_vasp_data(dirname):
    """智能加载VASP数据 - 不缩减AIMD"""
    data_type = detect_data_type(dirname)
    if data_type == "aimd":
        return load_aimd_trajectory(dirname)
    else:
        result = load_single_point(dirname)
        if result is not None:
            return [result]
        # 备用方案
        try:
            vasprun_file = os.path.join(dirname, "vasprun.xml")
            if os.path.exists(vasprun_file):
                vr = Vasprun(vasprun_file, parse_potcar_file=False, exception_on_bad_xml=False)
                if vr.ionic_steps:
                    last_step = vr.ionic_steps[-1]
                    structure = last_step.get('structure')
                    if structure is not None:
                        structure.set_charge(0.0)
                        energy = last_step.get('e_0_energy', last_step.get('energy', 0.0))
                        forces = last_step.get('forces')
                        if forces is not None:
                            forces = forces.tolist() if hasattr(forces, 'tolist') else forces
                        result = {
                            "structure": structure.as_dict(),
                            "energy": float(energy),
                            "forces": forces,
                            "stress": None,
                            "step": -1,
                            "total_steps": len(vr.ionic_steps),
                            "source_type": "fallback"
                        }
                        return [result]
        except:
            pass
        return None

def extract_split_from_path(dirname):
    """从路径中提取train/test/fold信息"""
    info = {'split': 'unknown', 'fold': None, 'subdir': None}
    parts = dirname.split('/')
    for part in parts:
        if part.startswith('fold_'):
            info['fold'] = part
        if part in ['train', 'test', 'validation']:
            info['split'] = part
        if part.startswith('md_iter') or part.startswith('vasp_'):
            info['subdir'] = part
    return info

def save_chunk_data(chunk_data, chunk_file):
    """保存单个批次的数据到临时文件"""
    with open(chunk_file, 'w') as f:
        json.dump(chunk_data, f, indent=2)

# ==================== 合并函数（不删除原文件） ====================

def merge_chunks_safe(
    chunk_dir="./chunk_batches",
    output_dir="./merged_data",
    final_name="all_vasp_data.json",
    batch_size=5
):
    """
    安全合并分块文件：不删除原文件，在输出目录生成合并结果
    
    Args:
        chunk_dir: 分块文件所在目录
        output_dir: 合并结果输出目录
        final_name: 最终文件名
        batch_size: 每批合并的文件数
    """
    print("="*60)
    print("分块文件安全合并工具")
    print("="*60)
    print(f"分块目录: {chunk_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最终文件: {final_name}")
    print(f"每批合并: {batch_size} 个文件")
    print("="*60 + "\n")
    
    # 1. 获取所有分块文件
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.json')])
    chunk_files = [os.path.join(chunk_dir, f) for f in chunk_files]
    
    if not chunk_files:
        print(f"❌ 没有找到分块文件！请检查目录: {chunk_dir}")
        return None
    
    print(f"✅ 找到 {len(chunk_files)} 个分块文件\n")
    
    # 2. 显示文件信息
    total_size = 0
    for f in chunk_files:
        size = os.path.getsize(f) / (1024 * 1024)
        total_size += size
        print(f"   {os.path.basename(f)}: {size:.2f} MB")
    print(f"\n   总大小: {total_size:.2f} MB\n")
    
    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 创建临时合并目录
    temp_merge_dir = os.path.join(output_dir, ".temp_merge")
    os.makedirs(temp_merge_dir, exist_ok=True)
    
    # 5. 第一轮：分批合并
    print("🔄 第一轮合并...")
    merged_chunks = []
    batch_info = []
    
    for i in tqdm(range(0, len(chunk_files), batch_size), desc="分批合并"):
        batch_files = chunk_files[i:i+batch_size]
        batch_data = []
        
        for chunk_file in batch_files:
            with open(chunk_file, 'r') as f:
                data = json.load(f)
                batch_data.extend(data)
        
        if batch_data:
            merge_idx = len(merged_chunks)
            merge_file = os.path.join(temp_merge_dir, f"merged_{merge_idx:04d}.json")
            with open(merge_file, 'w') as f:
                json.dump(batch_data, f, indent=2)
            merged_chunks.append(merge_file)
            batch_info.append((f"批次{i//batch_size + 1}", len(batch_data)))
            
            # 释放内存
            batch_data = None
            gc.collect()
    
    print(f"\n   第一轮完成: {len(merged_chunks)} 个中间文件")
    
    # 6. 继续合并直到只剩一个文件
    round_num = 2
    while len(merged_chunks) > 1:
        print(f"\n🔄 第 {round_num} 轮合并 ({len(merged_chunks)} 个文件)...")
        new_merged = []
        
        for i in tqdm(range(0, len(merged_chunks), batch_size), desc=f"第{round_num}轮合并"):
            batch_files = merged_chunks[i:i+batch_size]
            batch_data = []
            
            for chunk_file in batch_files:
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'r') as f:
                        data = json.load(f)
                        batch_data.extend(data)
                    os.remove(chunk_file)
            
            if batch_data:
                merge_idx = len(new_merged)
                merge_file = os.path.join(temp_merge_dir, f"merged_{merge_idx:04d}.json")
                with open(merge_file, 'w') as f:
                    json.dump(batch_data, f, indent=2)
                new_merged.append(merge_file)
                
                batch_data = None
                gc.collect()
        
        merged_chunks = new_merged
        round_num += 1
    
    # 7. 最终输出
    if merged_chunks:
        final_merge_file = merged_chunks[0]
        if os.path.exists(final_merge_file):
            # 读取最终数据
            with open(final_merge_file, 'r') as f:
                final_data = json.load(f)
                final_count = len(final_data)
            
            # 保存到最终位置
            final_output = os.path.join(output_dir, final_name)
            os.rename(final_merge_file, final_output)
            
            # 清理临时目录
            try:
                os.rmdir(temp_merge_dir)
            except:
                pass
            
            file_size = os.path.getsize(final_output) / (1024 * 1024)
            
            print(f"\n{'='*60}")
            print(f"✅ 合并完成!")
            print(f"   总数据条数: {final_count:,}")
            print(f"   最终文件大小: {file_size:.2f} MB")
            print(f"   保存到: {final_output}")
            print(f"\n📁 原始分块文件保留在: {chunk_dir}")
            print(f"{'='*60}")
            
            # 保存合并日志
            log_file = os.path.join(output_dir, "merge_log.txt")
            with open(log_file, 'w') as f:
                f.write(f"合并时间: {datetime.now()}\n")
                f.write(f"分块目录: {chunk_dir}\n")
                f.write(f"分块数量: {len(chunk_files)}\n")
                f.write(f"总数据条数: {final_count}\n")
                f.write(f"最终文件: {final_output}\n")
                f.write(f"文件大小: {file_size:.2f} MB\n")
                f.write(f"\n批次信息:\n")
                for name, count in batch_info:
                    f.write(f"  {name}: {count} 条\n")
            
            print(f"📝 合并日志保存到: {log_file}")
            
            return final_data
    
    print("\n⚠️ 没有数据被合并!")
    return None

# ==================== 主处理函数 ====================

def process_all_data_batch_safe(
    base_path, 
    chunk_dir="./chunk_batches",
    output_dir="./merged_data",
    final_output="all_vasp_data.json",
    chunk_size=20,
    max_total_dirs=None,
):
    """
    分批处理VASP数据，保留每个批次的JSON文件
    """
    print("="*60)
    print("VASP数据分批处理器 (安全版)")
    print("="*60)
    print(f"数据路径: {base_path}")
    print(f"每批处理: {chunk_size} 个目录")
    print(f"分块保存目录: {chunk_dir}")
    print(f"最终合并目录: {output_dir}")
    print(f"AIMD数据: 全部保留，不采样")
    print("="*60 + "\n")
    
    # 创建目录
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 收集所有目录
    print("📁 扫描目录...")
    outcar_files = glob.glob(os.path.join(base_path, "**", "OUTCAR"), recursive=True)
    outcar_files = [f for f in outcar_files if 'archive' not in f]
    all_dirs = list(set([os.path.dirname(f) for f in outcar_files]))
    
    if max_total_dirs:
        all_dirs = all_dirs[:max_total_dirs]
    
    total_dirs = len(all_dirs)
    print(f"找到 {total_dirs} 个VASP计算目录\n")
    
    # 2. 统计信息
    stats = {'single_point': 0, 'aimd': 0, 'fallback': 0, 'failed': 0}
    chunk_files = []
    
    # 3. 分批次处理
    num_chunks = (total_dirs + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="处理批次"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_dirs)
        chunk_dirs = all_dirs[start_idx:end_idx]
        chunk_data = []
        
        for dirname in chunk_dirs:
            data = load_vasp_data(dirname)
            
            if data is None:
                stats['failed'] += 1
                continue
            
            if data and len(data) > 0:
                source_type = data[0].get('source_type', 'unknown')
                if source_type == 'single_point':
                    stats['single_point'] += 1
                elif source_type == 'aimd_trajectory':
                    stats['aimd'] += 1
                else:
                    stats['fallback'] += 1
            
            path_info = extract_split_from_path(dirname)
            for item in data:
                item['split'] = path_info['split']
                if path_info['fold']:
                    item['fold'] = path_info['fold']
                if path_info['subdir']:
                    item['subdir'] = path_info['subdir']
                item['source_dir'] = dirname
            
            chunk_data.extend(data)
        
        # 4. 保存本批次到分块文件
        if chunk_data:
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.json")
            save_chunk_data(chunk_data, chunk_file)
            chunk_files.append(chunk_file)
            print(f"\n  ✅ 批次 {chunk_idx+1}/{num_chunks}: {len(chunk_data)} 条数据")
        else:
            print(f"\n  ⚠️ 批次 {chunk_idx+1}/{num_chunks}: 无数据")
        
        # 5. 释放内存
        chunk_data = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 6. 统计
    print(f"\n{'='*60}")
    print(f"📊 处理完成!")
    print(f"   总目录: {total_dirs}")
    print(f"   成功处理: {total_dirs - stats['failed']}")
    print(f"   失败: {stats['failed']}")
    print(f"   AIMD轨迹目录: {stats['aimd']}")
    print(f"   单点能目录: {stats['single_point']}")
    print(f"   备用方法: {stats['fallback']}")
    print(f"   分块文件数: {len(chunk_files)}")
    print(f"   分块保存位置: {chunk_dir}")
    
    # 7. 自动合并
    if chunk_files:
        print(f"\n🔄 开始合并分块文件到 {output_dir} ...")
        final_data = merge_chunks_safe(
            chunk_dir=chunk_dir,
            output_dir=output_dir,
            final_name=final_output,
            batch_size=5
        )
        
        if final_data:
            # 统计分布
            train_count = sum(1 for item in final_data if item.get('split') == 'train')
            test_count = sum(1 for item in final_data if item.get('split') == 'test')
            unknown_count = sum(1 for item in final_data if item.get('split') == 'unknown')
            aimd_count = sum(1 for item in final_data if item.get('source_type') == 'aimd_trajectory')
            sp_count = sum(1 for item in final_data if item.get('source_type') == 'single_point')
            
            print(f"\n📊 数据分布:")
            print(f"   训练集: {train_count:,}")
            print(f"   测试集: {test_count:,}")
            print(f"   未知: {unknown_count:,}")
            print(f"\n📊 数据类型:")
            print(f"   AIMD轨迹: {aimd_count:,} 条")
            print(f"   单点能: {sp_count:,} 条")
    else:
        print("\n⚠️ 没有数据被处理!")
    
    print("="*60)

# ==================== 独立合并入口 ====================

def merge_existing_chunks():
    """
    独立合并已存在的分块文件
    """
    print("="*60)
    print("独立合并模式")
    print("="*60)
    
    chunk_dir = "./chunk_batches"
    output_dir = "./merged_data"
    
    if not os.path.exists(chunk_dir):
        print(f"❌ 分块目录不存在: {chunk_dir}")
        return
    
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.json')]
    if not chunk_files:
        print(f"❌ 没有找到分块文件")
        return
    
    print(f"找到 {len(chunk_files)} 个分块文件")
    
    merge_chunks_safe(
        chunk_dir=chunk_dir,
        output_dir=output_dir,
        final_name="all_vasp_data.json",
        batch_size=5
    )

# ==================== 主程序 ====================

if __name__ == "__main__":
    import sys
    
    # ========== 配置参数 ==========
    BASE_PATH = "/AI4S/Users/zyf/ML_TEST/graphene-EC/DFT_data"
    CHUNK_DIR = "./chunk_batches"      # 分块文件保存目录
    OUTPUT_DIR = "./merged_data"       # 合并结果目录
    FINAL_OUTPUT = "all_vasp_data.json"  # 最终文件名
    CHUNK_SIZE = 20                    # 每批处理的目录数
    
    # ========== 检查运行模式 ==========
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        # 只执行合并
        print("🔧 运行模式: 仅合并\n")
        merge_existing_chunks()
    else:
        # 完整处理 + 合并
        print("🔧 运行模式: 完整处理\n")
        process_all_data_batch_safe(
            base_path=BASE_PATH,
            chunk_dir=CHUNK_DIR,
            output_dir=OUTPUT_DIR,
            final_output=FINAL_OUTPUT,
            chunk_size=CHUNK_SIZE,
            max_total_dirs=None,
        )
