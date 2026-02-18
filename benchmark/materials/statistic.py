import json
import glob
import re
import numpy as np

# --- 1. 物理常数映射 (用于估算 HPC 负载) ---
# 常见元素的价电子数 (基于常用赝势配置)
VALENCE_MAP = {
    'H': 1, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7,
    'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
    'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12,
    'Cs': 1, 'Ba': 2, 'La': 3, 'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12,
    'Pb': 4, 'Bi': 5, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7
}

# 平均原子体积 (Angstrom^3)，用于估算晶胞大小
AVG_ATOMIC_VOL = 16.0 

def parse_formula(formula):
    """解析化学式: 'Bi2Te3' -> {'Bi': 2, 'Te': 3}"""
    formula = re.sub(r'\(.*?\)', '', formula).strip() # 去除 (diamond) 等标注
    elements = {}
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    for el, count in matches:
        count = int(count) if count else 1
        elements[el] = elements.get(el, 0) + count
    return elements

def calculate_hpc_metrics(formula, atoms_in_cell):
    """计算 HPC 相关的复杂度指标: 电子数 (矩阵维度) 和 体积 (K点密度)"""
    elements = parse_formula(formula)
    formula_atoms = sum(elements.values())
    
    # 1. 估算体积 (Volume ~ N_atoms * V_avg)
    est_volume = atoms_in_cell * AVG_ATOMIC_VOL
    
    # 2. 计算总价电子数 (Total Electrons)
    # 先算出化学式包含的电子总数，再按晶胞原子数比例缩放
    formula_electrons = sum(VALENCE_MAP.get(el, 4) * count for el, count in elements.items())
    # 缩放因子 = 晶胞原子数 / 化学式原子数
    scale = atoms_in_cell / formula_atoms if formula_atoms > 0 else 1
    total_electrons = formula_electrons * scale
    
    return total_electrons, est_volume, list(elements.keys())

# --- 2. 主处理逻辑 ---
files = glob.glob("*.json")
stats = []
all_elements = set()
all_space_groups = set()

print(f"Processing {len(files)} files...")

for f in files:
    with open(f, 'r') as file:
        data = json.load(file)
        cat = data.get('category', 'Unknown')
        
        for mat in data.get('materials', []):
            info = mat['info']
            formula = info.get('formula', '')
            sg = info.get('space_group', '')
            atoms = info.get('atoms_per_primitive_cell', 1)
            
            # 计算衍生指标
            n_elec, vol, elems = calculate_hpc_metrics(formula, atoms)
            
            stats.append({
                'Category': cat,
                'Formula': formula,
                'SpaceGroup': sg,
                'Atoms': atoms,
                'Electrons': n_elec,
                'Volume': vol
            })
            all_elements.update(elems)
            all_space_groups.add(sg)

# --- 3. 生成 LaTeX 表格所需数据 ---
import pandas as pd
df = pd.DataFrame(stats)

print("-" * 30)
print(">>> BENCHMARK STATISTICS <<<")
print("-" * 30)
print(f"Total Unique Materials: {len(df)}")
print(f"Material Categories:    {df['Category'].nunique()}")
print(f"Unique Elements:        {len(all_elements)}")
print(f"Unique Space Groups:    {len(all_space_groups)}")
print("-" * 30)
print("HPC Complexity Ranges:")
print(f"Atoms per Cell:        {df['Atoms'].min()} ~ {df['Atoms'].max()}")
print(f"Est. Unit Cell Vol:    {df['Volume'].min():.1f} ~ {df['Volume'].max():.1f} A^3")
print(f"Total Electrons:       {df['Electrons'].min():.1f} ~ {df['Electrons'].max():.1f}")
print("-" * 30)