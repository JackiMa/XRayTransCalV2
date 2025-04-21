#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X射线数据模型
用于处理元素和化合物的X射线吸收/透射数据
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
import traceback
from typing import List, Dict, Tuple, Union, Optional, Any

# 配置
PROCESSED_DATA_DIR = 'processed_nist_xcom_data'  # 处理后的数据目录
DEFAULT_ENERGY_MIN = 0.001  # MeV
DEFAULT_ENERGY_MAX = 1.0E5  # MeV (100 GeV)
DEFAULT_INTERPOLATION_POINTS = 1000  # 默认插值点数

# 常量
COLUMN_NAMES = [
    'Energy (MeV)',          # 光子能量
    'Coherent (cm2/g)',      # 相干散射
    'Incoherent (cm2/g)',    # 非相干散射
    'Photoelectric (cm2/g)', # 光电吸收
    'Nuclear (cm2/g)',       # 核场对产生
    'Electron (cm2/g)',      # 电子场对产生
    'Total_w_coh (cm2/g)',   # 总质量衰减系数 (含相干散射)
    'Total_wo_coh (cm2/g)'   # 总质量衰减系数 (不含相干散射)
]

# 原子质量数据 (近似值, g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81, 
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 
    'Ga': 69.723, 'Ge': 72.63, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 
    'Nb': 92.906, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91, 
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71, 
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 
    'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 
    'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 
    'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 
    'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59, 
    'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 'At': 210.0, 
    'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04, 
    'Pa': 231.04, 'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 
    'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 
    'Md': 258.0, 'No': 259.0, 'Lr': 262.0
}

class Element:
    """元素类，表示具有X射线衰减数据和方法的化学元素。"""
    
    def __init__(self, filepath: str = None, z: int = None, symbol: str = None, name: str = None):
        """初始化元素对象"""
        self.filepath = filepath
        self.z = z
        self.symbol = symbol
        self.name = name
        self.metadata = {}
        self.data = None
        self.energy_min = DEFAULT_ENERGY_MIN
        self.energy_max = DEFAULT_ENERGY_MAX
        self._interpolators: Dict[str, interp1d] = {}
        self.A = ATOMIC_WEIGHTS.get(symbol, 0.0) # 添加原子量
        self._is_loaded_successfully = False
        
        if filepath:
            self._load_from_file(filepath)
    
    def _load_from_file(self, filepath: str) -> None:
        """从处理后的数据文件加载元素数据，并处理特殊行。"""
        self._is_loaded_successfully = False
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                warnings.warn(f"文件不存在或为空: {filepath}")
                return

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 解析第一行元数据
            if lines and lines[0].startswith('#'):
                metadata_str = lines[0][1:].strip()
                metadata_pattern = r'([^=,]+)=([^,]+)(?:,|$)'
                matches = re.findall(metadata_pattern, metadata_str)
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    try:
                        if key == 'Z': self.z = int(value)
                        elif key == 'Symbol': self.symbol = value; self.A = ATOMIC_WEIGHTS.get(value, 0.0)
                        elif key == 'Name': self.name = value
                        elif key in ['Z/A', 'I', 'Density']: value = float(value)
                    except ValueError: pass
                    self.metadata[key] = value

            # --- 更健壮的数据行查找逻辑 ---
            data_lines = []
            header_found = False
            data_started = False
            # 用于匹配数据行的模式：以数字开头（可以是科学计数法），后面跟至少7个数字列
            data_pattern = re.compile(r'^\s*\d+\.\d+E[+-]\d+.*')
            shell_pattern = re.compile(r'^\s*\d+\s+[A-Z]+\d*') # 匹配如 "90 N3", "95 L1" 等

            for line in lines[1:]: # 从第二行开始处理
                line_stripped = line.strip()

                if not line_stripped: continue # 跳过空行

                # 查找明确的标题行结束标志 (包含多个 cm2/g 的行)
                # 或者简单地查找包含 "FIELD" 和 "SCATT." 的行作为标题结束
                if not header_found and ('FIELD' in line_stripped and 'SCATT.' in line_stripped):
                     header_found = True
                     continue # 跳过这行标题

                # 在找到标题行之后，开始寻找数据
                if header_found:
                     # 检查是否是壳层标记行，如果是则跳过
                     if shell_pattern.match(line_stripped):
                         continue

                     # 检查是否是有效数据行
                     if data_pattern.match(line_stripped):
                         parts = line_stripped.split()
                         # 确保分割后至少有8个部分，并且第一个部分是数字
                         if len(parts) >= 8:
                             try:
                                 # 尝试将第一个元素转换为浮点数以确认是数据
                                 float(parts[0])
                                 data_lines.append(line_stripped)
                                 data_started = True
                             except ValueError:
                                 # 如果第一个部分不是数字，则不是数据行
                                 pass
                     # 如果已经开始记录数据，但遇到非数据行（可能是文件末尾的其他信息），则停止
                     elif data_started:
                         # 考虑是否需要更灵活的停止条件，暂时保留 break
                         # pass # 或者继续寻找，以防数据中有非标准行
                          break # 假设数据块已结束

            if not data_lines:
                warnings.warn(f"在 {filepath} 中未找到有效的数据行。")
                return

            # --- 使用 Pandas 处理收集到的数据行 ---
            from io import StringIO # 确保导入 StringIO
            data_io = StringIO("\n".join(data_lines))

            try:
                self.data = pd.read_csv(
                    data_io,
                    sep=r'\s+',
                    header=None,
                    names=COLUMN_NAMES,
                    engine='python'
                )

                # 数据清理和转换
                self.data = self.data[pd.to_numeric(self.data['Energy (MeV)'], errors='coerce').notna()]
                if self.data.empty:
                    warnings.warn(f"从 {filepath} 解析的数据无效或为空。")
                    self.data = None
                    return

                for col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data.dropna(inplace=True)

                if self.data.empty: # 再次检查清理后是否为空
                     warnings.warn(f"清理后从 {filepath} 加载的数据为空。")
                     self.data = None
                     return

                # 成功加载数据
                self.energy_min = self.data['Energy (MeV)'].min()
                self.energy_max = self.data['Energy (MeV)'].max()
                self.data.dropna(inplace=True)
                self._is_loaded_successfully = True
                # print(f"成功加载元素: Z={self.z}, {self.symbol} ({self.name}) ({len(self.data)} 个数据点)")

            except Exception as e:
                warnings.warn(f"处理 {filepath} 数据时出错: {str(e)}")
                print(traceback.format_exc())
                self.data = None

        except Exception as e:
            warnings.warn(f"加载文件 {filepath} 时出错: {str(e)}")
            print(traceback.format_exc())
            self.data = None
            self._is_loaded_successfully = False

    def _create_interpolator(self, column: str) -> None:
        """为指定列创建对数插值函数"""
        if self.data is None or self.data.empty or column not in self.data.columns:
             warnings.warn(f"元素 {self.symbol} 的数据未加载或为空，无法为 {column} 创建插值器。")
             self._interpolators[column] = None
             return

        try:
            energies = self.data['Energy (MeV)'].values
            values = self.data[column].values
            
            # 记录零值位置
            zero_mask = (values == 0)
            
            if np.any(values <= 0):
                min_positive = np.min(values[values > 0]) if np.any(values > 0) else 1e-10
                values = np.maximum(values, min_positive * 0.1)
            log_energies = np.log10(energies)
            log_values = np.log10(values)
            valid_indices = np.isfinite(log_values) & np.isfinite(log_energies)
            
            if np.sum(valid_indices) < 2:
                warnings.warn(f"{self.symbol} 的 {column} 没有足够的有效值用于插值，插值器可能不准确")
                # 还是创建一个简单的插值器，但基于有效数据（即使只有一个点）
                if np.sum(valid_indices) == 1:
                    single_log_e = log_energies[valid_indices][0]
                    single_log_v = log_values[valid_indices][0]
                    interp = lambda log_e: np.full_like(log_e, single_log_v) # 返回常数
                else: # 0 个有效点，返回极小值
                    interp = lambda log_e: np.full_like(log_e, -10.0)
                self._interpolators[column] = interp
                # 存储零值位置
                self._interpolators[f"{column}_zeros"] = zero_mask
            else:
                interp = interp1d(
                    log_energies[valid_indices], 
                    log_values[valid_indices], 
                    kind='linear', 
                    bounds_error=False,
                    fill_value=(log_values[valid_indices][0], log_values[valid_indices][-1])
                )
                self._interpolators[column] = interp
                # 存储零值位置
                self._interpolators[f"{column}_zeros"] = zero_mask
                
        except Exception as e:
            warnings.warn(f"为 {self.symbol} {column} 创建插值器时出错: {e}")
            print(traceback.format_exc())
            self._interpolators[column] = None
            self._interpolators[f"{column}_zeros"] = None

    def get_cross_section(self, energies: Union[float, List[float], np.ndarray], column: str) -> Optional[np.ndarray]:
        """获取指定能量下的特定截面数据"""
        if self.data is None or self.data.empty:
             warnings.warn(f"元素 {self.symbol} 数据未加载，无法获取 {column} 的截面")
             return None

        # 确保插值器已创建或尝试创建
        if column not in self._interpolators:
            self._create_interpolator(column)
        
        # 检查插值器是否可用
        interpolator = self._interpolators.get(column)
        if interpolator is None:
            warnings.warn(f"元素 {self.symbol} 的 {column} 插值器不可用。")
            # 根据能量数量返回None或None数组
            if isinstance(energies, (int, float)): return None
            else: return np.full(np.asarray(energies).shape, np.nan)

        if isinstance(energies, (int, float)):
            energies = np.array([energies])
        elif isinstance(energies, list):
            energies = np.array(energies)
        
        original_shape = energies.shape
        energies = energies.flatten()
        
        # 检查并处理能量范围
        clipped_energies = np.clip(energies, self.energy_min, self.energy_max)
        if np.any(clipped_energies != energies):
             warnings.warn(
                f"部分能量值超出了 {self.symbol} 的数据范围 "
                f"({self.energy_min:.3e} - {self.energy_max:.3e} MeV)，已截断到有效范围"
            )
        # 处理零或负能量
        if np.any(clipped_energies <= 0):
            min_positive_energy = self.energy_min if self.energy_min > 0 else 1e-9
            clipped_energies = np.maximum(clipped_energies, min_positive_energy)
            warnings.warn(f"部分能量值小于或等于零，已调整为最小正能量 {min_positive_energy:.3e} MeV")
        try:
            log_energies = np.log10(clipped_energies)
            log_results = interpolator(log_energies)
            results = np.power(10.0, log_results)
            
            # 从原始数据中识别能量对应的值是否为零
            # 对于对产生效应，低能量区域的值应该全为零（小于特定阈值）
            # 先检查是否存在零值记录
            zero_mask = self._interpolators.get(f"{column}_zeros")
            
            # 特别处理Nuclear和Electron对产生效应，这些效应在低能量区域应该为零
            if column in ['Nuclear (cm2/g)', 'Electron (cm2/g)']:
                # 判断是否在对产生效应的阈值能量以下
                if column == 'Nuclear (cm2/g)':
                    threshold = 1.022  # MeV, 对产生阈值
                elif column == 'Electron (cm2/g)':
                    threshold = 2.044  # MeV, 双对产生阈值
                
                # 在阈值以下的能量位置设置为NaN
                below_threshold = clipped_energies < threshold
                results[below_threshold] = np.nan
            
            # 替换任何非有限值为较小的值
            results[~np.isfinite(results)] = 1e-10
            
            return results.reshape(original_shape)
        except Exception as e:
            warnings.warn(f"计算 {self.symbol} {column} 时出错: {e}")
            print(traceback.format_exc())
            return np.full(original_shape, np.nan)

    def get_total_cross_section(self, energies: Union[float, List[float], np.ndarray], 
                                with_coherent: bool = True) -> Optional[np.ndarray]:
        """获取总质量衰减系数"""
        col = 'Total_w_coh (cm2/g)' if with_coherent else 'Total_wo_coh (cm2/g)'
        return self.get_cross_section(energies, col)

    def get_mu_rho(self, energies: Union[float, List[float], np.ndarray], 
                     with_coherent: bool = True) -> Optional[np.ndarray]:
        """获取总质量衰减系数 μ/ρ"""
        return self.get_total_cross_section(energies, with_coherent)

class Elements:
    """元素集合类，用于对化合物进行计算的方法。"""
    
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR):
        """使用目录中的数据初始化元素集合"""
        self.data_dir = data_dir
        self.elements: Dict[int, Element] = {}
        self.symbol_to_z: Dict[str, int] = {}
        self.name_to_z: Dict[str, int] = {}
        self._load_elements(data_dir)
    
    def count(self) -> int:
        """返回已加载的元素数量"""
        return len(self.elements)

    def _load_elements(self, data_dir: str) -> None:
        """从数据目录加载所有元素"""
        file_count = 0
        loaded_count = 0
        error_count = 0
        
        if not os.path.isdir(data_dir):
             print(f"错误: 数据目录 '{data_dir}' 不存在或不是目录。")
             return

        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if not files:
            print(f"警告: 在 {data_dir} 中未找到 .txt 文件。请确保已生成数据。")
            return
        
        print(f"在 {data_dir} 中找到 {len(files)} 个潜在数据文件，开始加载...")
        
        for filename in sorted(files):
            file_count += 1
            filepath = os.path.join(data_dir, filename)
            
            match = re.match(r'(\d+)_([^_]+)_(.+)\.txt', filename)
            if match:
                try:
                    z = int(match.group(1))
                    symbol = match.group(2)
                    name = match.group(3).replace('_', ' ')
                    element = Element(filepath)
                    if element._is_loaded_successfully and element.data is not None:
                        self.elements[z] = element
                        self.symbol_to_z[symbol.upper()] = z
                        self.name_to_z[name.lower()] = z
                        loaded_count += 1
                    else:
                         error_count += 1
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    print(traceback.format_exc())
                    error_count += 1
            else:
                print(f"跳过无效文件名: {filename}")
                error_count += 1
        
        print(f"\n加载完成: {loaded_count} 个元素成功加载, {error_count} 个文件处理失败或无效。")
        
        if loaded_count == 0 and file_count > 0:
            print("\n*** 错误: 没有成功加载任何元素! ***")
            print("请检查:")
            print("1. 目录中的数据文件格式是否正确 (Z_Symbol_Name.txt)")
            print("2. 文件是否包含 'PHOTON ENERGY' 标题行和有效的数字数据")
            print("3. 检查文件权限和编码问题")
            print("4. 查看上面的详细警告和错误信息")

    def get(self, z: int = None, symbol: str = None, name: str = None) -> Optional[Element]:
        """根据原子序数、符号或名称获取元素对象"""
        if z is not None:
            return self.elements.get(z)
        if symbol is not None:
            z_from_symbol = self.symbol_to_z.get(symbol.upper())
            return self.elements.get(z_from_symbol) if z_from_symbol else None
        if name is not None:
            z_from_name = self.name_to_z.get(name.lower())
            return self.elements.get(z_from_name) if z_from_name else None
        return None
    
    def parse_chemical_formula(self, formula: str) -> Optional[List[Tuple[str, int]]]:
        """解析化学式，返回元素符号和对应的原子数"""
        if not formula:
            return None
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)
        # 基础验证：解析出的内容重新组合是否等于原字符串
        if not matches or ''.join([f'{el}{num}' for el, num in matches]) != formula:
             warnings.warn(f"无效的化学式格式: {formula}")
             return None

        elements_in_formula = []
        for symbol, count_str in matches:
            count = int(count_str) if count_str else 1
            # 检查元素是否存在于已加载数据中
            if self.get(symbol=symbol) is None:
                 warnings.warn(f"化学式 {formula} 包含未加载或未知的元素 {symbol}。")
                 return None
            elements_in_formula.append((symbol, count))
        return elements_in_formula

    def calculate_mass_fractions(self, formula: str) -> Dict[str, float]:
        """计算化合物中各元素的质量分数"""
        # 解析化学式以获取元素计数
        element_counts = self.parse_chemical_formula(formula)
        if not element_counts:
            print(f"错误: 无法解析化学式 '{formula}'")
            return {}
            
        # 计算总公式质量和元素质量分数
        total_weight = 0
        element_weights = {}
        
        for symbol, count in element_counts:
            element = self.get(symbol=symbol)
            if not element:
                print(f"错误: 数据库中未找到元素 '{symbol}'")
                return {}
                
            element_weights[symbol] = element.A * count
            total_weight += element_weights[symbol]
            
        # 计算质量分数
        mass_fractions = {
            symbol: weight / total_weight 
            for symbol, weight in element_weights.items()
        }
        
        return mass_fractions

    def calculate_compound_cross_section(self,
                                         formula: str,
                                         energies: Union[float, List[float], np.ndarray],
                                         with_coherent: bool = True) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Calculate the mass attenuation coefficient of a compound and the contributions of each element and each effect.

        Args:
            formula (str): Chemical formula of the compound.
            energies (Union[float, List[float], np.ndarray]): Energy values (MeV).
            with_coherent (bool): Whether to include coherent scattering in the total.

        Returns:
            Optional[Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
            A tuple containing:
            - Total mass attenuation coefficient (mu/rho) array for the compound.
            - Dictionary mapping element symbol to its contribution to mu/rho.
            - Dictionary mapping effect name to its contribution to mu/rho.
            Returns None if calculation fails.
        """
        mass_fractions = self.calculate_mass_fractions(formula)
        if not mass_fractions:
            return None

        if isinstance(energies, (float, int)):
            energies = np.array([energies])
        elif isinstance(energies, list):
            energies = np.array(energies)

        total_mu_rho = np.zeros_like(energies, dtype=float)
        element_contributions = {}
        effect_contributions = {
            'Photoelectric': np.zeros_like(energies, dtype=float),
            'Coherent': np.zeros_like(energies, dtype=float),
            'Incoherent': np.zeros_like(energies, dtype=float),
            'Nuclear': np.zeros_like(energies, dtype=float),
            'Electron': np.zeros_like(energies, dtype=float),
            'Total_w_coh': np.zeros_like(energies, dtype=float), # Keep track of total with coherent separately
            'Total_wo_coh': np.zeros_like(energies, dtype=float) # Keep track of total without coherent separately
        }
        effect_cols = { # Map effect names to column names in Element.data
            'Photoelectric': 'Photoelectric (cm2/g)',
            'Coherent': 'Coherent (cm2/g)',
            'Incoherent': 'Incoherent (cm2/g)',
            'Nuclear': 'Nuclear (cm2/g)',
            'Electron': 'Electron (cm2/g)'
        }

        valid_calculation = False
        for symbol, mass_fraction in mass_fractions.items():
            element = self.get(symbol=symbol)
            if element is None or element.data is None:
                warnings.warn(f"Element {symbol} not found or data not loaded for compound {formula}.")
                # Return None immediately, as compound calculation is impossible
                return None

            element_contribution_total = np.zeros_like(energies, dtype=float)

            for effect, col_name in effect_cols.items():
                mu_rho_element_effect = element.get_cross_section(energies, col_name)
                if mu_rho_element_effect is None or np.any(np.isnan(mu_rho_element_effect)):
                     warnings.warn(f"Could not get valid cross section for {effect} ({col_name}) for element {symbol}.")
                     # Return None if any essential component fails
                     return None

                # Add to compound's effect contribution
                effect_contributions[effect] += mu_rho_element_effect * mass_fraction
                # Add to element's total contribution (respecting with_coherent for the element total)
                if with_coherent or effect != 'Coherent':
                     element_contribution_total += mu_rho_element_effect * mass_fraction

            element_contributions[symbol] = element_contribution_total
            valid_calculation = True # Mark as successful if we process at least one element fully

        if not valid_calculation:
             warnings.warn(f"Could not calculate contributions for any element in {formula}.")
             return None

        # Calculate final totals based on accumulated effect contributions
        effect_contributions['Total_wo_coh'] = (
            effect_contributions['Photoelectric'] +
            effect_contributions['Incoherent'] +
            effect_contributions['Nuclear'] +
            effect_contributions['Electron']
        )
        effect_contributions['Total_w_coh'] = effect_contributions['Total_wo_coh'] + effect_contributions['Coherent']

        # Select the total based on the with_coherent flag for the primary return value
        total_mu_rho = effect_contributions['Total_w_coh'] if with_coherent else effect_contributions['Total_wo_coh']

        # Clean up the effect dictionary to return only base effects
        effects_to_return = {k: v for k, v in effect_contributions.items() if k in effect_cols}

        # Explicitly ensure three items are returned, handling potential NaNs in total
        if total_mu_rho is None or np.all(np.isnan(total_mu_rho)):
             warnings.warn(f"Calculated total mu/rho for {formula} is invalid.")
             # Still return the structure, but total might be NaN
             total_mu_rho = np.full_like(energies, np.nan)

        return total_mu_rho, element_contributions, effects_to_return

    def calculate_compound_transmission(self,
                                        formula: str,
                                        energies: Union[float, List[float], np.ndarray],
                                        density: float,
                                        thickness: float,
                                        with_coherent: bool = True) -> Optional[np.ndarray]:
        """计算X射线通过化合物材料的透射率 (修正后以处理新的 cross_section 返回值)"""
        # Correctly unpack 3 values now, although we only need the first
        compound_results = self.calculate_compound_cross_section(formula, energies, with_coherent)

        if compound_results is None:
            warnings.warn(f"无法计算化合物 {formula} 的有效衰减系数，透射率计算失败。")
            if isinstance(energies, (int, float)): return None
            else: return np.full(np.asarray(energies).shape, np.nan)

        # Unpack 3, use only the first (total mu_rho)
        compound_mu_rho, _, _ = compound_results

        # Check if the unpacked mu_rho is valid
        if compound_mu_rho is None or np.any(np.isnan(compound_mu_rho)):
            warnings.warn(f"从 cross_section 获取的化合物 {formula} 总衰减系数无效，透射率计算失败。")
            if isinstance(energies, (int, float)): return None
            else: return np.full(np.asarray(energies).shape, np.nan)

        # Ensure thickness and density are valid
        if thickness < 0 or density <= 0:
            warnings.warn(f"无效的厚度 ({thickness}) 或密度 ({density}) 用于透射率计算。")
            if isinstance(energies, (int, float)): return None
            else: return np.full(np.asarray(energies).shape, np.nan)

        try:
            exponent = -compound_mu_rho * density * thickness
            # Use np.exp safely
            transmission_fraction = np.exp(np.clip(exponent, -700, 700)) # Clip exponent to avoid overflow
            return transmission_fraction * 100
        except Exception as e:
            warnings.warn(f"计算化合物透射率指数时出错: {e}")
            if isinstance(energies, (int, float)): return None
            else: return np.full(np.asarray(energies).shape, np.nan)

    # Method added: calculate_mixture_density
    def calculate_mixture_density(self,
                                 mixture_definition: List[Dict[str, Union[str, float]]]) -> Optional[float]:
        """
        Calculate the effective density of a homogeneous mixture based on component
        proportions and individual densities using the formula: 1 / ρ_mix = Σ (w_i / ρ_i).

        Args:
            mixture_definition (List[Dict[str, Union[str, float]]]):
                A list of dictionaries, each defining a component with keys:
                'formula' (str): Chemical formula.
                'proportion' (float): Relative amount (any positive number).
                'density' (float): Density of the pure component (g/cm³).

        Returns:
            Optional[float]: The calculated mixture density (g/cm³), or None if calculation fails.
        """
        if not isinstance(mixture_definition, list) or not all(
                isinstance(item, dict) and 'formula' in item and 'proportion' in item and 'density' in item
                for item in mixture_definition):
            warnings.warn("Invalid mixture_definition format for density calculation.")
            return None

        formulas = []
        proportions = []
        densities = []
        for item in mixture_definition:
            try:
                formula = item['formula']
                prop = float(item['proportion'])
                density = float(item['density'])
                if prop < 0 or density <= 0:
                    warnings.warn(f"Invalid proportion (<0) or density (<=0) for {formula}.")
                    return None
                # Check if formula is valid (basic check: try parsing)
                if self.parse_chemical_formula(formula) is None:
                     warnings.warn(f"Invalid chemical formula for density calculation: {formula}")
                     return None

                formulas.append(formula)
                proportions.append(prop)
                densities.append(density)
            except (ValueError, TypeError, KeyError):
                warnings.warn(f"Invalid data in mixture definition item: {item}")
                return None

        total_proportion = sum(proportions)
        if total_proportion <= 0:
            warnings.warn("Total proportion is zero or negative, cannot calculate mixture density.")
            return None

        # Calculate mass fractions
        mass_fractions = [(p / total_proportion) for p in proportions]

        # Calculate sum(w_i / rho_i)
        sum_w_over_rho = 0.0
        for w_i, rho_i in zip(mass_fractions, densities):
            if rho_i <= 0: # Should be caught earlier, but double check
                warnings.warn(f"Component density is zero or negative ({rho_i}).")
                return None
            sum_w_over_rho += w_i / rho_i

        if sum_w_over_rho <= 0:
             warnings.warn("Calculated sum(w_i / rho_i) is zero or negative.")
             return None

        mixture_density = 1.0 / sum_w_over_rho
        return mixture_density

    # Method added: calculate_mixture_cross_section_components
    def calculate_mixture_cross_section_components(self,
                                                   mixture_definition: List[Dict[str, Union[str, float]]],
                                                   energies: Union[float, List[float], np.ndarray],
                                                   with_coherent: bool = True
                                                   ) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Calculate the total mass attenuation coefficient for a mixture, along with contributions
        from each component formula and each interaction effect.

        Args:
            mixture_definition (List[Dict[str, Union[str, float]]]):
                List defining the mixture components. Each dict needs 'formula' and 'proportion'.
                'density' key is ignored here but needed for transmission.
            energies (Union[float, List[float], np.ndarray]): Energy values (MeV).
            with_coherent (bool): Whether to include coherent scattering in the total.

        Returns:
            Optional[Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
            A tuple containing:
            - Total mass attenuation coefficient (mu/rho) array for the mixture.
            - Dictionary mapping component formula to its contribution to mixture mu/rho.
            - Dictionary mapping effect name to its contribution to mixture mu/rho.
            Returns None if calculation fails.
        """
        if not isinstance(mixture_definition, list) or not all(
                isinstance(item, dict) and 'formula' in item and 'proportion' in item
                for item in mixture_definition):
            warnings.warn("Invalid mixture_definition format for cross section calculation.")
            return None

        # --- Normalize Proportions ---
        formulas = []
        proportions = []
        for item in mixture_definition:
             try:
                 formula = item['formula']
                 prop = float(item['proportion'])
                 if prop < 0:
                     warnings.warn(f"Negative proportion for {formula}.")
                     return None
                 formulas.append(formula)
                 proportions.append(prop)
             except (ValueError, TypeError, KeyError):
                 warnings.warn(f"Invalid data in mixture definition item: {item}")
                 return None
        total_proportion = sum(proportions)
        if total_proportion <= 0:
            warnings.warn("Total proportion is zero or negative.")
            return None
        normalized_mixture_data = [{'formula': f, 'mass_fraction': p / total_proportion}
                                  for f, p in zip(formulas, proportions)]

        # --- Initialize results ---
        if isinstance(energies, (float, int)): energies = np.array([energies])
        elif isinstance(energies, list): energies = np.array(energies)

        mixture_total_mu_rho = np.zeros_like(energies, dtype=float)
        formula_contributions = {} # Contribution of each formula to the mixture total
        mixture_effect_contributions = { # Contribution of each effect to the mixture total
            'Photoelectric': np.zeros_like(energies, dtype=float),
            'Coherent': np.zeros_like(energies, dtype=float),
            'Incoherent': np.zeros_like(energies, dtype=float),
            'Nuclear': np.zeros_like(energies, dtype=float),
            'Electron': np.zeros_like(energies, dtype=float)
        }
        calculation_possible = False

        # --- Sum contributions ---
        for item in normalized_mixture_data:
            formula = item['formula']
            mass_fraction = item['mass_fraction']

            # Get compound cross sections (total and effects)
            compound_results = self.calculate_compound_cross_section(formula, energies, with_coherent=True) # Always calculate with coherent internally first

            if compound_results is None:
                warnings.warn(f"Failed to get cross sections for compound {formula}. Mixture calculation incomplete.")
                # Cannot proceed accurately if one component fails
                return None

            # Unpack results from compound calculation
            compound_total_w_coh, _, compound_effects = compound_results

            # Calculate the contribution of this formula to the mixture total
            formula_contribution_total = compound_total_w_coh if with_coherent else (compound_total_w_coh - compound_effects.get('Coherent', 0.0))
            formula_contributions[formula] = formula_contribution_total * mass_fraction

            # Add the weighted effect contributions to the mixture totals
            for effect_name, effect_values in compound_effects.items():
                 if effect_name in mixture_effect_contributions:
                     mixture_effect_contributions[effect_name] += effect_values * mass_fraction

            calculation_possible = True

        if not calculation_possible:
            warnings.warn("Could not calculate contribution for any component in the mixture.")
            return None

        # Calculate the final mixture total based on summed effects
        mixture_total_wo_coh = (mixture_effect_contributions['Photoelectric'] +
                                mixture_effect_contributions['Incoherent'] +
                                mixture_effect_contributions['Nuclear'] +
                                mixture_effect_contributions['Electron'])
        mixture_total_w_coh = mixture_total_wo_coh + mixture_effect_contributions['Coherent']
        final_mixture_total = mixture_total_w_coh if with_coherent else mixture_total_wo_coh

        return final_mixture_total, formula_contributions, mixture_effect_contributions

    # Method added: calculate_mixture_transmission
    def calculate_mixture_transmission(self,
                                       mixture_definition: List[Dict[str, Union[str, float]]],
                                       energies: Union[float, List[float], np.ndarray],
                                       total_thickness: float,
                                       mixture_density: Optional[float] = None,
                                       with_coherent: bool = True) -> Optional[np.ndarray]:
        """
        Calculate the transmission rate through a homogeneous mixture.

        Args:
            mixture_definition (List[Dict[str, Union[str, float]]]):
                List defining the mixture components. Each dict needs 'formula',
                'proportion', and 'density' (for density calculation if not provided).
            energies (Union[float, List[float], np.ndarray]): Energy values (MeV).
            total_thickness (float): Total thickness of the mixture layer (cm).
            mixture_density (Optional[float]): Pre-calculated or measured density
                of the mixture (g/cm³). If None, it will be calculated from
                component densities in mixture_definition.
            with_coherent (bool): Whether to include coherent scattering.

        Returns:
            Optional[np.ndarray]: Transmission rate array (%). Returns None if calculation fails.
        """
        # Calculate mixture density if not provided
        if mixture_density is None:
            mixture_density = self.calculate_mixture_density(mixture_definition)
            if mixture_density is None:
                warnings.warn("Failed to calculate mixture density, cannot calculate transmission.")
                return None
        elif mixture_density <= 0:
             warnings.warn(f"Provided mixture density ({mixture_density}) is invalid.")
             return None

        # Calculate mixture mass attenuation coefficient
        mixture_results = self.calculate_mixture_cross_section_components(
            mixture_definition, energies, with_coherent=with_coherent
        )

        if mixture_results is None:
            warnings.warn("Failed to calculate mixture cross section, cannot calculate transmission.")
            return None

        total_mu_rho_mix, _, _ = mixture_results # We only need the total mu/rho

        # Calculate transmission: T(%) = 100 * exp(- (μ/ρ)_mix * ρ_mix * t_total )
        try:
             # Ensure inputs are valid for exp function
            if total_thickness < 0:
                warnings.warn(f"Total thickness ({total_thickness}) cannot be negative.")
                return None

            exponent = -total_mu_rho_mix * mixture_density * total_thickness
            # Handle potential overflow/underflow in exponent - np.exp handles large negative okay
            transmission = 100 * np.exp(exponent)
            return transmission

        except Exception as e:
             warnings.warn(f"Error during transmission calculation: {e}")
             print(traceback.format_exc())
             return None 