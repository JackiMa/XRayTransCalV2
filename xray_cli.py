#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X射线透射计算程序 - 命令行界面
提供元素查询、化合物/混合物透射计算功能
"""

import os
import sys
import re
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import tempfile
import webbrowser
import matplotlib.pyplot as plt
import subprocess

from xray_model import Element, Elements, DEFAULT_INTERPOLATION_POINTS, PROCESSED_DATA_DIR
from xray_plot import plot_element_cross_sections,plot_mixture_all, plot_compound_all


# 设置默认参数
DEFAULT_E_MIN = 0.001  # 默认最小能量 (MeV)
DEFAULT_E_MAX = 20.0   # 默认最大能量 (MeV)
DEFAULT_POINTS = 1000  # 默认插值点数
DEFAULT_DENSITY = 1.0  # 默认密度 (g/cm³)
DEFAULT_THICKNESS = 1.0  # 默认厚度 (cm)

def init_elements(data_dir: str = PROCESSED_DATA_DIR) -> Elements:
    """初始化元素数据"""
    print(f"尝试从目录 '{data_dir}' 加载元素数据...")
    elements = Elements(data_dir)
    if elements.count() == 0:
        print(f"警告: 未能在 '{data_dir}' 加载任何元素数据")
    else:
        print(f"已加载 {elements.count()} 种元素数据")
    return elements

def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """解析化学式为元素符号和原子数量的字典"""
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    # 处理结果，如果数字为空，则默认为1
    parsed = {}
    for symbol, count in matches:
        count = float(count) if count else 1.0
        parsed[symbol] = parsed.get(symbol, 0) + count
        
    return parsed

def display_element_info(elements: Elements, symbol: str) -> None:
    """显示元素信息"""
    element = elements.get(symbol=symbol)
    if not element:
        print(f"未找到元素: {symbol}")
        return
    
    print(f"\n元素信息: {element.name} ({element.symbol})")
    print(f"  原子序数: {element.z}")
    print(f"  原子量: {element.A:.4f} g/mol")
    
    if element.data is not None:
        min_energy = element.energy_min
        max_energy = element.energy_max
        print(f"  能量范围: {min_energy:.6f} - {max_energy:.6f} MeV")
        energy_points = len(element.data)
        print(f"  数据点数: {energy_points}")
        print(element.data)
    else:
        print("  注意: 未加载详细数据")

def element_lookup_mode(elements: Elements) -> None:
    """元素查询模式"""
    print("\n=== 元素查询模式 ===")
    print("输入元素符号(如 'Fe') 或原子序数(如 '26')查询元素信息")
    print("输入 'list' 显示所有可用元素, 输入 'q' 返回主菜单")

    while True:
        query = input("\n> 请输入元素符号或原子序数: ").strip()
        if not query:
            continue
        if query.lower() == 'q':
            break
        if query.lower() == 'list':
            # 按序数排序显示所有可用元素
            sorted_elements = sorted(elements.elements.values(), key=lambda e: e.z)
            print("\n可用元素列表:")
            for i, element in enumerate(sorted_elements):
                print(f"{element.z:3d} {element.symbol:2s} {element.name:12s}", end="\t")
                if (i + 1) % 5 == 0:  # 每5个元素换行
                    print()
            print("\n")
            continue

        # 尝试查询
        element = None
        if query.isdigit():
            z = int(query)
            element = elements.get(z=z)
        else:
            element = elements.get(symbol=query)

        if not element:
            print(f"未找到匹配的元素: '{query}'")
            continue

        # 显示元素信息
        display_element_info(elements, element.symbol)

        # --- 默认进行绘图 --- #
        if element.data is None or element.data.empty:
            print(f"警告: 元素 {element.symbol} 的详细数据未能加载或数据为空，无法绘制图表。")
            continue # 跳过绘图步骤

        print("\n--- 绘制元素质量衰减系数图 --- ")
        try:
            # 获取能量范围
            e_min_input = input(f"最小能量 (MeV) [{DEFAULT_E_MIN}]: ").strip()
            e_min = float(e_min_input) if e_min_input else DEFAULT_E_MIN

            e_max_input = input(f"最大能量 (MeV) [{DEFAULT_E_MAX}]: ").strip()
            e_max = float(e_max_input) if e_max_input else DEFAULT_E_MAX

            # 获取坐标轴尺度
            scale_prompt = "坐标轴尺度 [log-log] (可选: linear-log, log-linear, linear-linear): "
            scale_input = input(scale_prompt).strip().lower()
            if not scale_input:
                scale_input = "log-log" # Default for element attenuation

            scale_parts = scale_input.split('-')
            if len(scale_parts) == 2 and scale_parts[0] in ['log', 'linear'] and scale_parts[1] in ['log', 'linear']:
                x_scale_choice = scale_parts[0]
                y_scale_choice = scale_parts[1]
            else:
                print(f"警告: 无效的尺度格式 '{scale_input}'。将使用默认值 'log-log'。")
                x_scale_choice = 'log'
                y_scale_choice = 'log'

            # 确保能量输入有效 (检查 element.energy_min/max 是否存在)
            if not hasattr(element, 'energy_min') or not hasattr(element, 'energy_max'):
                 print(f"警告: 元素 {element.symbol} 数据范围信息缺失。")
                 effective_e_min = DEFAULT_E_MIN
                 effective_e_max = DEFAULT_E_MAX
            else:
                effective_e_min = element.energy_min
                effective_e_max = element.energy_max

            if e_min <= 0 and x_scale_choice == 'log':
                e_min_orig = e_min
                e_min = effective_e_min if effective_e_min > 0 else 1e-6
                print(f"警告: 对数 X 轴能量必须为正数, 已从 {e_min_orig} 调整为 {e_min:.3e} MeV")

            if e_max <= e_min:
                e_max = effective_e_max
                if e_max <= e_min: # 再次检查
                    e_max = e_min * 1000 if x_scale_choice == 'log' else e_min + 10.0
                    print(f"警告: 最大能量必须大于最小能量, 已调整为 {e_max:.3e} MeV")
                else:
                    print(f"警告: 最大能量必须大于最小能量, 已重置为数据最大值 {e_max:.3e} MeV")
            elif e_max > effective_e_max:
                 e_max = effective_e_max # Clip to max available energy
                 print(f"提示: 最大能量已限制为数据最大值 {e_max:.3e} MeV")
            if e_min < effective_e_min:
                 e_min = effective_e_min # Clip to min available energy
                 print(f"提示: 最小能量已限制为数据最小值 {e_min:.3e} MeV")


            # 创建和保存临时文件
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, f"{element.symbol}_cross_sections.png")

            # 绘制图表, 传递尺度
            print("\n正在生成图表...")
            fig = plot_element_cross_sections(
                element,
                e_min, e_max,
                points=DEFAULT_POINTS,
                show_plot=False,
                save_path=temp_file,
                return_fig=True,
                x_scale=x_scale_choice,
                y_scale=y_scale_choice
            )

            # 使用系统默认的图片查看器打开图像文件
            if fig is not None and os.path.exists(temp_file):
                print(f"图表已生成: {temp_file}")
                try:
                    if sys.platform == "win32":
                        os.startfile(os.path.abspath(temp_file))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", os.path.abspath(temp_file)])
                    else:
                        subprocess.Popen(["xdg-open", os.path.abspath(temp_file)])
                except NameError:
                     print("无法自动打开文件: 需要导入 'subprocess' 模块。")
                except Exception as web_err:
                    print(f"无法自动打开图表文件: {web_err}")
            elif fig is None:
                 print(f"注意: 绘图函数未能生成有效的图表对象。可能是因为数据问题。")
            else: # fig is not None but file doesn't exist
                 print(f"注意: 图表文件未成功保存。")

            # 清理 Figure 对象以释放内存
            if fig is not None:
                plt.close(fig)

        except ValueError as ve:
            print(f"输入错误: {ve}. 请输入有效的数字。")
        except Exception as e:
            print(f"生成图表时发生意外错误: {e}")
            import traceback
            traceback.print_exc()

def validate_formula(elements: Elements, formula: str) -> bool:
    """验证化学式是否有效，所有元素是否都存在于数据库中"""
    try:
        parsed = elements.parse_chemical_formula(formula)
        if not parsed:
            print(f"化学式 '{formula}' 格式无效")
            return False
            
        return True
    except Exception as e:
        print(f"验证化学式 '{formula}' 时出错: {e}")
        return False

def calc_display_mass_fractions(elements: Elements, formula: str) -> None:
    """计算并显示化合物的质量分数"""
    try:
        # 计算质量分数
        mass_fractions = elements.calculate_mass_fractions(formula)

        if not mass_fractions:
            print("无法计算质量分数 (可能是化学式无效或包含未知元素)")
            return

        # 计算总分子量 (可以从 mass_fractions 反推，或者在 calculate_mass_fractions 中返回)
        total_weight = 0
        element_counts = elements.parse_chemical_formula(formula)
        if element_counts:
            for symbol, count in element_counts:
                element = elements.get(symbol=symbol)
                if element:
                    total_weight += element.A * count

        # 显示结果
        print(f"\n化合物: {formula}")
        print(f"近似分子量: {total_weight:.4f} g/mol")
        print("质量分数:")
        for symbol, fraction in mass_fractions.items():
            element = elements.get(symbol=symbol)
            print(f"  {symbol} ({element.name if element else '未知'}): {fraction*100:.4f}%")
    except Exception as e:
        print(f"计算质量分数时出错: {e}")

def compound_calculation_mode(elements: Elements) -> None:
    """化合物计算模式"""
    print("\n=== 化合物计算模式 ===")
    print("输入化学式(如 'H2O', 'CaCO3'), 计算质量分数并绘制相关图表")
    print("输入 'q' 返回主菜单")

    while True:
        formula = input("\n> 请输入化学式: ").strip()
        if not formula:
            continue
        if formula.lower() == 'q':
            break

        # 验证化学式
        if not validate_formula(elements, formula):
            continue

        # 计算并显示质量分数
        calc_display_mass_fractions(elements, formula)

        # --- 默认进行绘图 --- #
        print("\n--- 准备绘制化合物相关图表 --- ")
        try:
            # 获取能量范围
            e_min_input = input(f"最小能量 (MeV) [{DEFAULT_E_MIN}]: ").strip()
            e_min = float(e_min_input) if e_min_input else DEFAULT_E_MIN

            e_max_input = input(f"最大能量 (MeV) [{DEFAULT_E_MAX}]: ").strip()
            e_max = float(e_max_input) if e_max_input else DEFAULT_E_MAX

            # 获取密度和厚度
            density_input = input(f"密度 (g/cm³) [{DEFAULT_DENSITY}]: ").strip()
            density = float(density_input) if density_input else DEFAULT_DENSITY

            thickness_input = input(f"厚度 (cm) [{DEFAULT_THICKNESS}]: ").strip()
            thickness = float(thickness_input) if thickness_input else DEFAULT_THICKNESS

            # 获取坐标轴尺度
            scale_prompt = "衰减/效应图坐标轴尺度 [log-log] (可选: linear-log, log-linear, linear-linear): "
            scale_input = input(scale_prompt).strip().lower()
            if not scale_input:
                scale_input = "log-log"

            scale_parts = scale_input.split('-')
            if len(scale_parts) == 2 and scale_parts[0] in ['log', 'linear'] and scale_parts[1] in ['log', 'linear']:
                x_scale_choice = scale_parts[0]
                y_scale_choice = scale_parts[1]
            else:
                print(f"警告: 无效的尺度格式 '{scale_input}'。将使用默认值 'log-log'。")
                x_scale_choice = 'log'
                y_scale_choice = 'log'

            # 设置不同图表的 Y 轴尺度
            y_scale_atten_choice = y_scale_choice
            y_scale_effect_choice = y_scale_choice
            trans_scale_prompt = f"透射率图 Y 轴尺度 [{ 'linear' if y_scale_choice != 'linear' else 'log' }] (可选: linear, log): "
            trans_scale_input = input(trans_scale_prompt).strip().lower()
            if trans_scale_input in ['linear', 'log']:
                 y_scale_trans_choice = trans_scale_input
            else:
                 y_scale_trans_choice = 'linear'

            # 确保输入有效性
            if e_min <= 0 and x_scale_choice == 'log':
                e_min_orig = e_min
                # Attempt to find min energy from compound elements
                compound_elements_symbols = set(el_sym for el_sym, _ in elements.parse_chemical_formula(formula))
                valid_element_objs = [elements.get(symbol=sym) for sym in compound_elements_symbols if elements.get(symbol=sym) and elements.get(symbol=sym).data is not None and hasattr(elements.get(symbol=sym), 'energy_min') and elements.get(symbol=sym).energy_min > 0]
                if valid_element_objs:
                     e_min = min(el.energy_min for el in valid_element_objs)
                else:
                     e_min = 1e-6
                print(f"警告: 对数 X 轴最小能量必须为正数，已从 {e_min_orig} 调整为 {e_min:.3e} MeV")

            if e_max <= e_min:
                e_max = e_min * 1000 if x_scale_choice == 'log' else e_min + 10.0
                print(f"警告: 最大能量必须大于最小能量, 已重置为 {e_max:.3e} MeV")

            if density <= 0:
                density = 1.0
                print(f"警告: 密度必须为正数, 已重置为 {density} g/cm³")

            if thickness <= 0:
                thickness = 1.0
                print(f"警告: 厚度必须为正数, 已重置为 {thickness} cm")

            # 调用绘图 (不再询问)
            print("\n正在生成图表...")
            temp_dir = tempfile.mkdtemp()

            # 绘制所有相关图表, 传递所有尺度
            saved_dir = plot_compound_all(
                elements, formula, e_min, e_max, density, thickness,
                points=DEFAULT_POINTS,
                save_dir=temp_dir,
                x_scale=x_scale_choice,
                y_scale_atten=y_scale_atten_choice,
                y_scale_trans=y_scale_trans_choice,
                y_scale_effect=y_scale_effect_choice
            )

            # 打开文件夹 (检查 saved_dir)
            if saved_dir and os.path.exists(saved_dir):
                print(f"图表已生成，保存在: {saved_dir}")
                try:
                    if sys.platform == "win32":
                        os.startfile(os.path.abspath(saved_dir))
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", os.path.abspath(saved_dir)])
                    else:
                        subprocess.Popen(["xdg-open", os.path.abspath(saved_dir)])
                except NameError:
                     print("无法自动打开文件夹: 需要导入 'subprocess' 模块。")
                except Exception as open_err:
                    print(f"无法自动打开文件夹 '{saved_dir}': {open_err}")
                    print("请手动打开上述目录查看图表。")
            elif saved_dir:
                 print(f"错误: 绘图函数声称已保存到 {saved_dir} 但目录不存在。")
            else:
                 print("错误: 绘图函数未能成功执行或创建保存目录。")

        except ValueError as ve:
            print(f"输入错误: {ve}. 请输入有效的数字。")
        except Exception as e:
            print(f"处理化合物计算或绘图时发生意外错误: {e}")
            import traceback
            traceback.print_exc()

def parse_mixture_input(input_str: str) -> Optional[List[Dict[str, Union[str, float]]]]:
    """
    解析混合物输入，格式: 'Formula1 Prop1 Density1, Formula2 Prop2 Density2, ...'
    返回包含 'formula', 'proportion', 'density' 的字典列表，或在错误时返回 None。
    """
    mixture_definition = []
    components = [p.strip() for p in input_str.split(',') if p.strip()]

    if not components:
        print("错误: 未检测到任何混合物组分。")
        return None

    for i, part in enumerate(components):
        subparts = part.split()
        if len(subparts) != 3:
            print(f"错误: 第 {i+1} 个组分 '{part}' 格式不正确。应为 '化学式 占比 密度'。")
            return None

        formula = subparts[0].strip()
        try:
            proportion = float(subparts[1].strip())
            density = float(subparts[2].strip())

            if proportion < 0:
                print(f"错误: 组分 '{formula}' 的占比 ({proportion}) 不能为负。")
                return None
            if density <= 0:
                print(f"错误: 组分 '{formula}' 的密度 ({density}) 必须为正。")
                return None

            mixture_definition.append({
                'formula': formula,
                'proportion': proportion,
                'density': density
            })
        except ValueError:
            print(f"错误: 无法解析第 {i+1} 个组分 '{part}' 的占比或密度为数字。")
            return None
        except Exception as e: # Catch other potential errors during parsing
             print(f"解析第 {i+1} 个组分 '{part}' 时发生意外错误: {e}")
             return None

    # 基本验证: 检查是否有重复的化学式 (可选，但可能有用)
    formulas_seen = set()
    for item in mixture_definition:
        if item['formula'] in formulas_seen:
            print(f"警告: 化学式 '{item['formula']}' 在输入中重复出现。请检查输入是否正确。")
            # Decide whether to return None or continue based on desired strictness
            # return None
        formulas_seen.add(item['formula'])

    return mixture_definition

def mixture_calculation_mode(elements: Elements) -> None:
    """混合物计算模式"""
    print("\n=== 混合物计算模式 ===")
    print("输入混合物组成, 格式为: 化学式1 占比1 密度1, 化学式2 占比2 密度2, ...")
    print("   示例: CaCO3 0.7 2.71, SiO2 0.3 2.65")
    print("   占比可以是任意正数，内部会进行归一化处理。密度单位 g/cm³。")
    print("输入 'q' 返回主菜单")

    while True:
        mixture_input = input("\n> 请输入混合物组成: ").strip()
        if not mixture_input:
            continue
        if mixture_input.lower() == 'q':
            break

        mixture_definition = parse_mixture_input(mixture_input)

        if mixture_definition is None:
            continue # 解析失败，等待下一次输入

        # 验证每个组分的化学式
        valid_formulas = True
        formulas_to_plot = []
        print("\n验证混合物组分:")
        for item in mixture_definition:
            formula = item['formula']
            if not validate_formula(elements, formula):
                print(f" -> {formula} - 无效或元素数据缺失!")
                valid_formulas = False
            else:
                 print(f" -> {formula} - 有效")
                 formulas_to_plot.append(formula)

        if not valid_formulas or not formulas_to_plot:
            print("错误: 混合物包含无效化学式或没有有效的组分，无法继续。")
            continue

        try:
            # --- 获取能量范围 ---
            e_min_input = input(f"最小能量 (MeV) [{DEFAULT_E_MIN}]: ").strip()
            e_min = float(e_min_input) if e_min_input else DEFAULT_E_MIN

            e_max_input = input(f"最大能量 (MeV) [{DEFAULT_E_MAX}]: ").strip()
            e_max = float(e_max_input) if e_max_input else DEFAULT_E_MAX

            # --- 获取混合物总厚度 ---
            thickness_input = input(f"混合物总厚度 (cm) [{DEFAULT_THICKNESS}]: ").strip()
            mixture_thickness = float(thickness_input) if thickness_input else DEFAULT_THICKNESS

            # --- 获取坐标轴尺度 ---
            scale_prompt = "坐标轴尺度 [log-log] (可选: linear-log, log-linear, linear-linear): "
            scale_input = input(scale_prompt).strip().lower()
            if not scale_input:
                scale_input = "log-log" # Default for attenuation/effects

            scale_parts = scale_input.split('-')
            if len(scale_parts) == 2 and scale_parts[0] in ['log', 'linear'] and scale_parts[1] in ['log', 'linear']:
                x_scale_choice = scale_parts[0]
                y_scale_choice = scale_parts[1]
            else:
                print(f"警告: 无效的尺度格式 '{scale_input}'。将使用默认值 'log-log'。")
                x_scale_choice = 'log'
                y_scale_choice = 'log'

            # --- 设置不同图表的 Y 轴尺度 ---
            # 衰减和效应图使用用户选择或默认的 Y 尺度
            y_scale_atten_choice = y_scale_choice
            y_scale_effect_choice = y_scale_choice
            # 透射率图总是默认为 linear，但允许用户覆盖
            trans_scale_prompt = f"透射率图 Y 轴尺度 [{ 'linear' if y_scale_choice != 'linear' else 'log' }] (可选: linear, log): " # Suggest opposite of general choice
            trans_scale_input = input(trans_scale_prompt).strip().lower()
            if trans_scale_input in ['linear', 'log']:
                 y_scale_trans_choice = trans_scale_input
            else:
                 y_scale_trans_choice = 'linear' # Default transmission to linear

            # --- 输入有效性检查 ---
            if e_min <= 0 and x_scale_choice == 'log':
                e_min_orig = e_min
                # Attempt to find a valid minimum from components
                mixture_elements_symbols = set(el_sym for item in mixture_definition for el_sym, _ in elements.parse_chemical_formula(item['formula']))
                valid_element_objs = [elements.get(symbol=sym) for sym in mixture_elements_symbols if elements.get(symbol=sym) and elements.get(symbol=sym).data is not None and hasattr(elements.get(symbol=sym), 'energy_min') and elements.get(symbol=sym).energy_min > 0]
                if valid_element_objs:
                    e_min = min(el.energy_min for el in valid_element_objs)
                else:
                    e_min = 1e-6 # Absolute fallback
                print(f"警告: 对数 X 轴最小能量必须为正数，已从 {e_min_orig} 调整为 {e_min:.3e} MeV")

            if e_max <= e_min:
                e_max = e_min * 1000 if x_scale_choice == 'log' else e_min + 10.0 # Adjust based on scale
                print(f"警告: 最大能量必须大于最小能量, 已重置为 {e_max:.3e} MeV")

            if mixture_thickness <= 0:
                mixture_thickness = 1.0
                print(f"警告: 厚度必须为正数, 已重置为 {mixture_thickness} cm")

            # --- 调用绘图 (不再询问，直接绘图) ---
            print("\n正在生成图表...")
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()

            # 绘制所有相关图表
            saved_dir = plot_mixture_all(
                elements,
                mixture_definition, # Correctly parsed list of dicts
                e_min,
                e_max,
                mixture_thickness=mixture_thickness, # Pass total thickness
                points=DEFAULT_POINTS,
                save_dir=temp_dir,
                x_scale=x_scale_choice,
                y_scale_atten=y_scale_atten_choice,
                y_scale_trans=y_scale_trans_choice,
                y_scale_effect=y_scale_effect_choice,
                mixture_density_override=None # Allow density calculation by default
            )

            # --- 打开文件夹 (检查 saved_dir) ---
            if saved_dir and os.path.exists(saved_dir):
                print(f"图表已生成，保存在: {saved_dir}")
                try:
                    # Use os.startfile on Windows, 'open' on macOS/Linux
                    if sys.platform == "win32":
                        # webbrowser.open('file://' + os.path.abspath(saved_dir)) # webbrowser might not open folders well
                        os.startfile(os.path.abspath(saved_dir))
                    elif sys.platform == "darwin": # macOS
                        subprocess.Popen(["open", os.path.abspath(saved_dir)])
                    else: # Linux and other POSIX
                        subprocess.Popen(["xdg-open", os.path.abspath(saved_dir)])
                except NameError: # If subprocess wasn't imported
                     print("无法自动打开文件夹: 需要导入 'subprocess' 模块。")
                except Exception as open_err:
                    print(f"无法自动打开文件夹 '{saved_dir}': {open_err}")
                    print("请手动打开上述目录查看图表。")
            elif saved_dir: # Directory path was returned but doesn't exist?
                 print(f"错误: 绘图函数声称已保存到 {saved_dir} 但目录不存在。")
            else: # saved_dir was None
                 print("错误: 绘图函数未能成功执行或创建保存目录。")

        except ValueError as ve:
             print(f"输入错误: {ve}. 请输入有效的数字。")
        except Exception as e:
            print(f"处理混合物计算或绘图时发生意外错误: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

def main_menu() -> None:
    """主菜单"""
    # 初始化元素数据
    elements = init_elements()
    if elements.count() == 0:
        print("\n*** 无法加载元素数据，程序无法继续运行。 ***")
        print("请检查数据目录和文件格式。")
        return

    while True:
        print("\n==============================")
        print("   X射线透射计算程序 v1.1   ")
        print("==============================")
        print("1. 元素查询")
        print("2. 化合物计算")
        print("3. 混合物计算")
        print("0. 退出程序")
        
        choice = input("请选择功能: ").strip()
        
        if choice == '1':
            element_lookup_mode(elements)
        elif choice == '2':
            compound_calculation_mode(elements)
        elif choice == '3':
            mixture_calculation_mode(elements)
        elif choice == '0':
            print("感谢使用，再见！")
            break
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序意外终止: {e}")
        import traceback
        traceback.print_exc() 