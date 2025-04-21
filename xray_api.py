#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X射线数据分析API
提供X射线衰减和透射计算的RESTful API接口
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import tempfile
import re
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
import datetime

# 导入相关模块
from xray_model import Element, Elements, PROCESSED_DATA_DIR
from xray_plot import (
    plot_element_cross_sections,
    plot_compound_components,
    plot_compound_transmission,
    plot_compound_all,
    plot_mixture_components,
    plot_mixture_transmission,
    plot_mixture_all,
    TMP_PLOTS_DIR,
    save_base64_plot,
    plot_compound_effect_contributions,
    plot_mixture_effect_contributions
)

# 创建临时目录
if not os.path.exists(TMP_PLOTS_DIR):
    os.makedirs(TMP_PLOTS_DIR)

# 创建Flask应用
app = Flask(__name__, static_folder='static')
CORS(app)  # 启用CORS支持跨域请求

# 全局变量
ELEMENTS = None

# 配置
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_POINTS = 1000
DEFAULT_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'plots')

# 确保绘图目录存在
if not os.path.exists(DEFAULT_PLOT_DIR):
    os.makedirs(DEFAULT_PLOT_DIR, exist_ok=True)

# 初始化数据
def initialize_data():
    """在应用启动时初始化元素数据"""
    global ELEMENTS
    if ELEMENTS is None:
        try:
            print("正在加载元素数据...")
            ELEMENTS = Elements(PROCESSED_DATA_DIR)
            print(f"成功加载了 {ELEMENTS.count()} 个元素")
        except Exception as e:
            print(f"加载元素数据出错: {str(e)}")
            print(traceback.format_exc())
            return False
    return True

# 解析输入参数
def parse_input_parameters(params):
    """解析通用输入参数"""
    try:
        e_min = float(params.get('e_min', 0.001))
        e_max = float(params.get('e_max', 100.0))
        points = int(params.get('points', 1000))
        
        # 验证参数
        if e_min <= 0 or e_max <= 0 or e_min >= e_max:
            return None, "能量范围无效: 最小能量和最大能量必须为正数，且最小能量小于最大能量"
        
        if points <= 10:
            return None, "插值点数太少，至少需要10个点"
        
        return {
            'e_min': e_min,
            'e_max': e_max,
            'points': points
        }, None
    except ValueError as e:
        return None, f"参数格式错误: {str(e)}"

def parse_compound_parameters(params):
    """解析化合物相关参数"""
    common_params, error = parse_input_parameters(params)
    if error:
        return None, error
    
    try:
        formula = params.get('formula', '').strip()
        if not formula:
            return None, "缺少化学式参数"
            
        density = float(params.get('density', 1.0))
        thickness = float(params.get('thickness', 1.0))
        
        if density <= 0:
            return None, "密度必须为正数"
        if thickness <= 0:
            return None, "厚度必须为正数"
            
        return {
            **common_params,
            'formula': formula,
            'density': density,
            'thickness': thickness
        }, None
    except ValueError as e:
        return None, f"参数格式错误: {str(e)}"

def parse_mixture_parameters(params):
    """解析混合物相关参数"""
    common_params, error = parse_input_parameters(params)
    if error:
        return None, error
    
    try:
        mixture_data_str = params.get('mixture_data', '').strip()
        if not mixture_data_str:
            return None, "缺少混合物数据参数"
        
        # 尝试解析JSON格式的混合物数据
        try:
            mixture_data = json.loads(mixture_data_str)
            if not isinstance(mixture_data, list):
                return None, "混合物数据必须是列表格式"
            
            # 验证每个混合物项的格式
            for item in mixture_data:
                if not isinstance(item, list) or len(item) != 2:
                    return None, "每个混合物项必须是 [化学式, 质量分数] 格式的列表"
                
                formula, mass_fraction = item
                if not isinstance(formula, str) or not formula:
                    return None, "化学式必须是非空字符串"
                
                try:
                    mass_fraction = float(mass_fraction)
                    if mass_fraction <= 0 or mass_fraction > 1:
                        return None, "质量分数必须在 (0, 1] 范围内"
                except ValueError:
                    return None, "质量分数必须是数字"
        except json.JSONDecodeError:
            return None, "混合物数据格式错误，必须是有效的JSON格式"
        
        density = float(params.get('density', 1.0))
        thickness = float(params.get('thickness', 1.0))
        
        if density <= 0:
            return None, "密度必须为正数"
        if thickness <= 0:
            return None, "厚度必须为正数"
            
        return {
            **common_params,
            'mixture_data': mixture_data,
            'density': density,
            'thickness': thickness
        }, None
    except ValueError as e:
        return None, f"参数格式错误: {str(e)}"

def parse_chemical_input(input_text: str) -> Optional[Union[str, List[Tuple[str, float]]]]:
    """解析化学输入文本，判断是元素、化合物还是混合物
    
    返回值:
        str: 如果是单个元素或化合物
        List[Tuple[str, float]]: 如果是混合物，返回[(化学式, 质量分数), ...]
        None: 如果解析失败
    """
    # 去除首尾空白
    input_text = input_text.strip()
    
    # 检查是否是混合物格式（包含多行或分号分隔）
    if '\n' in input_text or ';' in input_text:
        lines = input_text.replace(';', '\n').split('\n')
        mixture_data = []
        total_mass_fraction = 0.0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试解析每一行 "化学式 质量分数" 或 "化学式 百分比%"
            parts = line.split()
            if len(parts) >= 2:
                formula = parts[0]
                # 尝试解析最后一个部分作为质量分数或百分比
                try:
                    mass_fraction_str = parts[-1]
                    # 检查是否是百分比格式
                    if mass_fraction_str.endswith('%'):
                        mass_fraction = float(mass_fraction_str.rstrip('%')) / 100.0
                    else:
                        mass_fraction = float(mass_fraction_str)
                        
                    # 验证质量分数有效性
                    if mass_fraction <= 0 or mass_fraction > 1:
                        if mass_fraction > 1 and mass_fraction <= 100:
                            # 可能是用户以百分比值输入，但没有百分号
                            mass_fraction = mass_fraction / 100.0
                        else:
                            print(f"警告: 质量分数 {mass_fraction} 无效，必须在 (0, 1] 范围内")
                            continue
                            
                    mixture_data.append((formula, mass_fraction))
                    total_mass_fraction += mass_fraction
                except ValueError:
                    print(f"警告: 无法解析质量分数 '{parts[-1]}'")
                    continue
            else:
                print(f"警告: 无法解析混合物行 '{line}'")
        
        # 验证总质量分数
        if not mixture_data:
            return None
            
        # 如果总质量分数明显不是1，则进行归一化
        if abs(total_mass_fraction - 1.0) > 0.001:
            print(f"警告: 总质量分数 {total_mass_fraction} 不等于1，进行归一化处理")
            mixture_data = [(formula, fraction/total_mass_fraction) for formula, fraction in mixture_data]
            
        return mixture_data
    else:
        # 单个元素或化合物
        # 检查格式，是否包含厚度密度信息
        parts = input_text.split()
        if parts:
            return parts[0]  # 返回第一部分作为化学式
        return None

@app.before_first_request
def before_first_request():
    """在第一个请求到来前初始化元素数据"""
    initialize_data()

# API路由
@app.route('/api/elements', methods=['GET'])
def get_elements():
    """获取所有元素的列表"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    elements_list = []
    for z, element in elements_data.elements.items():
        elements_list.append({
            "z": element.z,
            "symbol": element.symbol,
            "name": element.name,
            "atomic_weight": element.A
        })
    
    return jsonify({"elements": elements_list})

@app.route('/api/elements/<identifier>', methods=['GET'])
def get_element(identifier):
    """获取特定元素的信息"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    element = None
    if identifier.isdigit():
        element = elements_data.get(z=int(identifier))
    else:
        # 尝试作为符号或名称查询
        element = elements_data.get(symbol=identifier)
        if element is None:
            element = elements_data.get(name=identifier)
    
    if element is None:
        return jsonify({"error": f"未找到元素: {identifier}"}), 404
    
    # 返回元素信息
    response = {
        "z": element.z,
        "symbol": element.symbol,
        "name": element.name,
        "atomic_weight": element.A,
        "metadata": element.metadata,
        "energy_range": {
            "min": element.energy_min,
            "max": element.energy_max
        }
    }
    
    return jsonify(response)

@app.route('/api/plot/element', methods=['GET'])
def plot_element():
    """绘制元素质量衰减系数图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 获取参数
    identifier = request.args.get('element', '').strip()
    if not identifier:
        return jsonify({"error": "缺少元素参数"}), 400
    
    # 解析通用参数
    params, error = parse_input_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 查找元素
    element = None
    if identifier.isdigit():
        element = elements_data.get(z=int(identifier))
    else:
        # 尝试作为符号或名称查询
        element = elements_data.get(symbol=identifier)
        if element is None:
            element = elements_data.get(name=identifier)
    
    if element is None:
        return jsonify({"error": f"未找到元素: {identifier}"}), 404
    
    # 生成临时文件路径用于保存图表
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_element_cross_sections(
            element, 
            e_min=params['e_min'], 
            e_max=params['e_max'], 
            points=params['points'], 
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/compound_components', methods=['GET'])
def plot_compound_components_api():
    """绘制化合物元素贡献图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_compound_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_compound_components(
            elements_data,
            formula=params['formula'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/compound_effects', methods=['GET'])
def plot_compound_effects_api():
    """绘制化合物相互作用效应图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_compound_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_compound_effect_contributions(
            elements_data,
            formula=params['formula'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/compound_transmission', methods=['GET'])
def plot_compound_transmission_api():
    """绘制化合物透射率图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_compound_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_compound_transmission(
            elements_data,
            formula=params['formula'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            density=params['density'],
            thickness=params['thickness'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/compound_all', methods=['GET'])
def plot_compound_all_api():
    """绘制并保存所有化合物相关图表"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_compound_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成图表
    try:
        save_dir = plot_compound_all(
            elements_data,
            formula=params['formula'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            density=params['density'],
            thickness=params['thickness'],
            points=params['points']
        )
        
        # 获取生成的文件
        files = []
        for file in os.listdir(save_dir):
            if file.endswith(('.png', '.csv')):
                files.append({
                    "name": file,
                    "path": f"/api/files/{os.path.basename(save_dir)}/{file}",
                    "full_path": os.path.join(save_dir, file)
                })
        
        return jsonify({
            "success": True,
            "directory": os.path.basename(save_dir),
            "files": files
        })
    except Exception as e:
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/mixture_components', methods=['GET'])
def plot_mixture_components_api():
    """绘制混合物组分贡献图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_mixture_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_mixture_components(
            elements_data,
            mixture_data=params['mixture_data'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/mixture_effects', methods=['GET'])
def plot_mixture_effects_api():
    """绘制混合物相互作用效应图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_mixture_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_mixture_effect_contributions(
            elements_data,
            mixture_data=params['mixture_data'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/mixture_transmission', methods=['GET'])
def plot_mixture_transmission_api():
    """绘制混合物透射率图"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_mixture_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成临时文件路径
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=TMP_PLOTS_DIR) as tmp:
        tmp_path = tmp.name
    
    # 绘制图表
    try:
        plot_mixture_transmission(
            elements_data,
            mixture_data=params['mixture_data'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            density=params['density'],
            thickness=params['thickness'],
            points=params['points'],
            save_path=tmp_path
        )
        return send_file(tmp_path, mimetype='image/png')
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/plot/mixture_all', methods=['GET'])
def plot_mixture_all_api():
    """绘制并保存所有混合物相关图表"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 解析参数
    params, error = parse_mixture_parameters(request.args)
    if error:
        return jsonify({"error": error}), 400
    
    # 生成图表
    try:
        save_dir = plot_mixture_all(
            elements_data,
            mixture_data=params['mixture_data'],
            e_min=params['e_min'],
            e_max=params['e_max'],
            density=params['density'],
            thickness=params['thickness'],
            points=params['points']
        )
        
        # 获取生成的文件
        files = []
        for file in os.listdir(save_dir):
            if file.endswith(('.png', '.csv')):
                files.append({
                    "name": file,
                    "path": f"/api/files/{os.path.basename(save_dir)}/{file}",
                    "full_path": os.path.join(save_dir, file)
                })
        
        return jsonify({
            "success": True,
            "directory": os.path.basename(save_dir),
            "files": files
        })
    except Exception as e:
        return jsonify({"error": f"绘图出错: {str(e)}"}), 500

@app.route('/api/files/<directory>/<filename>', methods=['GET'])
def get_file(directory, filename):
    """获取生成的文件"""
    file_path = os.path.join(TMP_PLOTS_DIR, directory, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 404
    
    # 确定MIME类型
    mime_type = 'application/octet-stream'
    if filename.endswith('.png'):
        mime_type = 'image/png'
    elif filename.endswith('.csv'):
        mime_type = 'text/csv'
    
    return send_file(file_path, mimetype=mime_type)

@app.route('/api/analyze', methods=['POST'])
def analyze_input():
    """分析用户输入并返回适当的结果"""
    elements_data = get_elements_data()
    if elements_data is None:
        return jsonify({"error": "初始化元素数据失败"}), 500
    
    # 获取JSON数据
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求格式错误，需要JSON数据"}), 400
    
    input_text = data.get('input', '').strip()
    if not input_text:
        return jsonify({"error": "输入为空"}), 400
    
    # 解析能量、密度、厚度参数
    try:
        e_min = float(data.get('e_min', 0.001))
        e_max = float(data.get('e_max', 100.0))
        density = float(data.get('density', 1.0))
        thickness = float(data.get('thickness', 1.0))
        points = int(data.get('points', 1000))
        
        if e_min <= 0 or e_max <= 0 or e_min >= e_max:
            return jsonify({"error": "能量范围无效"}), 400
        if density <= 0:
            return jsonify({"error": "密度必须大于0"}), 400
        if thickness <= 0:
            return jsonify({"error": "厚度必须大于0"}), 400
    except ValueError:
        return jsonify({"error": "参数格式错误"}), 400
    
    # 解析化学输入
    input_result = parse_chemical_input(input_text)
    if input_result is None:
        return jsonify({"error": "无法解析化学输入"}), 400
    
    # 创建临时目录用于保存图片
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(TMP_PLOTS_DIR, f"results_{timestamp}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 根据输入类型处理
    try:
        if isinstance(input_result, str):
            # 单个元素或化合物
            formula = input_result
            
            # 检查是否是单个元素
            element = elements_data.get(symbol=formula)
            
            if element is not None:
                # 是元素
                element_path = os.path.join(save_dir, f"{formula}_element.png")
                plot_element_cross_sections(
                    element, e_min, e_max, points, save_path=element_path
                )
                
                return jsonify({
                    "type": "element",
                    "formula": formula,
                    "files": [{
                        "name": f"{formula}_element.png",
                        "path": f"/api/files/{os.path.basename(save_dir)}/{formula}_element.png",
                        "description": f"{formula} 元素质量衰减系数"
                    }]
                })
            else:
                # 是化合物
                result_dir = plot_compound_all(
                    elements_data, formula, e_min, e_max, density, thickness, points, save_dir
                )
                
                # 获取生成的文件
                files = []
                for file in os.listdir(result_dir):
                    if file.endswith(('.png', '.csv')):
                        description = ""
                        if 'components' in file:
                            description = f"{formula} 元素组分贡献"
                        elif 'transmission' in file:
                            description = f"{formula} 透射率图"
                        elif 'data' in file:
                            description = f"{formula} 数据表格"
                            
                        files.append({
                            "name": file,
                            "path": f"/api/files/{os.path.basename(result_dir)}/{file}",
                            "description": description
                        })
                
                return jsonify({
                    "type": "compound",
                    "formula": formula,
                    "density": density,
                    "thickness": thickness,
                    "files": files
                })
        else:
            # 混合物
            mixture_data = input_result
            
            result_dir = plot_mixture_all(
                elements_data, mixture_data, e_min, e_max, density, thickness, points, save_dir
            )
            
            # 获取生成的文件
            files = []
            for file in os.listdir(result_dir):
                if file.endswith(('.png', '.csv')):
                    description = ""
                    if 'components' in file:
                        description = "混合物组分贡献"
                    elif 'transmission' in file:
                        description = "混合物透射率图"
                    elif 'data' in file:
                        description = "混合物数据表格"
                        
                    files.append({
                        "name": file,
                        "path": f"/api/files/{os.path.basename(result_dir)}/{file}",
                        "description": description
                    })
            
            # 创建混合物描述用于显示
            mixture_desc = "; ".join([f"{formula} ({fraction*100:.1f}%)" for formula, fraction in mixture_data])
            
            return jsonify({
                "type": "mixture",
                "mixture_desc": mixture_desc,
                "mixture_data": mixture_data,
                "density": density,
                "thickness": thickness,
                "files": files
            })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"处理出错: {str(e)}"}), 500

# 设置静态文件服务（用于服务前端文件）
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    """提供静态文件服务"""
    try:
        # 尝试从静态目录提供文件
        return send_from_directory(app.static_folder, path)
    except Exception:
        # 如果在静态目录找不到文件，尝试提供index.html
        if request.path.startswith('/api/'):
            # 对于API请求，返回JSON格式的404
            return jsonify({"error": f"API路径 {request.path} 不存在"}), 404
        else:
            # 对于其他请求，提供index.html（单页应用的常见做法）
            return send_from_directory(app.static_folder, 'index.html')

# 添加错误处理器确保所有错误返回JSON
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"error": f"API路径 {request.path} 不存在"}), 404
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # 对于API路径，返回JSON格式错误响应
    if request.path.startswith('/api/'):
        return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500
    
    # 对于非API路径的错误，仍然提供index.html
    return send_from_directory(app.static_folder, 'index.html')

# 确保每个请求都能访问已初始化的数据
def get_elements_data():
    """获取全局元素数据对象，如果未初始化则进行初始化"""
    global ELEMENTS
    if ELEMENTS is None:
        initialize_data()
    return ELEMENTS

if __name__ == '__main__':
    # 在应用启动前初始化数据
    initialize_data()
    
    # 设置主机和端口 (优先使用环境变量)
    host = os.environ.get('FLASK_HOST', '0.0.0.0')  # 默认绑定到所有接口，便于Docker使用
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true')
    
    print(f"X射线透射计算器 API 服务启动")
    print(f"已加载 {ELEMENTS.count() if ELEMENTS else 0} 种元素数据")
    print(f"API 服务地址: http://{host}:{port}/")
    print(f"调试模式: {'开启' if debug else '关闭'}")
    
    app.run(host=host, port=port, debug=debug) 