import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import os
import pandas as pd
import time

# Import core modules from the project
from xray_model import Elements
from xray_plot import (
    plot_element_cross_sections,
    plot_compound_components,
    plot_mixture_components,
    plot_compound_transmission,
    plot_mixture_transmission,
    plot_compound_effect_contributions,
    plot_mixture_effect_contributions
)

# Set page configuration
st.set_page_config(
    page_title="X-Ray Transmission Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    
    /* 选项卡样式设置 - 可以自定义修改 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #2a2a2a;  /* 选项卡列表背景色 */
        padding: 10px 10px 0 10px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #3a3a3a;  /* 选项卡背景色 - 更深色调 */
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 0 20px;
        font-weight: bold;
        font-size: 18px;
        min-width: 140px;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.05);
        border: none;
        margin-right: 5px;
        transition: all 0.2s ease;
        color: #f5f5dc;  /* 米黄色字体 */
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a4a4a;  /* 选中选项卡背景色 - 更深色调 */
        border-top: 3px solid #4e8cff;
        box-shadow: 0px -2px 8px rgba(0,0,0,0.1);
        transform: translateY(-5px);
        color: #ffffc0;  /* 选中时淡黄色字体 */
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #454545;  /* 悬停时背景色 - 更深色调 */
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-list"] button:focus {
        box-shadow: none;
    }
    /* 美化标题和内容区域 */
    .stTabs [data-baseweb="tab-content"] {
        background-color: white;
        border-radius: 0 0 10px 10px;
        padding: 20px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }
    /* 美化按钮 */
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #3a78e7;
        transform: translateY(-2px);
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize global variables
if 'elements' not in st.session_state:
    # 直接初始化 Elements 类，它会在构造函数中自动加载元素数据
    st.session_state.elements = Elements()
    # st.info(f"Loaded {len(st.session_state.elements.elements)} element data")

# Create temporary directory (if it doesn't exist)
os.makedirs("tmp_plots", exist_ok=True)

# Helper function: Convert image to base64 encoding to embed in the page
def get_image_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

# Main application interface
def main():
    st.title("X-Ray Transmission Calculator")
    
    # Create three tabs
    tabs = st.tabs(["Elements", "Compounds", "Mixtures"])
    
    # Elements tab
    with tabs[0]:
        st.header("Element X-Ray Transmission Calculation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            element_symbol = st.selectbox(
                "Select Element Symbol",
                options=list(st.session_state.elements.elements.keys()),
                format_func=lambda x: f"{x} - {st.session_state.elements.elements[x].name if x in st.session_state.elements.elements else ''}",
                key="element_symbol"
            )
            
            element = st.session_state.elements.elements[element_symbol] if element_symbol in st.session_state.elements.elements else None
            density_value = 1.0  # 默认值

            if element and hasattr(element, 'metadata') and 'Density' in element.metadata:
                density_value = float(element.metadata['Density'])

            density = st.number_input(
                "Density (g/cm³)",
                min_value=0.0001,
                value=density_value,
                format="%.4f",
                key="element_density"
            )
            
            thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.0001,
                max_value=1000.0,
                value=0.1,
                format="%.4f",
                key="element_thickness"
            )
            
            energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.0001,
                max_value=100000.0,
                value=0.1,
                format="%.4f",
                key="element_energy_min"
            )
            
            energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.0001,
                max_value=100000.0,
                value=20.0,
                format="%.4f",
                key="element_energy_max"
            )
            
            num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key="element_num_points"
            )
            
            plot_type = st.radio(
                "Plot Type",
                options=["Cross Section", "Transmission"],
                index=0,
                key="element_plot_type"
            )
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                if 'element_x_scale' not in st.session_state:
                    st.session_state.element_x_scale = "Log"
                
                x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.element_x_scale == "Log" else 0,  # 默认对数坐标
                    horizontal=True,
                    key="element_x_scale",
                    on_change=lambda: st.session_state.update({"element_auto_redraw": True})
                )
            
            with col2_scale:
                if 'element_y_scale' not in st.session_state:
                    st.session_state.element_y_scale = "Log"
                
                y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.element_y_scale == "Log" else 0,  # 默认对数坐标
                    horizontal=True,
                    key="element_y_scale",
                    on_change=lambda: st.session_state.update({"element_auto_redraw": True})
                )
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                y_min_enabled = st.checkbox("Set Y Min", value=False, key="element_y_min_enabled")
                if y_min_enabled:
                    y_min = st.number_input(
                        "Y Axis Minimum",
                        value=0.001 if y_scale == "Log" else 0.0,
                        format="%.6f",
                        key="element_y_min"
                    )
                else:
                    y_min = None
            
            with col2_yrange:
                y_max_enabled = st.checkbox("Set Y Max", value=False, key="element_y_max_enabled")
                if y_max_enabled:
                    y_max = st.number_input(
                        "Y Axis Maximum",
                        value=1.0,
                        format="%.6f",
                        key="element_y_max"
                    )
                else:
                    y_max = None
            
            calculate_button = st.button("Calculate", key="calculate_element")
            
            # 自动重绘的状态检查
            auto_redraw = False
            if 'element_auto_redraw' in st.session_state and st.session_state.element_auto_redraw:
                auto_redraw = True
                st.session_state.element_auto_redraw = False
        
        with col2:
            if calculate_button or auto_redraw:
                try:
                    element = st.session_state.elements.elements[element_symbol]
                    
                    # 转换坐标轴类型为小写
                    x_scale_value = x_scale.lower()
                    y_scale_value = y_scale.lower()
                    
                    # 存储绘图数据以便下载
                    plot_data = {}
                    
                    if plot_type == "Cross Section":
                        # 绘制元素截面系数图
                        fig = plot_element_cross_sections(
                            element, 
                            e_min=energy_min, 
                            e_max=energy_max, 
                            points=num_points,
                            x_scale=x_scale_value,
                            y_scale=y_scale_value,
                            return_fig=True
                        )
                        if fig:
                            # 应用自定义Y轴范围
                            if y_min_enabled or y_max_enabled:
                                for ax in fig.get_axes():
                                    current_ylim = ax.get_ylim()
                                    new_ymin = y_min if y_min_enabled else current_ylim[0]
                                    new_ymax = y_max if y_max_enabled else current_ylim[1]
                                    ax.set_ylim(new_ymin, new_ymax)
                                    
                            st.pyplot(fig)
                            
                            # 准备下载数据
                            df = pd.DataFrame()
                            energy_range = np.linspace(energy_min, energy_max, num_points)
                            df['Energy (MeV)'] = energy_range
                            
                            for ax in fig.get_axes():
                                for line in ax.get_lines():
                                    label = line.get_label()
                                    if label.startswith('_'): continue  # 跳过辅助线
                                    y_data = line.get_ydata()
                                    df[label] = y_data
                                    plot_data[label] = {'x': energy_range, 'y': y_data}
                            
                            plt.close(fig)
                            
                            # 同时绘制效应贡献图
                            st.subheader("Element Interaction Effects")
                            col_effects1, col_effects2 = st.columns(2)
                            with col_effects1:
                                st.write("主要物理效应包括：光电效应、相干散射、非相干散射、对产生等")
                            with col_effects2:
                                # 额外的效应说明或控制选项可以放在这里
                                pass
                        else:
                            st.error("无法生成图表，请检查参数设置。")
                    else:
                        # 计算透射率
                        energy_range = np.linspace(energy_min, energy_max, num_points)
                        transmission = element.calculate_transmission(
                            energy_range, 
                            thickness=thickness,
                            density=density
                        )
                        
                        # 保存传输率数据用于下载
                        plot_data["Transmission"] = {'x': energy_range, 'y': transmission}
                        
                        # 创建透射率图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(energy_range, transmission, 'b-', linewidth=2)
                        ax.set_xlabel('Photon Energy (MeV)')
                        ax.set_ylabel('Transmission')
                        ax.set_title(f'{element_symbol} - Thickness: {thickness} cm, Density: {density} g/cm³')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(energy_min, energy_max)
                        
                        # 应用自定义Y轴范围或默认范围
                        if y_min_enabled or y_max_enabled:
                            default_ymin = 0.001 if y_scale_value == 'log' else 0.0
                            default_ymax = 1.05
                            new_ymin = y_min if y_min_enabled else default_ymin
                            new_ymax = y_max if y_max_enabled else default_ymax
                            ax.set_ylim(new_ymin, new_ymax)
                        else:
                            ax.set_ylim(0, 1.05)
                        
                        # 设置坐标轴类型
                        ax.set_xscale(x_scale_value)
                        if y_scale_value == 'log':
                            # 对于透射图，对数坐标下需要处理0值
                            ax.set_yscale(y_scale_value)
                            if not y_min_enabled:
                                ax.set_ylim(0.001, 1.05)  # 对数坐标下调整下限，除非用户自定义
                        
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # 显示关键能量点的透射率
                        st.subheader("Key Point Transmission")
                        key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000  # 转换为MeV
                        key_energies = np.array([e for e in key_energies if energy_min <= e <= energy_max])
                        
                        if len(key_energies) > 0:
                            key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                            key_transmissions = [transmission[i] for i in key_indices]
                            
                            data = {
                                "Energy (MeV)": key_energies,
                                "Transmission": [f"{t:.4f}" for t in key_transmissions]
                            }
                            st.table(data)
                    
                    # 优化数据下载功能，合并到一个Excel文件中
                    if plot_data:
                        output = io.BytesIO()
                        writer = pd.ExcelWriter(output, engine='xlsxwriter')
                        
                        if plot_type == "Cross Section":
                            # 对于截面系数图，将所有数据放在一个工作表中
                            df = pd.DataFrame()
                            for label, data in plot_data.items():
                                if 'Energy (MeV)' not in df:
                                    df['Energy (MeV)'] = data['x']
                                df[label] = data['y']
                            df.to_excel(writer, sheet_name='Cross Sections')
                        else:
                            # 对于透射率图，创建透射率工作表
                            transmission_df = pd.DataFrame({
                                'Energy (MeV)': plot_data["Transmission"]['x'],
                                'Transmission': plot_data["Transmission"]['y']
                            })
                            transmission_df.to_excel(writer, sheet_name='Transmission')
                            
                            # 如果有关键点数据，添加关键点工作表
                            if len(key_energies) > 0:
                                key_points_df = pd.DataFrame({
                                    'Energy (MeV)': key_energies,
                                    'Transmission': [t for t in key_transmissions]
                                })
                                key_points_df.to_excel(writer, sheet_name='Key Points')
                        
                        writer.close()
                        output.seek(0)
                        
                        st.download_button(
                            label="下载图表数据",
                            data=output,
                            file_name=f"{element_symbol}_{plot_type.replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
    
    # Compounds tab
    with tabs[1]:
        st.header("Compound X-Ray Transmission Calculation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            compound_formula = st.text_input(
                "Compound Chemical Formula (e.g.: H2O, CaCO3)",
                value="H2O",
                key="compound_formula"
            )
            
            compound_density = st.number_input(
                "Density (g/cm³)",
                min_value=0.0001,
                value=1.0,
                format="%.4f",
                key="compound_density"
            )
            
            compound_thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.0001,
                max_value=1000.0,
                value=0.1,
                format="%.4f",
                key="compound_thickness"
            )
            
            compound_energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=0.1,
                format="%.4f",
                key="compound_energy_min"
            )
            
            compound_energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=20.0,
                format="%.4f",
                key="compound_energy_max"
            )
            
            compound_num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key="compound_num_points"
            )
            
            compound_plot_type = st.radio(
                "Plot Type",
                options=["Component Contribution", "Transmission"],
                index=0,
                key="compound_plot_type"
            )
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                if 'compound_x_scale' not in st.session_state:
                    st.session_state.compound_x_scale = "Log"
                
                compound_x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.compound_x_scale == "Log" else 0,
                    horizontal=True,
                    key="compound_x_scale",
                    on_change=lambda: st.session_state.update({"compound_auto_redraw": True})
                )
            
            with col2_scale:
                if 'compound_y_scale' not in st.session_state:
                    st.session_state.compound_y_scale = "Log"
                
                compound_y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.compound_y_scale == "Log" else 0,
                    horizontal=True,
                    key="compound_y_scale",
                    on_change=lambda: st.session_state.update({"compound_auto_redraw": True})
                )
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                compound_y_min_enabled = st.checkbox("Set Y Min", value=False, key="compound_y_min_enabled")
                if compound_y_min_enabled:
                    compound_y_min = st.number_input(
                        "Y Axis Minimum",
                        value=0.001 if compound_y_scale == "Log" else 0.0,
                        format="%.6f",
                        key="compound_y_min"
                    )
                else:
                    compound_y_min = None
            
            with col2_yrange:
                compound_y_max_enabled = st.checkbox("Set Y Max", value=False, key="compound_y_max_enabled")
                if compound_y_max_enabled:
                    compound_y_max = st.number_input(
                        "Y Axis Maximum",
                        value=1.0,
                        format="%.6f",
                        key="compound_y_max"
                    )
                else:
                    compound_y_max = None
            
            calculate_compound_button = st.button("Calculate", key="calculate_compound")
            
            # 自动重绘的状态检查
            compound_auto_redraw = False
            if 'compound_auto_redraw' in st.session_state and st.session_state.compound_auto_redraw:
                compound_auto_redraw = True
                st.session_state.compound_auto_redraw = False
        
        with col2:
            if calculate_compound_button or compound_auto_redraw:
                try:
                    # 解析化学式
                    formula_elements = st.session_state.elements.parse_chemical_formula(compound_formula)
                    
                    # 获取质量分数
                    mass_fractions = st.session_state.elements.calculate_mass_fractions(compound_formula)
                    
                    # 转换坐标轴类型为小写
                    x_scale_value = compound_x_scale.lower()
                    y_scale_value = compound_y_scale.lower()
                    
                    if not formula_elements:
                        st.error("Cannot parse chemical formula, please check input format.")
                    else:
                        if compound_plot_type == "Component Contribution":
                            # 使用plot_compound_all绘制多个图表
                            st.subheader("化合物成分与透射分析")
                            st.info("正在生成多个图表，包括成分贡献、物理效应和透射率...")
                            
                            col_figs1, col_figs2 = st.columns(2)
                            
                            # 创建一个字典存储所有图表的数据
                            plot_data_all = {}
                            
                            with col_figs1:
                                # 成分贡献图
                                fig = plot_compound_components(
                                    st.session_state.elements,
                                    compound_formula,
                                    e_min=compound_energy_min, 
                                    e_max=compound_energy_max, 
                                    points=compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                if fig:
                                    st.pyplot(fig)
                                    
                                    # 保存成分贡献图数据
                                    energy_range = np.linspace(compound_energy_min, compound_energy_max, compound_num_points)
                                    for ax in fig.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue  # 跳过辅助线
                                            x_data = line.get_xdata()
                                            y_data = line.get_ydata()
                                            plot_data_all[f"Component_{label}"] = {'x': x_data, 'y': y_data}
                                    
                                    plt.close(fig)
                                    st.caption("元素成分贡献图")
                                else:
                                    st.error("无法生成成分贡献图")
                            
                            with col_figs2:
                                # 物理效应图
                                fig = plot_compound_effect_contributions(
                                    st.session_state.elements,
                                    compound_formula,
                                    e_min=compound_energy_min, 
                                    e_max=compound_energy_max, 
                                    points=compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                if fig:
                                    st.pyplot(fig)
                                    
                                    # 保存物理效应图数据
                                    for ax in fig.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue  # 跳过辅助线
                                            x_data = line.get_xdata()
                                            y_data = line.get_ydata()
                                            plot_data_all[f"Effect_{label}"] = {'x': x_data, 'y': y_data}
                                    
                                    plt.close(fig)
                                    st.caption("物理效应贡献图")
                                else:
                                    st.error("无法生成物理效应图")
                            
                            # 透射率图
                            st.subheader("化合物透射率")
                            fig = plot_compound_transmission(
                                st.session_state.elements,
                                compound_formula,
                                e_min=compound_energy_min, 
                                e_max=compound_energy_max, 
                                density=compound_density,
                                thickness=compound_thickness,
                                points=compound_num_points,
                                x_scale=x_scale_value,
                                y_scale='linear',
                                return_fig=True
                            )
                            if fig:
                                st.pyplot(fig)
                                
                                # 保存透射率图数据
                                for ax in fig.get_axes():
                                    for line in ax.get_lines():
                                        label = line.get_label()
                                        if label.startswith('_'): continue  # 跳过辅助线
                                        if not label or label == '_line0': label = 'Transmission'
                                        x_data = line.get_xdata()
                                        y_data = line.get_ydata()
                                        plot_data_all[label] = {'x': x_data, 'y': y_data}
                                
                                plt.close(fig)
                                st.caption("透射率图")
                            else:
                                st.error("无法生成透射率图")
                                
                            # 添加数据下载按钮 - 所有图表数据合并到一个Excel文件
                            if plot_data_all:
                                output = io.BytesIO()
                                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                
                                # 创建三个主工作表
                                component_df = pd.DataFrame()
                                effect_df = pd.DataFrame()
                                transmission_df = pd.DataFrame()
                                
                                # 处理能量范围
                                energy_range = np.linspace(compound_energy_min, compound_energy_max, compound_num_points)
                                
                                # 分配数据到相应工作表
                                for label, data in plot_data_all.items():
                                    if label.startswith("Component_"):
                                        if 'Energy (MeV)' not in component_df:
                                            component_df['Energy (MeV)'] = data['x']
                                        component_df[label.replace("Component_", "")] = data['y']
                                    elif label.startswith("Effect_"):
                                        if 'Energy (MeV)' not in effect_df:
                                            effect_df['Energy (MeV)'] = data['x']
                                        effect_df[label.replace("Effect_", "")] = data['y']
                                    else:
                                        if 'Energy (MeV)' not in transmission_df:
                                            transmission_df['Energy (MeV)'] = data['x']
                                        transmission_df[label] = data['y']
                                
                                # 写入工作表
                                if not component_df.empty:
                                    component_df.to_excel(writer, sheet_name='Components')
                                if not effect_df.empty:
                                    effect_df.to_excel(writer, sheet_name='Effects')
                                if not transmission_df.empty:
                                    transmission_df.to_excel(writer, sheet_name='Transmission')
                                
                                # 创建混合物信息工作表
                                element_symbols = [elem[0] for elem in formula_elements]
                                element_counts = [elem[1] for elem in formula_elements]
                                mixture_info_df = pd.DataFrame({
                                    'Element': element_symbols,
                                    'Count': element_counts,
                                    'Mass Fraction': [mass_fractions.get(elem, 0) for elem in element_symbols],
                                    'Density (g/cm³)': compound_density
                                })
                                mixture_info_df.to_excel(writer, sheet_name='Compound Info')
                                
                                writer.close()
                                output.seek(0)
                                
                                # 生成混合物名称用于文件命名
                                mixture_name = compound_formula.replace(" ", "")
                                
                                st.download_button(
                                    label="下载所有图表数据",
                                    data=output,
                                    file_name=f"{mixture_name}_all_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            # 仅显示透射率图
                            fig = plot_compound_transmission(
                                st.session_state.elements,
                                compound_formula,
                                e_min=compound_energy_min, 
                                e_max=compound_energy_max, 
                                density=compound_density,
                                thickness=compound_thickness,
                                points=compound_num_points,
                                x_scale=x_scale_value,
                                y_scale='linear' if y_scale_value == 'linear' else 'log',
                                return_fig=True
                            )
                            
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                                st.caption("透射率图")
                            else:
                                st.error("无法生成图表，请检查参数设置。")
                            
                            # 显示关键能量点的透射率
                            st.subheader("Key Point Transmission")
                            key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000  # 转换为MeV
                            key_energies = np.array([e for e in key_energies if compound_energy_min <= e <= compound_energy_max])
                            
                            if len(key_energies) > 0:
                                # 计算能量范围
                                energy_range = np.linspace(compound_energy_min, compound_energy_max, compound_num_points)
                                
                                # 计算总截面 - 使用正确的API
                                result = st.session_state.elements.calculate_compound_cross_section(
                                    compound_formula, energy_range
                                )
                                
                                if result is not None:
                                    # 正确解包三元组返回值，只使用第一个元素(总截面)
                                    total_cross_section, _, _ = result
                                    
                                    # 计算透射率
                                    transmission = np.exp(-total_cross_section * compound_density * compound_thickness)
                                    
                                    key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                                    key_transmissions = [transmission[i] for i in key_indices]
                                    
                                    data = {
                                        "Energy (MeV)": key_energies,
                                        "Transmission": [f"{t:.4f}" for t in key_transmissions]
                                    }
                                    st.table(data)
                                    
                                    # 添加数据下载按钮
                                    output = io.BytesIO()
                                    writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                    
                                    # 生成化合物名称用于文件命名
                                    compound_name = compound_formula.replace(" ", "")
                                    
                                    # 保存透射率数据
                                    transmission_df = pd.DataFrame({
                                        'Energy (MeV)': energy_range,
                                        'Transmission': transmission
                                    })
                                    transmission_df.to_excel(writer, sheet_name='Transmission')
                                    
                                    # 保存关键点数据
                                    key_points_df = pd.DataFrame({
                                        'Energy (MeV)': key_energies,
                                        'Transmission': [t for t in key_transmissions]
                                    })
                                    key_points_df.to_excel(writer, sheet_name='Key Points')
                                    
                                    # 保存化合物信息
                                    element_symbols = [elem[0] for elem in formula_elements]
                                    element_counts = [elem[1] for elem in formula_elements]
                                    compound_info_df = pd.DataFrame({
                                        'Element': element_symbols,
                                        'Count': element_counts,
                                        'Mass Fraction': [mass_fractions.get(elem, 0) for elem in element_symbols],
                                        'Density (g/cm³)': compound_density
                                    })
                                    compound_info_df.to_excel(writer, sheet_name='Compound Info')
                                    
                                    writer.close()
                                    output.seek(0)
                                    
                                    st.download_button(
                                        label="下载透射率数据",
                                        data=output,
                                        file_name=f"{compound_name}_transmission.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                else:
                                    st.error("无法计算该化合物的截面系数，请检查化学式。")
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")
    
    # Mixtures tab
    with tabs[2]:
        st.header("Mixture X-Ray Transmission Calculation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Mixture Components")
            
            # 使用session_state跟踪组件
            if 'mixture_components' not in st.session_state:
                st.session_state.mixture_components = [
                    {"formula": "H2O", "weight_percent": 70.0, "density": 1.0}
                ]
            
            # 显示现有组件
            for i, comp in enumerate(st.session_state.mixture_components):
                cols = st.columns([3, 2, 2, 1])
                with cols[0]:
                    st.session_state.mixture_components[i]["formula"] = st.text_input(
                        "Formula",
                        value=comp["formula"],
                        key=f"mix_formula_{i}"
                    )
                with cols[1]:
                    st.session_state.mixture_components[i]["weight_percent"] = st.number_input(
                        "Weight Percent",
                        min_value=0.01,
                        max_value=100.0,
                        value=comp["weight_percent"],
                        format="%.2f",
                        key=f"mix_weight_{i}"
                    )
                with cols[2]:
                    st.session_state.mixture_components[i]["density"] = st.number_input(
                        "Density (g/cm³)",
                        min_value=0.001,
                        value=comp["density"],
                        format="%.4f",
                        key=f"mix_density_{i}"
                    )
                with cols[3]:
                    if st.button("Delete", key=f"del_comp_{i}") and len(st.session_state.mixture_components) > 1:
                        st.session_state.mixture_components.pop(i)
                        st.rerun()
            
            # 添加新组件按钮
            if st.button("Add Component"):
                st.session_state.mixture_components.append(
                    {"formula": "", "weight_percent": 10.0, "density": 1.0}
                )
                st.rerun()
            
            # 归一化权重
            if st.button("Normalize Weights"):
                total_weight = sum(comp["weight_percent"] for comp in st.session_state.mixture_components)
                if total_weight > 0:
                    for i in range(len(st.session_state.mixture_components)):
                        st.session_state.mixture_components[i]["weight_percent"] = (
                            st.session_state.mixture_components[i]["weight_percent"] / total_weight * 100
                        )
                    st.rerun()
            
            # 混合物计算参数
            st.subheader("Calculation Parameters")
            
            mixture_thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.00001,
                max_value=1000.0,
                value=0.1,
                format="%.4f",
                key="mixture_thickness"
            )
            
            mixture_energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=0.1,
                format="%.4f",
                key="mixture_energy_min"
            )
            
            mixture_energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=20.0,
                format="%.1f",
                key="mixture_energy_max"
            )
            
            mixture_num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100,
                key="mixture_num_points"
            )
            
            mixture_plot_type = st.radio(
                "Plot Type",
                options=["Component Contribution", "Transmission"],
                index=0,
                key="mixture_plot_type"
            )
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                if 'mixture_x_scale' not in st.session_state:
                    st.session_state.mixture_x_scale = "Log"
                
                mixture_x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.mixture_x_scale == "Log" else 0,
                    horizontal=True,
                    key="mixture_x_scale",
                    on_change=lambda: st.session_state.update({"mixture_auto_redraw": True})
                )
            
            with col2_scale:
                if 'mixture_y_scale' not in st.session_state:
                    st.session_state.mixture_y_scale = "Log"
                
                mixture_y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.mixture_y_scale == "Log" else 0,
                    horizontal=True,
                    key="mixture_y_scale",
                    on_change=lambda: st.session_state.update({"mixture_auto_redraw": True})
                )
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                mixture_y_min_enabled = st.checkbox("Set Y Min", value=False, key="mixture_y_min_enabled")
                if mixture_y_min_enabled:
                    mixture_y_min = st.number_input(
                        "Y Axis Minimum",
                        value=0.001 if mixture_y_scale == "Log" else 0.0,
                        format="%.6f",
                        key="mixture_y_min"
                    )
                else:
                    mixture_y_min = None
            
            with col2_yrange:
                mixture_y_max_enabled = st.checkbox("Set Y Max", value=False, key="mixture_y_max_enabled")
                if mixture_y_max_enabled:
                    mixture_y_max = st.number_input(
                        "Y Axis Maximum",
                        value=1.0,
                        format="%.6f",
                        key="mixture_y_max"
                    )
                else:
                    mixture_y_max = None
            
            calculate_mixture_button = st.button("Calculate", key="calculate_mixture")
            
            # 自动重绘的状态检查
            mixture_auto_redraw = False
            if 'mixture_auto_redraw' in st.session_state and st.session_state.mixture_auto_redraw:
                mixture_auto_redraw = True
                st.session_state.mixture_auto_redraw = False
        
        with col2:
            if calculate_mixture_button or mixture_auto_redraw:
                try:
                    # 检查有效组件
                    valid_components = [comp for comp in st.session_state.mixture_components 
                                      if comp["formula"].strip() and comp["weight_percent"] > 0]
                    
                    # 转换坐标轴类型为小写
                    x_scale_value = mixture_x_scale.lower()
                    y_scale_value = mixture_y_scale.lower()
                    
                    if not valid_components:
                        st.error("Please add at least one valid mixture component.")
                    else:
                        # 提取组件信息
                        formulas = [comp["formula"] for comp in valid_components]
                        weights = [comp["weight_percent"] for comp in valid_components]
                        densities = [comp["density"] for comp in valid_components]
                        
                        # 归一化权重
                        total_weight = sum(weights)
                        weights = [w / total_weight * 100 for w in weights]
                        
                        # 计算混合物平均密度
                        mixture_density = sum((w/100) * d for w, d in zip(weights, densities))
                        
                        # 显示混合物信息
                        st.subheader("Mixture Information")
                        st.write(f"Mixture Average Density: {mixture_density:.4f} g/cm³")
                        
                        component_data = []
                        for formula, weight, density in zip(formulas, weights, densities):
                            component_data.append({
                                "Formula": formula,
                                "Weight Percent": f"{weight:.2f}%",
                                "Density": f"{density:.4f} g/cm³"
                            })
                        st.table(component_data)
                        
                        if mixture_plot_type == "Component Contribution":
                            # 使用plot_mixture_all显示多个图表
                            st.subheader("混合物成分与透射分析")
                            st.info("正在生成多个图表，包括成分贡献、物理效应和透射率...")
                            
                            col_figs1, col_figs2 = st.columns(2)
                            
                            # 准备混合物定义
                            mixture_definition = [{"formula": f, "proportion": w, "density": d} 
                                                for f, w, d in zip(formulas, weights, densities)]
                            
                            # 创建一个字典存储所有图表的数据
                            plot_data_all = {}
                            
                            with col_figs1:
                                # 成分贡献图
                                fig = plot_mixture_components(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=mixture_energy_min, 
                                    e_max=mixture_energy_max, 
                                    points=mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                if fig:
                                    st.pyplot(fig)
                                    
                                    # 保存成分贡献图数据
                                    for ax in fig.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue  # 跳过辅助线
                                            x_data = line.get_xdata()
                                            y_data = line.get_ydata()
                                            plot_data_all[f"Component_{label}"] = {'x': x_data, 'y': y_data}
                                    
                                    plt.close(fig)
                                    st.caption("混合物成分贡献图")
                                else:
                                    st.error("无法生成成分贡献图")
                            
                            with col_figs2:
                                # 物理效应图
                                fig = plot_mixture_effect_contributions(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=mixture_energy_min, 
                                    e_max=mixture_energy_max, 
                                    points=mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                if fig:
                                    st.pyplot(fig)
                                    
                                    # 保存物理效应图数据
                                    for ax in fig.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue  # 跳过辅助线
                                            x_data = line.get_xdata()
                                            y_data = line.get_ydata()
                                            plot_data_all[f"Effect_{label}"] = {'x': x_data, 'y': y_data}
                                    
                                    plt.close(fig)
                                    st.caption("物理效应贡献图")
                                else:
                                    st.error("无法生成物理效应图")
                            
                            # 透射率图
                            st.subheader("混合物透射率")
                            fig = plot_mixture_transmission(
                                st.session_state.elements,
                                mixture_definition,
                                e_min=mixture_energy_min, 
                                e_max=mixture_energy_max, 
                                mixture_thickness=mixture_thickness,
                                points=mixture_num_points,
                                x_scale=x_scale_value,
                                y_scale='linear',
                                return_fig=True
                            )
                            if fig:
                                st.pyplot(fig)
                                
                                # 保存透射率图数据
                                for ax in fig.get_axes():
                                    for line in ax.get_lines():
                                        label = line.get_label()
                                        if label.startswith('_'): continue  # 跳过辅助线
                                        if not label or label == '_line0': label = 'Transmission'
                                        x_data = line.get_xdata()
                                        y_data = line.get_ydata()
                                        plot_data_all[label] = {'x': x_data, 'y': y_data}
                                
                                plt.close(fig)
                                st.caption("透射率图")
                            else:
                                st.error("无法生成透射率图")
                                
                            # 添加数据下载按钮 - 所有图表数据合并到一个Excel文件
                            if plot_data_all:
                                output = io.BytesIO()
                                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                
                                # 创建三个主工作表
                                component_df = pd.DataFrame()
                                effect_df = pd.DataFrame()
                                transmission_df = pd.DataFrame()
                                
                                # 处理能量范围
                                energy_range = np.linspace(mixture_energy_min, mixture_energy_max, mixture_num_points)
                                
                                # 分配数据到相应工作表
                                for label, data in plot_data_all.items():
                                    if label.startswith("Component_"):
                                        if 'Energy (MeV)' not in component_df:
                                            component_df['Energy (MeV)'] = data['x']
                                        component_df[label.replace("Component_", "")] = data['y']
                                    elif label.startswith("Effect_"):
                                        if 'Energy (MeV)' not in effect_df:
                                            effect_df['Energy (MeV)'] = data['x']
                                        effect_df[label.replace("Effect_", "")] = data['y']
                                    else:
                                        if 'Energy (MeV)' not in transmission_df:
                                            transmission_df['Energy (MeV)'] = data['x']
                                        transmission_df[label] = data['y']
                                
                                # 写入工作表
                                if not component_df.empty:
                                    component_df.to_excel(writer, sheet_name='Components')
                                if not effect_df.empty:
                                    effect_df.to_excel(writer, sheet_name='Effects')
                                if not transmission_df.empty:
                                    transmission_df.to_excel(writer, sheet_name='Transmission')
                                
                                # 创建混合物信息工作表
                                element_symbols = [elem[0] for elem in mixture_definition]
                                element_counts = [elem[1] for elem in mixture_definition]
                                mixture_info_df = pd.DataFrame({
                                    'Element': element_symbols,
                                    'Count': element_counts,
                                    'Mass Fraction': [weights.get(elem, 0) for elem in element_symbols],
                                    'Density (g/cm³)': densities
                                })
                                mixture_info_df.to_excel(writer, sheet_name='Mixture Info')
                                
                                writer.close()
                                output.seek(0)
                                
                                # 生成混合物名称用于文件命名
                                mixture_name = "_".join([f.replace(" ", "") for f in element_symbols[:3]])
                                if len(element_symbols) > 3:
                                    mixture_name += "_etc"
                                
                                st.download_button(
                                    label="下载所有图表数据",
                                    data=output,
                                    file_name=f"{mixture_name}_all_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        else:
                            # 仅显示透射率图
                            mixture_definition = [{"formula": f, "proportion": w, "density": d} 
                                                for f, w, d in zip(formulas, weights, densities)]
                            
                            fig = plot_mixture_transmission(
                                st.session_state.elements,
                                mixture_definition,
                                e_min=mixture_energy_min, 
                                e_max=mixture_energy_max, 
                                mixture_thickness=mixture_thickness,
                                points=mixture_num_points,
                                x_scale=x_scale_value,
                                y_scale='linear' if y_scale_value == 'linear' else 'log',
                                return_fig=True
                            )
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                                st.caption("透射率图")
                            else:
                                st.error("无法生成图表，请检查参数设置。")
                            
                            # 显示关键能量点的透射率
                            st.subheader("Key Point Transmission")
                            key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000  # 转换为MeV
                            key_energies = np.array([e for e in key_energies if mixture_energy_min <= e <= mixture_energy_max])
                            
                            if len(key_energies) > 0:
                                # 重新计算传输率
                                energy_range = np.linspace(mixture_energy_min, mixture_energy_max, mixture_num_points)
                                
                                # 计算总截面
                                total_cross_section = np.zeros_like(energy_range)
                                for formula, weight in zip(formulas, weights):
                                    try:
                                        cross_section = st.session_state.elements.calculate_compound_cross_section(
                                            formula, energy_range
                                        )
                                        total_cross_section += cross_section * (weight / 100)
                                    except Exception as e:
                                        continue
                                
                                # 计算传输率
                                transmission = np.exp(-total_cross_section * mixture_density * mixture_thickness)
                                
                                key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                                key_transmissions = [transmission[i] for i in key_indices]
                                
                                data = {
                                    "Energy (MeV)": key_energies,
                                    "Transmission": [f"{t:.4f}" for t in key_transmissions]
                                }
                                st.table(data)
                                
                                # 添加数据下载按钮 - 优化版本
                                output = io.BytesIO()
                                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                                
                                # 生成混合物名称用于文件命名
                                mixture_name = "_".join([f.replace(" ", "") for f in formulas[:3]])
                                if len(formulas) > 3:
                                    mixture_name += "_etc"
                                
                                # 保存透射率数据
                                transmission_df = pd.DataFrame({
                                    'Energy (MeV)': energy_range,
                                    'Transmission': transmission
                                })
                                transmission_df.to_excel(writer, sheet_name='Transmission')
                                
                                # 保存关键点数据
                                if len(key_energies) > 0:
                                    key_points_df = pd.DataFrame({
                                        'Energy (MeV)': key_energies,
                                        'Transmission': [t for t in key_transmissions]
                                    })
                                    key_points_df.to_excel(writer, sheet_name='Key Points')
                                
                                # 保存混合物信息
                                element_symbols = [elem[0] for elem in mixture_definition]
                                element_counts = [elem[1] for elem in mixture_definition]
                                mixture_info_df = pd.DataFrame({
                                    'Element': element_symbols,
                                    'Count': element_counts,
                                    'Mass Fraction': [weights.get(elem, 0) for elem in element_symbols],
                                    'Density (g/cm³)': densities
                                })
                                mixture_info_df.to_excel(writer, sheet_name='Mixture Info')
                                
                                writer.close()
                                output.seek(0)
                                
                                st.download_button(
                                    label="下载透射率数据",
                                    data=output,
                                    file_name=f"{mixture_name}_transmission.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>X-Ray Transmission Calculator | Interactive Interface Based on Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 