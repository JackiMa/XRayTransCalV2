import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import os
import pandas as pd
import time

# 设置matplotlib的显示DPI，提高图像清晰度
plt.rcParams['figure.dpi'] = 150  # 显示DPI适中即可
plt.rcParams['savefig.dpi'] = 300  # PDF矢量格式，DPI不需要太高

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

# 初始化所有需要持久化的状态
def initialize_session_state():
    """初始化所有session state变量"""
    # Elements tab 默认值
    if 'element_symbol' not in st.session_state:
        st.session_state.element_symbol = list(st.session_state.elements.elements.keys())[0] if st.session_state.elements.elements else 'H'
    if 'element_density' not in st.session_state:
        st.session_state.element_density = 1.0
    if 'element_thickness' not in st.session_state:
        st.session_state.element_thickness = 0.1
    if 'element_energy_min' not in st.session_state:
        st.session_state.element_energy_min = 0.01
    if 'element_energy_max' not in st.session_state:
        st.session_state.element_energy_max = 20.0
    if 'element_num_points' not in st.session_state:
        st.session_state.element_num_points = 5000
    if 'element_plot_type' not in st.session_state:
        st.session_state.element_plot_type = "Cross Section"
    if 'element_x_scale' not in st.session_state:
        st.session_state.element_x_scale = "Log"
    if 'element_y_scale' not in st.session_state:
        st.session_state.element_y_scale = "Log"
    if 'element_y_min_enabled' not in st.session_state:
        st.session_state.element_y_min_enabled = False
    if 'element_y_max_enabled' not in st.session_state:
        st.session_state.element_y_max_enabled = False
    if 'element_y_min' not in st.session_state:
        st.session_state.element_y_min = 0.00001
    if 'element_y_max' not in st.session_state:
        st.session_state.element_y_max = 10.0
    
    # Compounds tab 默认值
    if 'compound_formula' not in st.session_state:
        st.session_state.compound_formula = "H2O"
    if 'compound_density' not in st.session_state:
        st.session_state.compound_density = 1.0
    if 'compound_thickness' not in st.session_state:
        st.session_state.compound_thickness = 0.1
    if 'compound_energy_min' not in st.session_state:
        st.session_state.compound_energy_min = 0.01
    if 'compound_energy_max' not in st.session_state:
        st.session_state.compound_energy_max = 20.0
    if 'compound_num_points' not in st.session_state:
        st.session_state.compound_num_points = 5000
    if 'compound_plot_type' not in st.session_state:
        st.session_state.compound_plot_type = "Component Contribution"
    if 'compound_x_scale' not in st.session_state:
        st.session_state.compound_x_scale = "Log"
    if 'compound_y_scale' not in st.session_state:
        st.session_state.compound_y_scale = "Log"
    if 'compound_y_min_enabled' not in st.session_state:
        st.session_state.compound_y_min_enabled = False
    if 'compound_y_max_enabled' not in st.session_state:
        st.session_state.compound_y_max_enabled = False
    if 'compound_y_min' not in st.session_state:
        st.session_state.compound_y_min = 0.001
    if 'compound_y_max' not in st.session_state:
        st.session_state.compound_y_max = 1.0
    
    # Mixtures tab 默认值
    if 'mixture_components' not in st.session_state:
        st.session_state.mixture_components = [
            {"formula": "H2O", "weight_percent": 70.0, "density": 1.0}
        ]
    if 'mixture_thickness' not in st.session_state:
        st.session_state.mixture_thickness = 0.1
    if 'mixture_energy_min' not in st.session_state:
        st.session_state.mixture_energy_min = 0.01
    if 'mixture_energy_max' not in st.session_state:
        st.session_state.mixture_energy_max = 20.0
    if 'mixture_num_points' not in st.session_state:
        st.session_state.mixture_num_points = 5000
    if 'mixture_plot_type' not in st.session_state:
        st.session_state.mixture_plot_type = "Component Contribution"
    if 'mixture_x_scale' not in st.session_state:
        st.session_state.mixture_x_scale = "Log"
    if 'mixture_y_scale' not in st.session_state:
        st.session_state.mixture_y_scale = "Log"
    if 'mixture_y_min_enabled' not in st.session_state:
        st.session_state.mixture_y_min_enabled = False
    if 'mixture_y_max_enabled' not in st.session_state:
        st.session_state.mixture_y_max_enabled = False
    if 'mixture_y_min' not in st.session_state:
        st.session_state.mixture_y_min = 0.001
    if 'mixture_y_max' not in st.session_state:
        st.session_state.mixture_y_max = 1.0

# 调用初始化函数
initialize_session_state()

# Create temporary directory (if it doesn't exist)
os.makedirs("tmp_plots", exist_ok=True)

# Helper function: Convert image to base64 encoding to embed in the page
def get_image_pdf(fig):
    """将matplotlib图形转换为高质量的PDF格式的字节数据"""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def create_download_buttons(fig, base_filename, plot_data=None):
    """创建图片和数据下载按钮"""
    import time
    timestamp = str(int(time.time() * 1000))  # 使用时间戳确保唯一性
    
    if fig is not None:
        # 尝试生成PDF，如果失败则生成PNG作为备选
        try:
            pdf_data = get_image_pdf(fig)
            st.download_button(
                label="📄 下载图片 (PDF)",
                data=pdf_data,
                file_name=f"{base_filename}.pdf",
                mime="application/pdf",
                key=f"pdf_{base_filename}_{timestamp}"
            )
        except Exception as e:
            # 如果PDF生成失败，提供PNG下载
            st.warning(f"PDF生成失败，提供PNG格式下载: {str(e)}")
            try:
                # 生成PNG作为备选
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                buf.seek(0)
                png_data = buf.getvalue()
                
                st.download_button(
                    label="📷 下载图片 (PNG)",
                    data=png_data,
                    file_name=f"{base_filename}.png",
                    mime="image/png",
                    key=f"png_{base_filename}_{timestamp}"
                )
            except Exception as png_error:
                st.error(f"图片生成失败: {str(png_error)}")
    
    if plot_data:
        # 数据下载按钮
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        
        # 根据数据类型创建不同的工作表
        if isinstance(plot_data, dict):
            if len(plot_data) == 1 and "Transmission" in plot_data:
                # 透射率数据
                transmission_df = pd.DataFrame({
                    'Energy (MeV)': plot_data["Transmission"]['x'],
                    'Transmission': plot_data["Transmission"]['y']
                })
                transmission_df.to_excel(writer, sheet_name='Transmission', index=False)
            else:
                # 截面系数数据
                df = pd.DataFrame()
                for label, data in plot_data.items():
                    if 'Energy (MeV)' not in df:
                        df['Energy (MeV)'] = data['x']
                    df[label] = data['y']
                df.to_excel(writer, sheet_name='Data', index=False)
        
        writer.close()
        output.seek(0)
        
        st.download_button(
            label="📊 下载数据 (Excel)",
            data=output,
            file_name=f"{base_filename}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"excel_{base_filename}_{timestamp}"
        )

# Main application interface
def main():
    st.title("X-Ray Transmission Calculator")
    
    # Create three tabs
    tabs = st.tabs(["Elements", "Defined Materials", "Mixtures"])
    
    # Elements tab
    with tabs[0]:
        st.header("Element X-Ray Transmission Calculation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            element_symbol = st.selectbox(
                "Select Element Symbol",
                options=list(st.session_state.elements.elements.keys()),
                format_func=lambda x: f"{x} - {st.session_state.elements.elements[x].name if x in st.session_state.elements.elements else ''}",
                index=list(st.session_state.elements.elements.keys()).index(st.session_state.element_symbol) if st.session_state.element_symbol in st.session_state.elements.elements else 0,
                key="element_symbol_select"
            )
            # 更新session_state
            if element_symbol != st.session_state.element_symbol:
                st.session_state.element_symbol = element_symbol
            
            element = st.session_state.elements.elements[element_symbol] if element_symbol in st.session_state.elements.elements else None
            
            # 根据选择的元素更新密度默认值
            if element and hasattr(element, 'metadata') and 'Density' in element.metadata:
                default_density = float(element.metadata['Density'])
            else:
                default_density = st.session_state.element_density

            density = st.number_input(
                "Density (g/cm³)",
                min_value=0.0001,
                value=default_density,
                format="%.4f",
                key="element_density_input"
            )
            st.session_state.element_density = density
            
            thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.0001,
                max_value=1000.0,
                value=st.session_state.element_thickness,
                format="%.4f",
                key="element_thickness_input"
            )
            st.session_state.element_thickness = thickness
            
            energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=st.session_state.element_energy_min,
                format="%.4f",
                key="element_energy_min_input"
            )
            st.session_state.element_energy_min = energy_min
            
            energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.0001,
                max_value=100000.0,
                value=st.session_state.element_energy_max,
                format="%.4f",
                key="element_energy_max_input"
            )
            st.session_state.element_energy_max = energy_max
            
            num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=st.session_state.element_num_points,
                step=100,
                key="element_num_points_input"
            )
            st.session_state.element_num_points = num_points
            
            plot_type = st.radio(
                "Plot Type",
                options=["Cross Section", "Transmission"],
                index=0 if st.session_state.element_plot_type == "Cross Section" else 1,
                key="element_plot_type_input"
            )
            st.session_state.element_plot_type = plot_type
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.element_x_scale == "Log" else 0,
                    horizontal=True,
                    key="element_x_scale_input"
                )
                st.session_state.element_x_scale = x_scale
            
            with col2_scale:
                y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.element_y_scale == "Log" else 0,
                    horizontal=True,
                    key="element_y_scale_input"
                )
                st.session_state.element_y_scale = y_scale
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                y_min_enabled = st.checkbox(
                    "Set Y Min", 
                    value=st.session_state.element_y_min_enabled, 
                    key="element_y_min_enabled_input"
                )
                st.session_state.element_y_min_enabled = y_min_enabled
                
                # 始终显示输入框，但只在启用时使用其值
                y_min = st.number_input(
                    "Y Axis Minimum",
                    value=st.session_state.element_y_min,
                    format="%.6f",
                    disabled=not y_min_enabled,
                    key="element_y_min_input"
                )
                if y_min_enabled:
                    st.session_state.element_y_min = y_min
            
            with col2_yrange:
                y_max_enabled = st.checkbox(
                    "Set Y Max", 
                    value=st.session_state.element_y_max_enabled, 
                    key="element_y_max_enabled_input"
                )
                st.session_state.element_y_max_enabled = y_max_enabled
                
                # 始终显示输入框，但只在启用时使用其值
                y_max = st.number_input(
                    "Y Axis Maximum",
                    value=st.session_state.element_y_max,
                    format="%.6f",
                    disabled=not y_max_enabled,
                    key="element_y_max_input"
                )
                if y_max_enabled:
                    st.session_state.element_y_max = y_max
            
            calculate_button = st.button("Calculate", key="calculate_element")
        
        with col2:
            # 检查是否需要重新计算
            needs_calculation = (
                calculate_button
            )
            
            # 生成结果的唯一标识
            result_key = f'element_result_{st.session_state.element_symbol}_{st.session_state.element_plot_type}_{st.session_state.element_energy_min}_{st.session_state.element_energy_max}_{st.session_state.element_num_points}_{st.session_state.element_thickness}_{st.session_state.element_density}'
            
            if needs_calculation:
                with st.spinner('正在计算中，请稍候...'):
                    try:
                        element = st.session_state.elements.elements[st.session_state.element_symbol]
                        
                        # 转换坐标轴类型为小写
                        x_scale_value = st.session_state.element_x_scale.lower()
                        y_scale_value = st.session_state.element_y_scale.lower()
                        
                        # 获取Y轴范围设置
                        y_min_val = st.session_state.element_y_min if st.session_state.element_y_min_enabled else None
                        y_max_val = st.session_state.element_y_max if st.session_state.element_y_max_enabled else None
                        
                        # 存储绘图数据以便下载
                        plot_data = {}
                        
                        if st.session_state.element_plot_type == "Cross Section":
                            # 绘制元素截面系数图
                            fig = plot_element_cross_sections(
                                element, 
                                e_min=st.session_state.element_energy_min, 
                                e_max=st.session_state.element_energy_max, 
                                points=st.session_state.element_num_points,
                                x_scale=x_scale_value,
                                y_scale=y_scale_value,
                                return_fig=True
                            )
                            if fig:
                                # 应用自定义Y轴范围
                                if st.session_state.element_y_min_enabled or st.session_state.element_y_max_enabled:
                                    for ax in fig.get_axes():
                                        current_ylim = ax.get_ylim()
                                        new_ymin = y_min_val if st.session_state.element_y_min_enabled else current_ylim[0]
                                        new_ymax = y_max_val if st.session_state.element_y_max_enabled else current_ylim[1]
                                        ax.set_ylim(new_ymin, new_ymax)
                                
                                # 准备下载数据
                                energy_range = np.linspace(st.session_state.element_energy_min, st.session_state.element_energy_max, st.session_state.element_num_points)
                                
                                for ax in fig.get_axes():
                                    for line in ax.get_lines():
                                        label = line.get_label()
                                        if label.startswith('_'): continue  # 跳过辅助线
                                        y_data = line.get_ydata()
                                        plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 保存结果到session_state
                                st.session_state[result_key] = {
                                    'fig': fig,
                                    'plot_data': plot_data,
                                    'plot_type': 'Cross Section'
                                }
                            else:
                                st.error("无法生成图表，请检查参数设置。")
                        else:
                            # 计算透射率
                            energy_range = np.linspace(st.session_state.element_energy_min, st.session_state.element_energy_max, st.session_state.element_num_points)
                            transmission = element.calculate_transmission(
                                energy_range, 
                                thickness=st.session_state.element_thickness,
                                density=st.session_state.element_density
                            )
                            
                            # 保存传输率数据用于下载
                            plot_data["Transmission"] = {'x': energy_range, 'y': transmission}
                            
                            # 创建透射率图
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(energy_range, transmission, 'b-', linewidth=2)
                            ax.set_xlabel('Photon Energy (MeV)')
                            ax.set_ylabel('Transmission')
                            ax.set_title(f'{st.session_state.element_symbol} - Thickness: {st.session_state.element_thickness} cm, Density: {st.session_state.element_density} g/cm³')
                            ax.grid(True, alpha=0.3)
                            ax.set_xlim(st.session_state.element_energy_min, st.session_state.element_energy_max)
                            
                            # 应用自定义Y轴范围或默认范围
                            if st.session_state.element_y_min_enabled or st.session_state.element_y_max_enabled:
                                default_ymin = 0.001 if y_scale_value == 'log' else 0.0
                                default_ymax = 1.05
                                new_ymin = y_min_val if st.session_state.element_y_min_enabled else default_ymin
                                new_ymax = y_max_val if st.session_state.element_y_max_enabled else default_ymax
                                ax.set_ylim(new_ymin, new_ymax)
                            else:
                                ax.set_ylim(0, 1.05)
                            
                            # 设置坐标轴类型
                            ax.set_xscale(x_scale_value)
                            if y_scale_value == 'log':
                                # 对于透射图，对数坐标下需要处理0值
                                ax.set_yscale(y_scale_value)
                                if not st.session_state.element_y_min_enabled:
                                    ax.set_ylim(0.001, 1.05)  # 对数坐标下调整下限，除非用户自定义
                            
                            # 计算关键能量点的透射率
                            key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000  # 转换为MeV
                            key_energies = np.array([e for e in key_energies if st.session_state.element_energy_min <= e <= st.session_state.element_energy_max])
                            
                            key_transmissions = []
                            transmission_plot_data = {}
                            if len(key_energies) > 0:
                                key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                                key_transmissions = [transmission[i] for i in key_indices]
                                for i in key_indices:
                                    transmission_plot_data[f'Key Point {i+1}'] = {'x': energy_range[i], 'y': transmission[i]}
                            
                            # 保存结果到session_state
                            st.session_state[result_key] = {
                                'fig': fig,
                                'plot_data': plot_data,
                                'plot_type': 'Transmission',
                                'key_energies': key_energies,
                                'key_transmissions': key_transmissions,
                                'transmission_plot_data': transmission_plot_data
                            }
                    
                    except Exception as e:
                        st.error(f"Calculation error: {str(e)}")
                        return
            
            # 显示保存的结果
            if result_key in st.session_state:
                result = st.session_state[result_key]
                fig = result['fig']
                plot_data = result['plot_data']
                
                st.pyplot(fig)
                
                # 添加下载按钮
                create_download_buttons(
                    fig, 
                    f"{st.session_state.element_symbol}_{result['plot_type'].replace(' ', '_').lower()}", 
                    plot_data
                )
                
                if result['plot_type'] == 'Cross Section':
                    # 同时绘制效应贡献图
                    st.subheader("Element Interaction Effects")
                    col_effects1, col_effects2 = st.columns(2)
                    with col_effects1:
                        st.write("主要物理效应包括：光电效应、相干/非相干散射和对产生")
                    with col_effects2:
                        st.write("data from https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html")
                else:
                    # 显示关键能量点的透射率
                    if len(result['key_energies']) > 0:
                        st.subheader("Key Point Transmission")
                        data = {
                            "Energy (MeV)": result['key_energies'],
                            "Transmission": [f"{t:.4f}" for t in result['key_transmissions']]
                        }
                        st.table(data)
    
    # Compounds tab
    with tabs[1]:
        st.header("Defined Materials X-Ray Transmission Calculation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            compound_formula = st.text_input(
                "Specific Chemical Formula (e.g.: Au, H2O, Cd0.9Zn0.1Te)",
                value=st.session_state.compound_formula,
                key="compound_formula_input"
            )
            st.session_state.compound_formula = compound_formula
            
            compound_density = st.number_input(
                "Density (g/cm³)",
                min_value=0.0001,
                value=st.session_state.compound_density,
                format="%.4f",
                key="compound_density_input"
            )
            st.session_state.compound_density = compound_density
            
            compound_thickness = st.number_input(
                "Thickness (cm)",
                min_value=0.0001,
                max_value=1000.0,
                value=st.session_state.compound_thickness,
                format="%.4f",
                key="compound_thickness_input"
            )
            st.session_state.compound_thickness = compound_thickness
            
            compound_energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=st.session_state.compound_energy_min,
                format="%.4f",
                key="compound_energy_min_input"
            )
            st.session_state.compound_energy_min = compound_energy_min
            
            compound_energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=st.session_state.compound_energy_max,
                format="%.4f",
                key="compound_energy_max_input"
            )
            st.session_state.compound_energy_max = compound_energy_max
            
            compound_num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=st.session_state.compound_num_points,
                step=100,
                key="compound_num_points_input"
            )
            st.session_state.compound_num_points = compound_num_points
            
            compound_plot_type = st.radio(
                "Plot Type",
                options=["Component Contribution", "Transmission"],
                index=0 if st.session_state.compound_plot_type == "Component Contribution" else 1,
                key="compound_plot_type_input"
            )
            st.session_state.compound_plot_type = compound_plot_type
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                compound_x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.compound_x_scale == "Log" else 0,
                    horizontal=True,
                    key="compound_x_scale_input"
                )
                st.session_state.compound_x_scale = compound_x_scale
            
            with col2_scale:
                compound_y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.compound_y_scale == "Log" else 0,
                    horizontal=True,
                    key="compound_y_scale_input"
                )
                st.session_state.compound_y_scale = compound_y_scale
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                compound_y_min_enabled = st.checkbox(
                    "Set Y Min", 
                    value=st.session_state.compound_y_min_enabled, 
                    key="compound_y_min_enabled_input"
                )
                st.session_state.compound_y_min_enabled = compound_y_min_enabled
                
                compound_y_min = st.number_input(
                    "Y Axis Minimum",
                    value=st.session_state.compound_y_min,
                    format="%.6f",
                    disabled=not compound_y_min_enabled,
                    key="compound_y_min_input"
                )
                if compound_y_min_enabled:
                    st.session_state.compound_y_min = compound_y_min
            
            with col2_yrange:
                compound_y_max_enabled = st.checkbox(
                    "Set Y Max", 
                    value=st.session_state.compound_y_max_enabled, 
                    key="compound_y_max_enabled_input"
                )
                st.session_state.compound_y_max_enabled = compound_y_max_enabled
                
                compound_y_max = st.number_input(
                    "Y Axis Maximum",
                    value=st.session_state.compound_y_max,
                    format="%.6f",
                    disabled=not compound_y_max_enabled,
                    key="compound_y_max_input"
                )
                if compound_y_max_enabled:
                    st.session_state.compound_y_max = compound_y_max
            
            calculate_compound_button = st.button("Calculate", key="calculate_compound")
        
        with col2:
            # 检查是否需要重新计算
            needs_calculation = (
                calculate_compound_button
            )
            
            # 生成结果的唯一标识
            result_key = f'compound_result_{st.session_state.compound_formula}_{st.session_state.compound_plot_type}_{st.session_state.compound_energy_min}_{st.session_state.compound_energy_max}_{st.session_state.compound_num_points}_{st.session_state.compound_thickness}_{st.session_state.compound_density}'
            
            if needs_calculation:
                with st.spinner('正在计算化合物数据，请稍候...'):
                    try:
                        # 解析化学式
                        formula_elements = st.session_state.elements.parse_chemical_formula(st.session_state.compound_formula)
                        
                        # 获取质量分数
                        mass_fractions = st.session_state.elements.calculate_mass_fractions(st.session_state.compound_formula)
                        
                        # 转换坐标轴类型为小写
                        x_scale_value = st.session_state.compound_x_scale.lower()
                        y_scale_value = st.session_state.compound_y_scale.lower()
                        
                        if not formula_elements:
                            st.error("Cannot parse chemical formula, please check input format.")
                        else:
                            if st.session_state.compound_plot_type == "Component Contribution":
                                # 准备混合物定义
                                plot_data_all = {}
                                
                                # 成分贡献图
                                fig_components = plot_compound_components(
                                    st.session_state.elements,
                                    st.session_state.compound_formula,
                                    e_min=st.session_state.compound_energy_min, 
                                    e_max=st.session_state.compound_energy_max, 
                                    points=st.session_state.compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                
                                # 提取成分贡献图数据
                                components_plot_data = {}
                                if fig_components:
                                    energy_range = np.linspace(st.session_state.compound_energy_min, st.session_state.compound_energy_max, st.session_state.compound_num_points)
                                    for ax in fig_components.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            y_data = line.get_ydata()
                                            components_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 物理效应图
                                fig_effects = plot_compound_effect_contributions(
                                    st.session_state.elements,
                                    st.session_state.compound_formula,
                                    e_min=st.session_state.compound_energy_min, 
                                    e_max=st.session_state.compound_energy_max, 
                                    points=st.session_state.compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                
                                # 提取物理效应图数据
                                effects_plot_data = {}
                                if fig_effects:
                                    energy_range = np.linspace(st.session_state.compound_energy_min, st.session_state.compound_energy_max, st.session_state.compound_num_points)
                                    for ax in fig_effects.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            y_data = line.get_ydata()
                                            effects_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 透射率图
                                fig_transmission = plot_compound_transmission(
                                    st.session_state.elements,
                                    st.session_state.compound_formula,
                                    e_min=st.session_state.compound_energy_min, 
                                    e_max=st.session_state.compound_energy_max, 
                                    density=st.session_state.compound_density,
                                    thickness=st.session_state.compound_thickness,
                                    points=st.session_state.compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale='linear',
                                    return_fig=True
                                )
                                
                                # 提取透射率图数据
                                transmission_plot_data = {}
                                if fig_transmission:
                                    energy_range = np.linspace(st.session_state.compound_energy_min, st.session_state.compound_energy_max, st.session_state.compound_num_points)
                                    for ax in fig_transmission.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            if not label or label == '_line0': label = 'Transmission'
                                            y_data = line.get_ydata()
                                            transmission_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 保存结果到session_state
                                st.session_state[result_key] = {
                                    'fig_components': fig_components,
                                    'fig_effects': fig_effects,
                                    'fig_transmission': fig_transmission,
                                    'components_plot_data': components_plot_data,
                                    'effects_plot_data': effects_plot_data,
                                    'transmission_plot_data': transmission_plot_data,
                                    'plot_type': 'Component Contribution',
                                    'formula_elements': formula_elements,
                                    'mass_fractions': mass_fractions
                                }
                            else:
                                # 仅显示透射率图
                                fig = plot_compound_transmission(
                                    st.session_state.elements,
                                    st.session_state.compound_formula,
                                    e_min=st.session_state.compound_energy_min, 
                                    e_max=st.session_state.compound_energy_max, 
                                    density=st.session_state.compound_density,
                                    thickness=st.session_state.compound_thickness,
                                    points=st.session_state.compound_num_points,
                                    x_scale=x_scale_value,
                                    y_scale='linear' if y_scale_value == 'linear' else 'log',
                                    return_fig=True
                                )
                                
                                # 计算关键能量点的透射率
                                key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000
                                key_energies = np.array([e for e in key_energies if st.session_state.compound_energy_min <= e <= st.session_state.compound_energy_max])
                                
                                key_transmissions = []
                                transmission_plot_data = {}
                                if len(key_energies) > 0:
                                    energy_range = np.linspace(st.session_state.compound_energy_min, st.session_state.compound_energy_max, st.session_state.compound_num_points)
                                    result = st.session_state.elements.calculate_compound_cross_section(
                                        st.session_state.compound_formula, energy_range
                                    )
                                    if result is not None:
                                        total_cross_section, _, _ = result
                                        transmission = np.exp(-total_cross_section * st.session_state.compound_density * st.session_state.compound_thickness)
                                        key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                                        key_transmissions = [transmission[i] for i in key_indices]
                                        # 保存透射率数据
                                        transmission_plot_data['Transmission'] = {'x': energy_range, 'y': transmission}
                                
                                # 保存结果到session_state
                                st.session_state[result_key] = {
                                    'fig': fig,
                                    'plot_type': 'Transmission',
                                    'formula_elements': formula_elements,
                                    'mass_fractions': mass_fractions,
                                    'key_energies': key_energies,
                                    'key_transmissions': key_transmissions,
                                    'transmission_plot_data': transmission_plot_data
                                }
                    
                    except Exception as e:
                        st.error(f"Calculation error: {str(e)}")
                        return
            
            # 显示保存的结果
            if result_key in st.session_state:
                result = st.session_state[result_key]
                
                if result['plot_type'] == "Component Contribution":
                    # 使用plot_compound_all绘制多个图表
                    st.subheader("化合物成分与透射分析")
                    st.info("正在显示缓存的图表...")
                    
                    col_figs1, col_figs2 = st.columns(2)
                    
                    with col_figs1:
                        # 成分贡献图
                        if result['fig_components']:
                            st.pyplot(result['fig_components'])
                            compound_name = st.session_state.compound_formula.replace(" ", "")
                            create_download_buttons(
                                result['fig_components'], 
                                f"{compound_name}_components",
                                result['components_plot_data']
                            )
                            st.caption("混合物成分贡献图")
                    
                    with col_figs2:
                        # 物理效应图
                        if result['fig_effects']:
                            st.pyplot(result['fig_effects'])
                            create_download_buttons(
                                result['fig_effects'], 
                                f"{compound_name}_effects",
                                result['effects_plot_data']
                            )
                            st.caption("物理效应贡献图")
                    
                    # 透射率图
                    st.subheader("混合物透射率")
                    if result['fig_transmission']:
                        st.pyplot(result['fig_transmission'])
                        create_download_buttons(
                            result['fig_transmission'], 
                            f"{compound_name}_transmission",
                            result['transmission_plot_data']
                        )
                        st.caption("透射率图")
                else:
                    # 仅显示透射率图
                    if result['fig']:
                        st.pyplot(result['fig'])
                        compound_name = st.session_state.compound_formula.replace(" ", "")
                        create_download_buttons(
                            result['fig'], 
                            f"{compound_name}_transmission",
                            result['transmission_plot_data']
                        )
                        st.caption("透射率图")
                    
                    # 显示关键能量点的透射率
                    if len(result['key_energies']) > 0:
                        st.subheader("Key Point Transmission")
                        data = {
                            "Energy (MeV)": result['key_energies'],
                            "Transmission": [f"{t:.4f}" for t in result['key_transmissions']]
                        }
                        st.table(data)
    
    # Mixtures tab
    with tabs[2]:
        st.header("""Mixture X-Ray Transmission Calculation
                  \n
                  !!! This function has not been fully verified !!!\n
                  \n""")
        
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
                value=st.session_state.mixture_thickness,
                format="%.4f",
                key="mixture_thickness_input"
            )
            st.session_state.mixture_thickness = mixture_thickness
            
            mixture_energy_min = st.number_input(
                "Minimum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=st.session_state.mixture_energy_min,
                format="%.4f",
                key="mixture_energy_min_input"
            )
            st.session_state.mixture_energy_min = mixture_energy_min
            
            mixture_energy_max = st.number_input(
                "Maximum Energy (MeV)",
                min_value=0.001,
                max_value=100000.0,
                value=st.session_state.mixture_energy_max,
                format="%.1f",
                key="mixture_energy_max_input"
            )
            st.session_state.mixture_energy_max = mixture_energy_max
            
            mixture_num_points = st.number_input(
                "Number of Points",
                min_value=100,
                max_value=100000,
                value=st.session_state.mixture_num_points,
                step=100,
                key="mixture_num_points_input"
            )
            st.session_state.mixture_num_points = mixture_num_points
            
            mixture_plot_type = st.radio(
                "Plot Type",
                options=["Component Contribution", "Transmission"],
                index=0 if st.session_state.mixture_plot_type == "Component Contribution" else 1,
                key="mixture_plot_type_input"
            )
            st.session_state.mixture_plot_type = mixture_plot_type
            
            # 添加坐标轴类型选择
            col1_scale, col2_scale = st.columns(2)
            
            with col1_scale:
                mixture_x_scale = st.radio(
                    "X-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.mixture_x_scale == "Log" else 0,
                    horizontal=True,
                    key="mixture_x_scale_input"
                )
                st.session_state.mixture_x_scale = mixture_x_scale
            
            with col2_scale:
                mixture_y_scale = st.radio(
                    "Y-Axis Scale",
                    options=["Linear", "Log"],
                    index=1 if st.session_state.mixture_y_scale == "Log" else 0,
                    horizontal=True,
                    key="mixture_y_scale_input"
                )
                st.session_state.mixture_y_scale = mixture_y_scale
            
            # 添加Y轴范围设置
            col1_yrange, col2_yrange = st.columns(2)
            
            with col1_yrange:
                mixture_y_min_enabled = st.checkbox(
                    "Set Y Min", 
                    value=st.session_state.mixture_y_min_enabled, 
                    key="mixture_y_min_enabled_input"
                )
                st.session_state.mixture_y_min_enabled = mixture_y_min_enabled
                
                mixture_y_min = st.number_input(
                    "Y Axis Minimum",
                    value=st.session_state.mixture_y_min,
                    format="%.6f",
                    disabled=not mixture_y_min_enabled,
                    key="mixture_y_min_input"
                )
                if mixture_y_min_enabled:
                    st.session_state.mixture_y_min = mixture_y_min
            
            with col2_yrange:
                mixture_y_max_enabled = st.checkbox(
                    "Set Y Max", 
                    value=st.session_state.mixture_y_max_enabled, 
                    key="mixture_y_max_enabled_input"
                )
                st.session_state.mixture_y_max_enabled = mixture_y_max_enabled
                
                mixture_y_max = st.number_input(
                    "Y Axis Maximum",
                    value=st.session_state.mixture_y_max,
                    format="%.6f",
                    disabled=not mixture_y_max_enabled,
                    key="mixture_y_max_input"
                )
                if mixture_y_max_enabled:
                    st.session_state.mixture_y_max = mixture_y_max
            
            calculate_mixture_button = st.button("Calculate", key="calculate_mixture")
        
        with col2:
            # 检查是否需要重新计算
            needs_calculation = (
                calculate_mixture_button
            )
            
            # 生成结果的唯一标识 - 包含组件信息的hash
            components_str = str([(comp["formula"], comp["weight_percent"], comp["density"]) for comp in st.session_state.mixture_components])
            result_key = f'mixture_result_{hash(components_str)}_{st.session_state.mixture_plot_type}_{st.session_state.mixture_energy_min}_{st.session_state.mixture_energy_max}_{st.session_state.mixture_num_points}_{st.session_state.mixture_thickness}'
            
            if needs_calculation:
                with st.spinner('正在计算混合物数据，请稍候...'):
                    try:
                        # 检查有效组件
                        valid_components = [comp for comp in st.session_state.mixture_components 
                                          if comp["formula"].strip() and comp["weight_percent"] > 0]
                        
                        # 转换坐标轴类型为小写
                        x_scale_value = st.session_state.mixture_x_scale.lower()
                        y_scale_value = st.session_state.mixture_y_scale.lower()
                        
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
                            
                            if st.session_state.mixture_plot_type == "Component Contribution":
                                # 准备混合物定义
                                mixture_definition = [{"formula": f, "proportion": w, "density": d} 
                                                    for f, w, d in zip(formulas, weights, densities)]
                                
                                # 成分贡献图
                                fig_components = plot_mixture_components(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=st.session_state.mixture_energy_min, 
                                    e_max=st.session_state.mixture_energy_max, 
                                    points=st.session_state.mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                
                                # 提取成分贡献图数据
                                components_plot_data = {}
                                if fig_components:
                                    energy_range = np.linspace(st.session_state.mixture_energy_min, st.session_state.mixture_energy_max, st.session_state.mixture_num_points)
                                    for ax in fig_components.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            y_data = line.get_ydata()
                                            components_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 物理效应图
                                fig_effects = plot_mixture_effect_contributions(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=st.session_state.mixture_energy_min, 
                                    e_max=st.session_state.mixture_energy_max, 
                                    points=st.session_state.mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale=y_scale_value,
                                    return_fig=True
                                )
                                
                                # 提取物理效应图数据
                                effects_plot_data = {}
                                if fig_effects:
                                    energy_range = np.linspace(st.session_state.mixture_energy_min, st.session_state.mixture_energy_max, st.session_state.mixture_num_points)
                                    for ax in fig_effects.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            y_data = line.get_ydata()
                                            effects_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 透射率图
                                fig_transmission = plot_mixture_transmission(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=st.session_state.mixture_energy_min, 
                                    e_max=st.session_state.mixture_energy_max, 
                                    mixture_thickness=st.session_state.mixture_thickness,
                                    points=st.session_state.mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale='linear',
                                    return_fig=True
                                )
                                
                                # 提取透射率图数据
                                transmission_plot_data = {}
                                if fig_transmission:
                                    energy_range = np.linspace(st.session_state.mixture_energy_min, st.session_state.mixture_energy_max, st.session_state.mixture_num_points)
                                    for ax in fig_transmission.get_axes():
                                        for line in ax.get_lines():
                                            label = line.get_label()
                                            if label.startswith('_'): continue
                                            if not label or label == '_line0': label = 'Transmission'
                                            y_data = line.get_ydata()
                                            transmission_plot_data[label] = {'x': energy_range, 'y': y_data}
                                
                                # 保存结果到session_state
                                st.session_state[result_key] = {
                                    'fig_components': fig_components,
                                    'fig_effects': fig_effects,
                                    'fig_transmission': fig_transmission,
                                    'components_plot_data': components_plot_data,
                                    'effects_plot_data': effects_plot_data,
                                    'transmission_plot_data': transmission_plot_data,
                                    'plot_type': 'Component Contribution',
                                    'formulas': formulas,
                                    'weights': weights,
                                    'densities': densities,
                                    'mixture_density': mixture_density
                                }
                            else:
                                # 仅显示透射率图
                                mixture_definition = [{"formula": f, "proportion": w, "density": d} 
                                                    for f, w, d in zip(formulas, weights, densities)]
                                
                                fig = plot_mixture_transmission(
                                    st.session_state.elements,
                                    mixture_definition,
                                    e_min=st.session_state.mixture_energy_min, 
                                    e_max=st.session_state.mixture_energy_max, 
                                    mixture_thickness=st.session_state.mixture_thickness,
                                    points=st.session_state.mixture_num_points,
                                    x_scale=x_scale_value,
                                    y_scale='linear' if y_scale_value == 'linear' else 'log',
                                    return_fig=True
                                )
                                
                                # 计算关键能量点的透射率
                                key_energies = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]) / 1000
                                key_energies = np.array([e for e in key_energies if st.session_state.mixture_energy_min <= e <= st.session_state.mixture_energy_max])
                                
                                key_transmissions = []
                                transmission_plot_data = {}
                                if len(key_energies) > 0:
                                    # 计算传输率 - 简化版本
                                    energy_range = np.linspace(st.session_state.mixture_energy_min, st.session_state.mixture_energy_max, st.session_state.mixture_num_points)
                                    total_cross_section = np.zeros_like(energy_range)
                                    for formula, weight in zip(formulas, weights):
                                        try:
                                            cross_section = st.session_state.elements.calculate_compound_cross_section(
                                                formula, energy_range
                                            )
                                            if cross_section is not None:
                                                if isinstance(cross_section, tuple):
                                                    cross_section = cross_section[0]  # 取第一个元素
                                                total_cross_section += cross_section * (weight / 100)
                                        except Exception as e:
                                            continue
                                    
                                    transmission = np.exp(-total_cross_section * mixture_density * st.session_state.mixture_thickness)
                                    key_indices = [np.abs(energy_range - e).argmin() for e in key_energies]
                                    key_transmissions = [transmission[i] for i in key_indices]
                                    # 保存透射率数据
                                    transmission_plot_data['Transmission'] = {'x': energy_range, 'y': transmission}
                                
                                # 保存结果到session_state
                                st.session_state[result_key] = {
                                    'fig': fig,
                                    'plot_type': 'Transmission',
                                    'formulas': formulas,
                                    'weights': weights,
                                    'densities': densities,
                                    'mixture_density': mixture_density,
                                    'key_energies': key_energies,
                                    'key_transmissions': key_transmissions,
                                    'transmission_plot_data': transmission_plot_data
                                }
                
                    except Exception as e:
                        st.error(f"Calculation error: {str(e)}")
                        return
            
            # 显示保存的结果
            if result_key in st.session_state:
                result = st.session_state[result_key]
                
                # 显示混合物信息
                st.subheader("Mixture Information")
                st.write(f"Mixture Average Density: {result['mixture_density']:.4f} g/cm³")
                
                component_data = []
                for formula, weight, density in zip(result['formulas'], result['weights'], result['densities']):
                    component_data.append({
                        "Formula": formula,
                        "Weight Percent": f"{weight:.2f}%",
                        "Density": f"{density:.4f} g/cm³"
                    })
                st.table(component_data)
                
                if result['plot_type'] == "Component Contribution":
                    # 使用plot_mixture_all显示多个图表
                    st.subheader("混合物成分与透射分析")
                    st.info("正在显示缓存的图表...")
                    
                    col_figs1, col_figs2 = st.columns(2)
                    
                    with col_figs1:
                        # 成分贡献图
                        if result['fig_components']:
                            st.pyplot(result['fig_components'])
                            compound_name = st.session_state.compound_formula.replace(" ", "")
                            create_download_buttons(
                                result['fig_components'], 
                                f"{compound_name}_components",
                                result['components_plot_data']
                            )
                            st.caption("混合物成分贡献图")
                    
                    with col_figs2:
                        # 物理效应图
                        if result['fig_effects']:
                            st.pyplot(result['fig_effects'])
                            create_download_buttons(
                                result['fig_effects'], 
                                f"{compound_name}_effects",
                                result['effects_plot_data']
                            )
                            st.caption("物理效应贡献图")
                    
                    # 透射率图
                    st.subheader("混合物透射率")
                    if result['fig_transmission']:
                        st.pyplot(result['fig_transmission'])
                        create_download_buttons(
                            result['fig_transmission'], 
                            f"{compound_name}_transmission",
                            result['transmission_plot_data']
                        )
                        st.caption("透射率图")
                else:
                    # 仅显示透射率图
                    if result['fig']:
                        st.pyplot(result['fig'])
                        compound_name = st.session_state.compound_formula.replace(" ", "")
                        create_download_buttons(
                            result['fig'], 
                            f"{compound_name}_transmission",
                            result['transmission_plot_data']
                        )
                        st.caption("透射率图")
                    
                    # 显示关键能量点的透射率
                    if len(result['key_energies']) > 0:
                        st.subheader("Key Point Transmission")
                        data = {
                            "Energy (MeV)": result['key_energies'],
                            "Transmission": [f"{t:.4f}" for t in result['key_transmissions']]
                        }
                        st.table(data)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>X-Ray Transmission Calculator | Interactive Interface Based on Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 