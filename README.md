# X射线透射计算器 (X-Ray Transmission Calculator)

这是一个基于Streamlit构建的交互式X射线透射计算工具，允许用户计算和可视化元素、化合物和混合物对X射线的透射特性。

## 主要功能

### 元素透射计算
- 单元素透射率计算和可视化
- 截面系数分析（光电效应、相干散射、非相干散射等）
- 可调节能量范围、密度和厚度

### 化合物透射计算
- 化学式解析（如H2O, CaCO3, Gd3Al2Ga3O12等）
- 化合物各元素贡献分析
- 物理效应贡献分析
- 透射率计算和可视化

### 混合物透射计算
- 多组分混合物配方设计
- 成分配比调整和归一化
- 混合物透射特性计算
- 各组分贡献分析

### 数据导出
- 所有计算结果可导出为Excel格式
- 支持图表数据的批量下载

## 安装方法

### 使用pip安装依赖

```bash
pip install -r requirements.txt
```

### 运行Streamlit应用

```bash
streamlit run streamlit_app.py
```

## Docker部署

本项目提供了Docker部署支持，详见`Dockerfile`和`docker-compose.yml`文件。

### 使用Docker Compose启动

```bash
docker-compose up -d
```

## 使用方法

1. 启动应用后，通过浏览器访问应用（默认地址：http://localhost:5000）
2. 在界面上选择"Elements"、"Compounds"或"Mixtures"标签页
3. 输入相关参数（元素符号、化学式、密度、厚度、能量范围等）
4. 点击"Calculate"按钮生成结果
5. 查看图表并根据需要下载数据

## 数据来源

本应用使用NIST XCOM数据库的元素X射线截面数据，涵盖1keV至20MeV能量范围。

## 注意事项

- 复杂化合物可能需要正确的化学式格式，遵循大小写规范（如H2O而非h2o）
- 推荐使用Chrome或Firefox浏览器以获得最佳体验

## 许可证

本项目仅用于教育和研究目的。 