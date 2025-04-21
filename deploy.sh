#!/bin/bash

# X射线透射计算器部署脚本

echo "====== 开始部署X射线透射计算器 ======"

# 确保 Docker 和 Docker Compose 已安装
if ! command -v docker &> /dev/null; then
    echo "错误: 未找到Docker，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "错误: 未找到Docker Compose，请先安装Docker Compose"
    exit 1
fi

# 确保目录存在
mkdir -p tmp_plots

# 构建并启动容器
echo "正在构建和启动容器..."
docker-compose up -d --build

# 检查容器状态
echo "正在检查容器状态..."
sleep 5
if [ "$(docker ps -q -f name=xray-calculator)" ]; then
    echo "部署成功! 访问 http://localhost:5000 查看应用"
    echo "或使用服务器IP替代localhost"
else
    echo "部署失败，请检查日志: docker logs xray-calculator"
fi

echo "====== 部署过程完成 ======" 