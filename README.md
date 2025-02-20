# AI-Generated-Detection

## 项目概述
本项目旨在构建一个基于深度学习的Web应用，利用GoogleNet Inception v3模型实现自动检测图像是否为AI生成（如GAN、Stable Diffusion、DALL-E等生成器创建）。该网站将提供用户友好的界面，支持图像上传并实时返回检测结果。

## 项目结构
- `GoogleNetV3.py`  
  ▸ Inception v3模型架构实现  
  ▸ 自定义数据集加载器
- `train.py`  
  ▸ 模型训练/验证流程  
- `main.py`  
  ▸ Flask Web服务入口  
  ▸ 包含路由和预测接口
- `pre-train.png`  
  ▸ 使用ImageNet预训练参数的loss曲线
- `no-pre-train.png`  
  ▸ 未使用预训练参数的loss对比
