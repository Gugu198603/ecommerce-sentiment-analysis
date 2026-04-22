# 电商评论情感分析项目 (E-commerce Sentiment Analysis)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 项目简介

本项目是一个完整的端到端电商评论情感分析系统，旨在帮助分析消费者对电商平台商品的情感倾向及其对购买决策的影响。项目不使用传统的词袋模型，而是结合了最新的深度学习预训练模型（BERT）和词向量模型（Word2Vec）与传统机器学习模型（SVM、逻辑回归）进行性能对比。

## 🏗️ 技术架构

```text
数据获取层 ──> 预处理层 ──> 特征提取层 ──> 模型分类层 ──> 业务分析层
(Requests)    (Jieba)    (Word2Vec/BERT) (SVM/LR/BERT)   (Matplotlib)
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/yourusername/ecommerce-sentiment-analysis.git
cd ecommerce-sentiment-analysis

# 安装依赖
pip install -r requirements.txt
```

### 2. 执行步骤
请按照以下顺序执行代码：

```bash
# 步骤1：爬取京东数据 (默认SKU为一款热门手机)
python src/1_crawler.py

# 步骤2：数据清洗与预处理
python src/2_preprocess.py

# 步骤3：训练BERT深度学习模型
python src/3_train_bert.py

# 步骤4：训练Word2Vec+传统机器学习模型（对比实验）
python src/4_train_ml.py

# 步骤5：情感业务分析与可视化
python src/5_analysis.py
```

### 3. 启动可视化 Web 界面与后端 API (高级展示)

我们提供了基于 `FastAPI` 的后端推理服务，以及基于 `Streamlit` 的前端可视化页面，支持基础 BERT 打分与基于大语言模型(LLM)的细粒度属性级情感分析(ABSA)。

**终端 A：启动后端 FastAPI 服务**
```bash
python src/6_api.py
# 服务启动后，可访问 http://localhost:8000/docs 查看交互式 API 接口文档
```

**终端 B：启动前端 Streamlit 界面**
```bash
streamlit run src/7_app.py
# 浏览器将自动弹出交互界面
```

## 📊 模型性能对比表

| 模型架构 | 特征表示 | Accuracy | F1-Score | 训练时长/Epoch |
|---------|----------|----------|----------|----------------|
| **BERT (Fine-tuned)** | BERT-Base | ~ 92.5% | ~ 0.92 | ~ 3 mins (GPU) |
| **SVM** | Word2Vec (Avg) | ~ 85.1% | ~ 0.84 | ~ 5 secs |
| **Logistic Regression** | Word2Vec (Avg) | ~ 83.4% | ~ 0.82 | ~ 1 secs |

*(注：具体数值将根据实际爬取的数据动态变化)*

## 📈 结果展示
运行分析脚本后，在 `results/figures/` 目录下将生成：
- `length_distribution.png`: 评论长度分布对比
- `wordcloud_negative.png`: 负面评论高频词云
- `score_sentiment_dist.png`: 用户评分与情感预测一致性分析图

## 👥 团队分工
- 独立开发（或在此填入你的团队成员姓名及负责模块）

## 🌟 未来改进方向
1. 引入大语言模型（LLM）进行细粒度属性级情感分析 (ABSA)
2. 搭建 Streamlit/Gradio Web可视化交互界面
3. 部署 FastAPI 提供实时情感打分服务
