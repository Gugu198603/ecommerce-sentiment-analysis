# 项目运行与数据报告

生成时间：2026-06-05  
运行环境：本地 `.venv`，标准 PyTorch LightGCN 使用 CPU 训练。

## 1. 本次运行范围

本次已跑通当前项目主链路：

| 模块 | 脚本 | 运行状态 | 主要输出 |
| --- | --- | --- | --- |
| 因果推断 | `src/causal_inference.py` | 成功 | `results/reports/causal_report.txt`、`results/figures/psm_balance.png` |
| 轻量推荐 | `src/recommender.py` | 成功 | `recommend_result.csv`、`recommend_result_sentiment.csv`、`sentiment_alpha_sweep.csv` |
| 标准 LightGCN | `src/lightgcn_torch.py --causal-mode residual --causal-alpha 5` | 成功 | `recommend_result_torch.csv`、`recommendation_metrics_torch.txt` |
| 业务洞察 | `src/5_analysis.py` | 成功 | `business_insights.txt`、`wordcloud_negative.png`、`length_distribution.png` |

说明：爬虫、BERT 训练、API/可视化服务没有纳入本次“全链路重跑”，因为当前推荐实验主线依赖的是已处理数据和推荐/因果模块。

## 2. 数据集概况

主数据文件：`data/processed/cleaned_data.csv`

| 指标 | 数值 |
| --- | ---: |
| 评论总数 | 36193 |
| 用户数 | 27999 |
| 商品数 | 37 |
| 平均评分 | 4.8396 |
| 正反馈数量（score >= 4） | 34660 |
| 正反馈比例 | 95.76% |
| 至少交互 2 个商品的用户数 | 3102 |
| 至少交互 3 个商品的用户数 | 1831 |
| sentiment 均值 | 0.7889 |
| sentiment 标准差 | 0.3298 |

字段含义：

| 字段 | 含义 | 在项目中的作用 |
| --- | --- | --- |
| `user_id` | 匿名用户 ID | 构建用户节点 |
| `product_id` | 商品 ID | 构建商品节点 |
| `content` | 原始评论 | 情感分析与解释来源 |
| `score` | 1-5 星评分 | 推荐正反馈和因果 outcome |
| `time` | 评论时间 | 推荐训练/测试切分 |
| `useful_vote` | 有用票数 | 可作为评论质量特征 |
| `sentiment` | 显式情感分数 | 情感增强与因果 treatment 构造 |
| `clean_text` | 清洗文本 | NLP 处理输入 |
| `tokenized` | 分词文本 | 文本建模输入 |
| `label` | 情感标签 | 正负面辅助标签 |

## 3. 情感特征分布

### 3.1 细粒度情感向量

文件：`data/processed/sentiment_vectors.csv`

| 指标 | 数值 |
| --- | ---: |
| 向量样本数 | 36193 |
| 向量维度 | 30 |
| 全部向量值均值 | 0.5029 |
| 全部向量值中位数 | 0.5000 |
| 全部向量值标准差 | 0.0236 |
| 精确等于 0.5 的比例 | 97.16% |
| 行均值落在 0.5±0.02 的比例 | 98.16% |

结论：细粒度情感向量明显集中在中性值 `0.5` 附近，因此它对推荐排序的直接区分度有限。

### 3.2 隐式情感分数

文件：`data/processed/implicit_sentiment_full.csv`

| 指标 | 数值 |
| --- | ---: |
| 有效读取样本数 | 36226 |
| `implicit_score = -1` 数量 | 32136 |
| 有效隐式分数数量（0-1） | 4090 |
| 有效隐式分数比例 | 11.29% |
| 有效隐式分数均值 | 0.6144 |
| 有效隐式分数标准差 | 0.2258 |

说明：`implicit_score = -1` 表示评论已有显式情感，因此未走隐式情感推理。推荐模块中已采用回退策略：如果是 `-1`，使用显式情感 `sentiment` 替代。

## 4. 推荐实验结果

### 4.1 轻量 LightGCN-style 与情感增强

| 模型 | Recall@10 | NDCG@10 | 说明 |
| --- | ---: | ---: | --- |
| Vanilla LightGCN-style | 0.306543 | 0.137126 | 只使用用户-商品交互 |
| 情感增强重排（alpha=3.0） | 0.309532 | 0.138932 | 融合细粒度情感向量和隐式情感 |

结论：情感增强重排相对轻量基础版有小幅提升，说明情感特征对排序有辅助作用，但受限于向量中性化和数据稀疏，提升幅度有限。

### 4.2 标准 PyTorch LightGCN + 因果边权

| 指标 | 数值 |
| --- | ---: |
| 模型 | Standard PyTorch LightGCN with BPR Loss + causal edge weights |
| Embedding 维度 | 32 |
| 图传播层数 | 2 |
| Epochs | 120 |
| causal_mode | residual |
| causal_alpha | 5.0 |
| Causal forest available | True |
| Recall@10 | 0.344072 |
| NDCG@10 | 0.165827 |

结论：标准 PyTorch LightGCN 明显优于轻量 LightGCN-style，是当前推荐模块更适合作为正式汇报的主结果。

## 5. 因果推断结果

文件：`results/reports/causal_report.txt`

| 指标 | 数值 |
| --- | ---: |
| Treatment 定义 | overall_sentiment > 0.5 |
| PSM 匹配对数 | 5848 |
| ATE 平均处理效应 | +0.6111 |
| ATT 处理组平均处理效应 | +0.6100 |

因果森林特征重要性 Top 结果：

| 排名 | 特征 | 重要性 |
| --- | --- | ---: |
| 1 | 评论长度 | 0.5350 |
| 2 | 属性情感分歧度 | 0.2577 |
| 3 | 属性情感均值 | 0.2350 |
| 4 | 属性多样性 | 0.0113 |

结论：情感倾向对评分存在正向因果效应；评论长度和属性情感相关变量是关键解释因素。

## 6. 业务洞察

文件：`results/reports/business_insights.txt`

| 指标 | 数值 |
| --- | ---: |
| 好评平均长度 | 199.9 字 |
| 差评平均长度 | 367.2 字 |

结论：差评往往更长，说明用户在负面体验中会更详细描述问题。负面词云建议重点关注“包装、物流、质量”等痛点。

## 7. 当前问题与限制

- 商品数只有 37 个，实验更适合定位为“手机类目内推荐”，不是全品类推荐。
- 用户交互较稀疏，很多用户只评论过少量商品，限制了 LightGCN 的协同过滤效果。
- 细粒度情感向量大量集中在 0.5，中性值过多，导致情感特征区分度不足。
- 隐式情感有效覆盖率较低，只有部分样本有 0-1 的隐式情感分数。
- 因果加权已经接入标准 LightGCN，但提升幅度仍受数据规模和边权扰动强度限制。

## 8. 主要输出文件

| 文件 | 说明 |
| --- | --- |
| `results/reports/recommend_result.csv` | 轻量基础推荐结果 |
| `results/reports/recommend_result_sentiment.csv` | 情感增强重排结果 |
| `results/reports/recommend_result_torch.csv` | 标准 PyTorch LightGCN 推荐结果 |
| `results/reports/recommendation_metrics.txt` | 轻量版和情感增强版指标 |
| `results/reports/recommendation_metrics_torch.txt` | 标准 PyTorch LightGCN 指标 |
| `results/reports/causal_report.txt` | PSM + Causal Forest 因果报告 |
| `results/reports/business_insights.txt` | 业务洞察报告 |
| `results/figures/psm_balance.png` | PSM 平衡性图 |
| `results/figures/wordcloud_negative.png` | 负面词云 |
| `results/figures/length_distribution.png` | 评论长度分布图 |

## 9. 汇报结论

本项目已经跑通“评论数据清洗 → 情感特征构建 → 因果分析 → LightGCN 推荐训练 → 推荐评估”的主流程。标准 PyTorch LightGCN + 因果边权版本当前取得 `Recall@10=0.344072`、`NDCG@10=0.165827`。情感与因果信息已经接入推荐流程，但由于数据集商品数较少、用户交互稀疏、情感向量中性化严重，增益幅度有限。后续优化重点应放在扩大商品规模、优化细粒度情感向量、将情感作为图边权更深度参与训练。
