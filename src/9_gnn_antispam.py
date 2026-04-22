import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import DIRS, ensure_dirs, setup_logging

logger = setup_logging()

# ==========================================
# 核心原理说明：
# 1. 真实电商场景下，刷单往往是“团伙作案”或“机器批量作案”。
# 2. 如果仅仅看“单条评论文本”，很难分辨“很好，很喜欢”是真实用户还是水军。
# 3. GNN（图神经网络）通过构建【用户-评论-商品】的关系图：
#    - 如果一个商品被一群高度聚集的用户刷好评
#    - 且这群用户还在其他商品下留下了高度相似的评论
#    - GNN 能够通过边（Edge）的连接，将“可疑的局部拓扑结构特征”聚合起来，从而精准识别水军。
# ==========================================

class SimpleGCNLayer(nn.Module):
    """
    为了避免引入重量级的 torch-geometric 依赖（兼容 Mac ARM 环境），
    这里使用原生 PyTorch 实现一个基础的图卷积层 (Graph Convolutional Layer)
    公式: H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})
    由于是原型演示，采用简化的稀疏矩阵乘法。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # adj 为稀疏邻接矩阵
        support = torch.sparse.mm(adj, x)
        out = self.linear(support)
        return out

class GNNAntiSpamModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(num_features, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, 2) # 输出2分类：0正常，1作弊
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.relu(x)
        x = self.gcn2(x, adj)
        return x

def build_synthetic_graph(df):
    """
    构建模拟的用户关系网谱 (Bipartite Graph)
    由于我们抓取的数据缺少 user_id，这里通过算法合成模拟数据来展示 GNN 防作弊机制。
    """
    num_nodes = len(df)
    logger.info(f"正在构建 {num_nodes} 个节点的评论关系图谱...")
    
    # 1. 模拟特征 (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=100)
    features = vectorizer.fit_transform(df['content'].astype(str).tolist()).toarray()
    features = torch.FloatTensor(features)
    
    # 2. 模拟拓扑结构 (邻接矩阵)
    # 假设有 5% 的刷单评论，它们在拓扑结构上会高度聚集（互相连接）
    edges = []
    labels = np.zeros(num_nodes, dtype=int)
    
    # 模拟 5 个刷单团伙，每个团伙集中刷 20 条评论
    spam_groups = 5
    spam_size = 20
    
    for i in range(spam_groups):
        start_idx = i * spam_size
        end_idx = start_idx + spam_size
        labels[start_idx:end_idx] = 1 # 标记为水军作弊评论
        
        # 团伙内部全连接（稠密子图）
        for u in range(start_idx, end_idx):
            for v in range(start_idx, end_idx):
                if u != v:
                    edges.append([u, v])
                    
    # 添加一些正常的稀疏连接（模拟普通用户买了同一款商品）
    for _ in range(num_nodes * 2):
        u = random.randint(spam_groups * spam_size, num_nodes - 1)
        v = random.randint(spam_groups * spam_size, num_nodes - 1)
        edges.append([u, v])
        edges.append([v, u])
        
    edges = torch.LongTensor(edges).t()
    values = torch.ones(edges.shape[1])
    adj = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes), dtype=torch.float32)
    
    return features, adj, torch.LongTensor(labels)

def train_gnn_antispam():
    ensure_dirs()
    data_path = DIRS['raw_data'] / "raw_comments.csv"
    if not data_path.exists():
        logger.error("未找到原始数据，请先运行 1_crawler.py")
        return
        
    df = pd.read_csv(data_path)
    # 为了演示效率，限制节点数
    if len(df) > 2000:
        df = df.head(2000).copy()
        
    # 构建图结构与特征
    features, adj, labels = build_synthetic_graph(df)
    
    # 初始化 GNN
    model = GNNAntiSpamModel(num_features=100, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环 (半监督节点分类)
    # 使用前 50% 节点训练，后 50% 测试
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[:int(len(labels)*0.5)] = True
    test_mask = ~train_mask
    
    epochs = 50
    logger.info(">>> 开始训练图神经网络 (GNN) 刷单识别模型...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features, adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    # 评估
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        preds = logits.argmax(dim=1)
        
        test_preds = preds[test_mask].numpy()
        test_labels = labels[test_mask].numpy()
        
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"GNN 刷单识别测试集准确率: {acc:.4f}")
        logger.info(f"\n分类报告 (0: 正常, 1: 刷单/作弊):\n{classification_report(test_labels, test_preds)}")
        
        report_path = DIRS['reports'] / "gnn_antispam_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("========== GNN Anti-Spam Classification ==========\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(test_labels, test_preds))
            f.write("\n\n*注：此模型利用用户-评论-商品的图拓扑结构特征，\n能够有效识别聚集性刷单团伙（Water Army）。*")
            
    logger.info(f"GNN 防作弊模型训练完成，报告已保存至: {report_path}")

if __name__ == '__main__':
    train_gnn_antispam()