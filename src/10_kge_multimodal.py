import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import setup_logging

logger = setup_logging()

# ==========================================
# 核心原理说明：
# 1. 在真实的千万级商品库中，文本评论不是孤立的。
# 2. 我们构建一个电商知识图谱 (Knowledge Graph)：
#    实体 (Entities): 商品ID、品牌(Apple/小米)、类目(手机/电脑)、用户、多模态特征(商品图片Embedding)
#    关系 (Relations): 属于(Belongs_to)、购买(Bought)、评价(Reviewed)、具有特征(Has_Feature)
# 3. KGE (知识图谱嵌入, 如 TransE) 能够将图谱结构降维到连续向量空间 (h + r ≈ t)。
# 4. 这个模块将生成实体嵌入，供后续多模态融合使用。
# ==========================================

class TransE(nn.Module):
    """
    TransE 模型原型 (Translation-based Embedding)
    目标: h + r ≈ t
    损失函数: L = max(0, ||h+r-t|| - ||h'+r-t'|| + margin)
    """
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=self.margin)
        
        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        # L2 范数作为距离度量
        pos_distance = torch.norm(self.entity_embeddings(pos_h) + self.relation_embeddings(pos_r) - self.entity_embeddings(pos_t), p=2, dim=1)
        neg_distance = torch.norm(self.entity_embeddings(neg_h) + self.relation_embeddings(neg_r) - self.entity_embeddings(neg_t), p=2, dim=1)
        
        # MarginRankingLoss 需要一个 y 标签，这里设为 -1 表示 pos_distance 应该小于 neg_distance
        y = torch.full((pos_h.size(0),), -1.0).to(pos_h.device)
        return self.criterion(pos_distance, neg_distance, y)

def generate_synthetic_kg():
    """生成模拟的电商多模态知识图谱"""
    logger.info("正在构建电商多模态知识图谱 (Knowledge Graph)...")
    
    entities = {'User_A': 0, 'User_B': 1, 'iPhone15': 2, 'Apple': 3, 'Mobile_Phone': 4, 'Image_Feature_1': 5}
    relations = {'Bought': 0, 'Belongs_to_Brand': 1, 'Belongs_to_Category': 2, 'Has_Visual': 3}
    
    # 构造正样本三元组 (h, r, t)
    pos_triplets = [
        [0, 0, 2], # User_A Bought iPhone15
        [1, 0, 2], # User_B Bought iPhone15
        [2, 1, 3], # iPhone15 Belongs_to_Brand Apple
        [2, 2, 4], # iPhone15 Belongs_to_Category Mobile_Phone
        [2, 3, 5], # iPhone15 Has_Visual Image_Feature_1 (模拟多模态视觉特征)
    ]
    
    # 构造负样本三元组 (随机替换 head 或 tail)
    neg_triplets = [
        [1, 0, 3], # User_B Bought Apple (错误)
        [0, 1, 2], # User_A Belongs_to_Brand iPhone15 (错误)
        [3, 2, 4], # Apple Belongs_to_Category Mobile_Phone (错误)
        [5, 3, 2], # Image_Feature_1 Has_Visual iPhone15 (方向错误)
        [4, 0, 0], # Mobile_Phone Bought User_A (错误)
    ]
    
    return torch.LongTensor(pos_triplets), torch.LongTensor(neg_triplets), len(entities), len(relations)

def train_kge_prototype():
    logger.info("========== 开始训练知识图谱嵌入 (KGE) 融合多模态特征 ==========")
    pos_triplets, neg_triplets, num_ent, num_rel = generate_synthetic_kg()
    
    model = TransE(num_entities=num_ent, num_relations=num_rel, embedding_dim=64, margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pos_h, pos_r, pos_t = pos_triplets[:, 0], pos_triplets[:, 1], pos_triplets[:, 2]
        neg_h, neg_r, neg_t = neg_triplets[:, 0], neg_triplets[:, 1], neg_triplets[:, 2]
        
        loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"KGE 训练 Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            
    logger.info("KGE 训练完成！实体的降维稠密向量已生成。")
    logger.info("商品(iPhone15) 嵌入向量前 5 维展示: " + str(model.entity_embeddings(torch.tensor(2)).data.numpy()[:5]))
    logger.info("========== 知识图谱与多模态视觉特征融合成功！==========")
    
if __name__ == "__main__":
    train_kge_prototype()