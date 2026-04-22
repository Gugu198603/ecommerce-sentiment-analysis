import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from utils import DIRS, ensure_dirs, setup_logging

logger = setup_logging()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert():
    ensure_dirs()
    data_path = DIRS['processed_data'] / "cleaned_data.csv"
    model_save_path = DIRS['models']
    
    if not data_path.exists():
        logger.error(f"未找到预处理数据 {data_path}")
        return

    logger.info("1. 正在加载数据...")
    df = pd.read_csv(data_path)
    
    # 为了演示效率，限制一下训练集大小，否则没有GPU需要跑很久
    # 如果有GPU则全量跑
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == 'cpu' and len(df) > 2000:
        logger.warning("未检测到GPU，为了保证能在课程设计期间跑完，将采样 2000 条数据进行训练演示。")
        df = df.sample(2000, random_state=42)

    X = df['clean_text'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("2. 初始化 BERT Tokenizer...")
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    logger.info(f"3. 加载预训练模型 ({device})...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    # PyTorch 自带的 AdamW 参数不需要 correct_bias
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    logger.info("4. 开始训练 BERT...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 训练循环
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    logger.info("5. 开始评估模型...")
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    logger.info(f"测试集准确率 (Accuracy): {acc:.4f}")
    logger.info(f"\n分类报告 (Classification Report):\n{report}")
    
    # 保存分类报告
    with open(DIRS['reports'] / "bert_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    logger.info("6. 保存模型与Tokenizer...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"BERT模型已保存至: {model_save_path}")

if __name__ == '__main__':
    train_bert()