import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# 去掉 gensim 依赖
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from utils import DIRS, ensure_dirs, setup_logging

logger = setup_logging()

def train_ml():
    ensure_dirs()
    data_path = DIRS['processed_data'] / "cleaned_data.csv"
    
    if not data_path.exists():
        logger.error(f"未找到预处理数据 {data_path}")
        return

    logger.info("1. 加载数据与分词结果...")
    df = pd.read_csv(data_path)
    df.dropna(subset=['tokenized'], inplace=True)
    
    # 获取特征和标签
    texts = df['tokenized'].values
    y = df['sentiment'].values
    
    logger.info("2. 提取 TF-IDF 特征 (由于环境限制，暂用 TF-IDF 替代 Word2Vec 作为传统模型基线)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"3. 训练传统机器学习模型...")
    
    results = {}
    
    # --- 逻辑回归 ---
    logger.info(">>> 正在训练 Logistic Regression...")
    lr_start = time.time()
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_time = time.time() - lr_start
    lr_acc = accuracy_score(y_test, lr_preds)
    results['Logistic Regression'] = {'Accuracy': lr_acc, 'Time': lr_time}
    logger.info(f"LR 准确率: {lr_acc:.4f} (耗时: {lr_time:.2f}s)")
    
    # --- 支持向量机 (SVM) ---
    logger.info(">>> 正在训练 Support Vector Machine (SVM)...")
    svm_start = time.time()
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_time = time.time() - svm_start
    svm_acc = accuracy_score(y_test, svm_preds)
    results['SVM'] = {'Accuracy': svm_acc, 'Time': svm_time}
    logger.info(f"SVM 准确率: {svm_acc:.4f} (耗时: {svm_time:.2f}s)")
    
    # 输出对比报告
    logger.info("\n========= 传统机器学习性能对比 =========")
    for model_name, metrics in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy : {metrics['Accuracy']:.4f}")
        logger.info(f"  Time     : {metrics['Time']:.4f} s")
        
    report = classification_report(y_test, svm_preds)
    with open(DIRS['reports'] / "ml_report.txt", "w", encoding="utf-8") as f:
        f.write("========== ML Models Comparison ==========\n")
        f.write(f"LR  Accuracy: {lr_acc:.4f}\n")
        f.write(f"SVM Accuracy: {svm_acc:.4f}\n\n")
        f.write("========== SVM Classification Report ==========\n")
        f.write(report)
        
    logger.info("对比实验完成，报告已保存。")

if __name__ == '__main__':
    train_ml()