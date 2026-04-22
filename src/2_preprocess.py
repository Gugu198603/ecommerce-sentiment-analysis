import pandas as pd
import jieba
import re
from tqdm import tqdm
from utils import DIRS, ensure_dirs, setup_logging

logger = setup_logging()
# 启用 tqdm 的 pandas 扩展
tqdm.pandas()

def clean_text(text):
    """
    清洗文本内容：去URL、去特殊符号、去换行符等
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？、]', '', text)
    return text.strip()

def tokenize(text, stopwords=None):
    """
    中文分词并去停用词
    """
    if not stopwords:
        stopwords = set()
    words = jieba.cut(text)
    return ' '.join([w for w in words if w.strip() and w not in stopwords])

def main():
    ensure_dirs()
    input_file = DIRS['raw_data'] / "raw_comments.csv"
    output_file = DIRS['processed_data'] / "cleaned_data.csv"
    
    if not input_file.exists():
        logger.error(f"未找到原始数据文件 {input_file}，请先运行 1_crawler.py")
        return
        
    logger.info("1. 开始读取数据...")
    df = pd.read_csv(input_file)
    logger.info(f"初始数据量: {len(df)} 条")
    
    # 2. 去重和去空
    df.drop_duplicates(subset=['content'], inplace=True)
    df.dropna(subset=['content'], inplace=True)
    
    # 3. 数据清洗与过短评论剔除
    logger.info("2. 清洗特殊字符...")
    df['clean_text'] = df['content'].progress_apply(clean_text)
    df = df[df['clean_text'].str.len() >= 4] # 过滤掉长度小于4的无意义评论
    
    # 4. 生成情感标签 (评分≥4为1，≤2为0，3分为中性暂时丢弃)
    logger.info("3. 自动生成情感标签...")
    def map_sentiment(score):
        if pd.isna(score): return -1
        score = int(score)
        if score >= 4: return 1
        elif score <= 2: return 0
        else: return -1 # 丢弃3分
        
    df['sentiment'] = df['score'].apply(map_sentiment)
    df = df[df['sentiment'] != -1]
    
    # 5. 中文分词 (jieba)
    logger.info("4. 进行中文分词 (Jieba)...")
    df['tokenized'] = df['clean_text'].progress_apply(lambda x: tokenize(x))
    
    # 6. 数据平衡 (正负样本 1:1)
    logger.info("5. 平衡正负样本...")
    pos_df = df[df['sentiment'] == 1]
    neg_df = df[df['sentiment'] == 0]
    
    min_count = min(len(pos_df), len(neg_df))
    if min_count == 0:
        logger.error("错误：正样本或负样本数量为0，请检查爬虫逻辑。")
        return
        
    # 如果某类太少，使用上采样(oversampling) 或下采样(undersampling)
    # 此处为保证数据量，如果极度不平衡，则采样达到一致
    pos_df = pos_df.sample(n=max(len(pos_df), len(neg_df)), replace=True, random_state=42)
    neg_df = neg_df.sample(n=max(len(pos_df), len(neg_df)), replace=True, random_state=42)
    
    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 7. 保存结果
    balanced_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"预处理完成！平衡后数据总量: {len(balanced_df)} 条 (正负各半)。")
    logger.info(f"已保存至: {output_file}")

if __name__ == '__main__':
    main()