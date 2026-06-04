import os
import re
import json
import sys
import pandas as pd
import jieba
import jieba.posseg as pseg
import requests
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from tqdm import tqdm

# ==================== 配置参数（预处理） ====================
FILE_PATH = "../data/raw/str_review.csv"
OUTPUT_CSV_CLEAN = "../data/processed/cleaned_data.csv"
OUTPUT_TRAIN = "../data/processed/train.csv"
OUTPUT_VAL = "../data/processed/val.csv"
OUTPUT_TEST = "../data/processed/test.csv"

MIN_COMMENT_LEN = 5
ENABLE_DEDUPLICATION = True
MAX_CLEAN_LEN = 2000

LABEL_THRESHOLD = 0.5
TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_STATE = 42

DO_BALANCE = False
MAX_IMBALANCE_RATIO = 1.5

STOPWORDS_FILE = "stopwords.txt"
STOPWORDS_URLS = [
    "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
]

# ==================== 配置参数（细粒度情感分析 - 规则版） ====================
ASPECT_OUTPUT_JSONL = "../data/processed/aspect_sentiment.jsonl"
ASPECT_OUTPUT_VECTORS = "../data/processed/sentiment_vectors.csv"
ASPECT_OUTPUT_VOCAB = "../data/processed/aspect_vocab.txt"
ASPECT_SAMPLE_SIZE = None
ASPECT_MIN_FREQ = 3                # 属性词最少出现次数
ASPECT_TOP_K = 30                  # 保留的最大属性数
ASPECT_WINDOW_SIZE = 5             # 窗口大小（词数）
ASPECT_DEFAULT_SENTIMENT = 0.5     # 未出现属性的默认值

# 内置情感词典（可扩展）
POS_WORDS = {
    '快', '迅速', '及时', '好', '不错', '满意', '喜欢', '爱', '棒', '赞',
    '清晰', '流畅', '稳定', '强', '给力', '优秀', '完美', '惊艳', '良心',
    '性价比高', '耐用', '省电', '亮', '细腻', '轻便', '舒适', '漂亮', '好看',
    '满意', '惊喜', '超值', '推荐', '值得', '放心', '认真', '耐心', '给力',
    '物超所值', '质量好', '速度快', '服务好', '效果好'
}
NEG_WORDS = {
    '慢', '卡', '顿', '差', '烂', '糟糕', '失望', '后悔', '辣鸡', '垃圾',
    '失望', '痛心', '简陋', '摇晃', '响', '不够', '不足', '勉强', '一般',
    '凑合', '不行', '麻烦', '难用', '发热', '烫', '掉电', '闪退', '模糊',
    '差评', '垃圾', '恶心', '坑', '骗', '假货', '破', '旧', '脏', '辣鸡',
    '卡顿', '死机', '后悔', '不值', '问题'
}
NEGATION_WORDS = {'不', '不是', '没', '没有', '无', '非', '别', '勿', '莫', '弗', '未', '否', '并非', '不可'}

# ==================== 辅助函数（预处理） ====================
def download_stopwords(filename=STOPWORDS_FILE):
    if os.path.exists(filename):
        print(f"停用词文件 {filename} 已存在，跳过下载。")
        return True
    print(f"未找到停用词文件 {filename}，开始下载...")
    for url in STOPWORDS_URLS:
        try:
            print(f"尝试从 {url} 下载...")
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(resp.text)
                print(f"✅ 停用词表下载成功，已保存为 {filename}")
                return True
        except Exception as e:
            print(f"下载出错: {e}")
            continue
    print("❌ 所有下载链接均失败，请手动下载停用词表并命名为 stopwords.txt 放在当前目录。")
    return False

def load_stopwords(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])

def clean_text_only_chinese(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5。，！？；：、“”‘’、]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

def tokenize_and_remove_stopwords(text, stopwords):
    if not text:
        return ""
    words = jieba.lcut(text)
    if stopwords:
        words = [w for w in words if w not in stopwords and w.strip()]
    return ' '.join(words)

def generate_report(df_original, df_clean, df_train, df_val, df_test):
    print("\n" + "="*60)
    print("                       数据预处理报告")
    print("="*60)
    original_count = len(df_original)
    clean_count = len(df_clean)
    print(f"\n📊 数据完整性")
    print(f"  原始数据总条数: {original_count}")
    print(f"  清洗后有效条数: {clean_count}")
    print("✅ 报告生成完毕")

# ==================== 预处理主函数 ====================
def run_preprocess():
    """执行数据预处理，生成清洗后数据及训练/验证/测试集"""
    if not download_stopwords(STOPWORDS_FILE):
        print("警告：没有停用词表，将跳过停用词过滤步骤。")
        stopwords = set()
    else:
        stopwords = load_stopwords(STOPWORDS_FILE)
        print(f"成功加载 {len(stopwords)} 个停用词")

    print("\n1. 加载原始数据...")
    df_raw = pd.read_csv(FILE_PATH)
    print(f"原始数据形状: {df_raw.shape}")

    col_mapping = {
        'product_id': 'product_id',
        'person_id': 'user_id',
        'review_content': 'content',
        'review_rating': 'score',
        'review_time': 'time',
        'review_helpful': 'useful_vote',
        'sentiments': 'sentiment'
    }
    exist_cols = [c for c in col_mapping.keys() if c in df_raw.columns]
    df = df_raw[exist_cols].rename(columns=col_mapping)
    print(f"保留列: {list(df.columns)}")

    init_len = len(df)
    df = df.dropna(subset=['content', 'sentiment'])
    print(f"删除缺失值: {init_len - len(df)} 条")

    if ENABLE_DEDUPLICATION:
        init_len = len(df)
        df = df.drop_duplicates(subset=['content', 'sentiment'])
        print(f"删除重复评论: {init_len - len(df)} 条")

    init_len = len(df)
    df = df[(df['sentiment'] >= 0) & (df['sentiment'] <= 1)]
    print(f"感情指标超出范围: {init_len - len(df)} 条")

    print("2. 对评论文本进行清洗（保留中文和常用标点）...")
    df['clean_text'] = df['content'].apply(clean_text_only_chinese)
    init_len = len(df)
    df = df[df['clean_text'].str.len() >= MIN_COMMENT_LEN]
    print(f"清洗后评论过短(长度<{MIN_COMMENT_LEN}): {init_len - len(df)} 条")
    df['clean_text'] = df['clean_text'].str[:MAX_CLEAN_LEN]

    print("3. 对清洗后文本进行分词并去除停用词...")
    df['tokenized'] = df['clean_text'].apply(lambda x: tokenize_and_remove_stopwords(x, stopwords))
    init_len = len(df)
    df = df[df['tokenized'].str.strip() != ""]
    print(f"分词后无有效词语: {init_len - len(df)} 条")

    df['label'] = df['sentiment'].apply(lambda x: 1 if x > LABEL_THRESHOLD else (0 if x < LABEL_THRESHOLD else None))
    init_len = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"删除中性样本 (sentiment=={LABEL_THRESHOLD}): {init_len - len(df)} 条")

    os.makedirs(os.path.dirname(OUTPUT_CSV_CLEAN), exist_ok=True)
    df.to_csv(OUTPUT_CSV_CLEAN, index=False, encoding='utf-8-sig')
    print(f"完整清洗数据已保存: {OUTPUT_CSV_CLEAN}")

    print("\n4. 进行数据集拆分...")
    X = df['tokenized']
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE+VAL_SIZE),
        stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"拆分结果: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")

    if DO_BALANCE:
        train_pos = (y_train == 1).sum()
        train_neg = (y_train == 0).sum()
        if train_pos > 0 and train_neg > 0:
            imbalance = max(train_pos, train_neg) / min(train_pos, train_neg)
            if imbalance > MAX_IMBALANCE_RATIO:
                print(f"训练集类别不平衡 ({train_pos}:{train_neg})，执行过采样...")
                try:
                    ros = RandomOverSampler(random_state=RANDOM_STATE)
                    X_train_resampled, y_train_resampled = ros.fit_resample(
                        X_train.values.reshape(-1, 1), y_train
                    )
                    X_train = pd.Series(X_train_resampled[:, 0], name='tokenized')
                    y_train = pd.Series(y_train_resampled, name='label')
                    print(f"过采样后: 正样本 {sum(y_train == 1)} 负样本 {sum(y_train == 0)}")
                except ImportError:
                    print("警告：未安装 imbalanced-learn，跳过过采样。")
        else:
            print("训练集只有一个类别，无法进行平衡")

    df_train = df.loc[X_train.index].copy()
    df_val = df.loc[X_val.index].copy()
    df_test = df.loc[X_test.index].copy()
    df_train['label'] = y_train.values if isinstance(y_train, pd.Series) else y_train
    df_val['label'] = y_val.values if isinstance(y_val, pd.Series) else y_val
    df_test['label'] = y_test.values if isinstance(y_test, pd.Series) else y_test

    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    df_train.to_csv(OUTPUT_TRAIN, index=False, encoding='utf-8-sig')
    df_val.to_csv(OUTPUT_VAL, index=False, encoding='utf-8-sig')
    df_test.to_csv(OUTPUT_TEST, index=False, encoding='utf-8-sig')
    print(f"\n数据集已保存:")
    print(f"  训练集: {OUTPUT_TRAIN} ({len(df_train)} 条)")
    print(f"  验证集: {OUTPUT_VAL} ({len(df_val)} 条)")
    print(f"  测试集: {OUTPUT_TEST} ({len(df_test)} 条)")

    generate_report(df_raw, df, df_train, df_val, df_test)
    print("\n✅ 预处理完成")

# ==================== 规则版细粒度情感分析 ====================
def extract_noun_phrases(text):
    """从文本中提取长度≥2的名词短语（用作候选属性词）"""
    words = pseg.cut(text)
    phrases = []
    cur = []
    for word, flag in words:
        if flag.startswith('n') or flag in ('vn', 'an'):
            if len(word) >= 2:
                cur.append(word)
        else:
            if len(cur) > 0:
                phrases.append(''.join(cur))
                cur = []
    if len(cur) > 0:
        phrases.append(''.join(cur))
    # 过滤纯数字
    return [p for p in phrases if not p.isdigit()]

def build_aspect_vocab_from_texts(texts, min_freq=3, top_k=30):
    """从文本列表中统计高频名词短语，构建属性词表"""
    print("📖 正在构建全局属性词表（统计高频名词短语）...")
    all_aspects = []
    for text in tqdm(texts, desc="抽取候选属性"):
        if not isinstance(text, str) or not text.strip():
            continue
        phrases = extract_noun_phrases(text)
        all_aspects.extend(phrases)
    freq = Counter(all_aspects)
    common = [(w, c) for w, c in freq.items() if c >= min_freq]
    common.sort(key=lambda x: x[1], reverse=True)

    blacklist = {
        '东西', '感觉', '时候', '这个', '那个', '可以', '没有', '还是', '就是', '一个',
        '我们', '自己', '什么', '但是', '因为', '所以', '如果', '然后', '已经', '还有',
        '真的', '只能', '使用', '购买', '收到', '整体', '第一次', '之前', '之后',
        '朋友', '家人', '孩子', '同事', '客服', '商家', '京东', '快递', '京东自营',
        '手机', '商品', '产品'  # 过于通用，可去掉
    }
    filtered = [(w, c) for w, c in common if w not in blacklist and len(w) >= 2]
    aspect_vocab = [w for w, _ in filtered[:top_k]]
    if not aspect_vocab:
        aspect_vocab = ['速度', '包装', '外观', '屏幕', '处理器', '反应', '网络',
                        '拍照', '电池', '续航', '音质', '手感', '价格', '物流', '服务', '质量']
        print("⚠️ 未抽取出属性词，使用默认列表")
    else:
        print(f"✅ 抽取到 {len(aspect_vocab)} 个高频属性词：{aspect_vocab[:10]}...")
    return aspect_vocab

def get_sentiment_score_for_aspect(text, aspect, pos_set, neg_set, negation_set, window=5):
    """
    在文本中查找aspect，在其前后window个词范围内计算情感得分。
    支持否定词反转。
    返回 0~1 之间的连续值。
    """
    if aspect not in text:
        return ASPECT_DEFAULT_SENTIMENT
    words = list(jieba.cut(text))
    # 查找aspect位置（简单包含匹配）
    positions = [i for i, w in enumerate(words) if aspect in w]
    if not positions:
        return ASPECT_DEFAULT_SENTIMENT
    pos_idx = positions[0]
    start = max(0, pos_idx - window)
    end = min(len(words), pos_idx + window + 1)
    window_words = words[start:end]

    score = 0.0
    negation_flag = False
    for w in window_words:
        if w in negation_set:
            negation_flag = not negation_flag
            continue
        if w in pos_set:
            delta = 1.0
            if negation_flag:
                delta = -delta
            score += delta
            negation_flag = False
        elif w in neg_set:
            delta = -1.0
            if negation_flag:
                delta = -delta
            score += delta
            negation_flag = False
    # 将score从[-window, window]映射到[0,1]
    # 使用 sigmoid-like 但简单 clip 后线性映射
    max_abs = window
    normalized = (score / max_abs) * 0.5 + 0.5
    normalized = max(0.0, min(1.0, normalized))
    return normalized

def extract_aspect_sentiment_rule_based(text, aspect_vocab, pos_set, neg_set, negation_set):
    """基于规则提取属性情感字典"""
    if not isinstance(text, str) or pd.isna(text) or not text.strip():
        return {}
    result = {}
    for asp in aspect_vocab:
        if asp in text:
            score = get_sentiment_score_for_aspect(text, asp, pos_set, neg_set, negation_set, ASPECT_WINDOW_SIZE)
            # 同一个属性可能多次出现，取平均值
            if asp in result:
                result[asp] = (result[asp] + score) / 2
            else:
                result[asp] = score
    return result

def dict_to_vector(aspect_sent, aspect_vocab, default=0.5):
    return [aspect_sent.get(asp, default) for asp in aspect_vocab]

def run_aspect_sentiment(input_csv_path=None):
    """执行基于规则的细粒度情感分析"""
    if input_csv_path is None:
        input_csv_path = OUTPUT_CSV_CLEAN
    print(f"\n📂 读取清洗后数据: {input_csv_path}")
    df = pd.read_csv(input_csv_path, encoding='utf-8')

    # 确定文本列（优先使用 clean_text 或 content）
    if 'clean_text' in df.columns:
        text_col = 'clean_text'
    elif 'content' in df.columns:
        text_col = 'content'
    elif 'review_content' in df.columns:
        text_col = 'review_content'
    else:
        raise ValueError("输入数据必须包含 'clean_text', 'content' 或 'review_content' 列")

    if 'product_id' not in df.columns and 'item_id' not in df.columns:
        raise ValueError("输入数据必须包含 product_id 或 item_id 列")
    if 'product_id' in df.columns:
        df['item_id'] = df['product_id'].astype(str)
    if 'user_id' not in df.columns:
        if 'person_id' in df.columns:
            df['user_id'] = df['person_id'].astype(str)
        elif 'reviewer_nickname' in df.columns:
            df['user_id'] = df['reviewer_nickname'].astype(str)
        else:
            df['user_id'] = 'user_' + df.index.astype(str)
    else:
        df['user_id'] = df['user_id'].astype(str)

    if ASPECT_SAMPLE_SIZE:
        df = df.head(ASPECT_SAMPLE_SIZE)
    print(f"✅ 加载数据完成，共 {len(df)} 条评论")

    # 构建属性词表（使用清洗后的文本）
    all_texts = df[text_col].tolist()
    aspect_vocab = build_aspect_vocab_from_texts(all_texts, min_freq=ASPECT_MIN_FREQ, top_k=ASPECT_TOP_K)

    # 保存词表
    os.makedirs(os.path.dirname(ASPECT_OUTPUT_VOCAB), exist_ok=True)
    with open(ASPECT_OUTPUT_VOCAB, 'w', encoding='utf-8') as f:
        f.write('\n'.join(aspect_vocab))
    print(f"✅ 属性词表已保存至 {ASPECT_OUTPUT_VOCAB}")

    # 准备情感词典
    pos_set = POS_WORDS
    neg_set = NEG_WORDS
    negation_set = NEGATION_WORDS

    jsonl_records = []
    vector_records = []
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc="细粒度情感分析（规则）"):
        text = row[text_col]
        aspect_sent = extract_aspect_sentiment_rule_based(text, aspect_vocab, pos_set, neg_set, negation_set)
        vector = dict_to_vector(aspect_sent, aspect_vocab, ASPECT_DEFAULT_SENTIMENT)

        jsonl_records.append({
            "user_id": row['user_id'],
            "item_id": row['item_id'],
            "content": row.get('content', row.get(text_col, '')),
            "aspect_sentiment": aspect_sent
        })
        vector_records.append({
            "user_id": row['user_id'],
            "item_id": row['item_id'],
            "vector": ' '.join([f"{v:.4f}" for v in vector])
        })

    # 输出 jsonl
    os.makedirs(os.path.dirname(ASPECT_OUTPUT_JSONL), exist_ok=True)
    with open(ASPECT_OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"✅ 已输出 {len(jsonl_records)} 条记录到 {ASPECT_OUTPUT_JSONL}")

    # 输出向量 CSV
    os.makedirs(os.path.dirname(ASPECT_OUTPUT_VECTORS), exist_ok=True)
    pd.DataFrame(vector_records).to_csv(ASPECT_OUTPUT_VECTORS, index=False, encoding='utf-8-sig')
    print(f"✅ 已输出 {len(vector_records)} 条向量到 {ASPECT_OUTPUT_VECTORS}")

    print("\n📊 细粒度情感分析完成")
    print(f"   属性词数量: {len(aspect_vocab)}")
    print(f"   有效评论（至少含一个属性）: {sum(1 for r in jsonl_records if r['aspect_sentiment'])} / {len(df)}")

# ==================== 主入口 ====================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "all"

    if mode == "preprocess":
        run_preprocess()
    elif mode == "aspect":
        run_aspect_sentiment()
    elif mode == "all":
        # run_preprocess()   # 如需同时运行，取消注释
        run_aspect_sentiment()
    else:
        print(f"未知参数: {mode}，可用选项: preprocess, aspect, all")
        sys.exit(1)