import os
import re
import pandas as pd
import jieba
import requests
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ==================== 配置参数 ====================
FILE_PATH = "../data/raw/review.csv"          # 原始数据文件路径
SHEET_NAME = 0                     # Excel工作表名或索引
OUTPUT_CSV_CLEAN = "../data/processed/cleaned_data.csv"      # 输出CSV文件名

OUTPUT_TRAIN = "../data/processed/train.csv"               # 训练集输出文件
OUTPUT_VAL = "../data/processed/val.csv"                   # 验证集输出文件
OUTPUT_TEST = "../data/processed/test.csv"                 # 测试集输出文件

# 文本处理参数
MIN_COMMENT_LEN = 5                      # 最小评论长度（字符数）
ENABLE_DEDUPLICATION = True              # 是否去除重复评论（基于content和sentiment）
MAX_CLEAN_LEN = 2000                     # 清洗后文本最大长度（截断）

# 拆分与标签参数
LABEL_THRESHOLD = 0.5                    # 情感二分类阈值
TEST_SIZE = 0.2                          # 测试集比例（占总数据的比例）
VAL_SIZE = 0.15                          # 验证集比例（占总数据的比例）
RANDOM_STATE = 42                        # 随机种子

# 样本平衡参数
DO_BALANCE = False                        # 是否对训练集进行过采样平衡
MAX_IMBALANCE_RATIO = 1.5                # 正负比例超过该阈值时进行过采样

# 停用词文件（自动下载）
STOPWORDS_FILE = "stopwords.txt"
STOPWORDS_URLS = [
    "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt",
    "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt",
]

# ==================== 辅助函数 ====================
def download_stopwords(filename=STOPWORDS_FILE):
    """如果本地没有停用词文件，尝试从网络下载"""
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
            else:
                print(f"下载失败，HTTP状态码: {resp.status_code}")
        except Exception as e:
            print(f"下载出错: {e}")
            continue
    print("❌ 所有下载链接均失败，请手动下载停用词表并命名为 stopwords.txt 放在当前目录。")
    print("推荐下载地址: https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt")
    return False

def load_stopwords(filepath):
    """加载停用词表，返回set"""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])

def clean_text_only_chinese(text):
    """
    清洗文本：只保留中文汉字、常用标点符号，去除HTML标签、URL、英文、数字、其他特殊符号。
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5。，！？；：、“”‘’、]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

def tokenize_and_remove_stopwords(text, stopwords):
    """jieba分词并去除停用词，返回空格分隔的字符串"""
    if not text:
        return ""
    words = jieba.lcut(text)
    if stopwords:
        words = [w for w in words if w not in stopwords and w.strip()]
    return ' '.join(words)

def generate_report(df_original, df_clean, df_train, df_val, df_test):
    """生成数据报告并打印"""
    print("\n" + "="*60)
    print("                       数据预处理报告")
    print("="*60)

    # 原始数据基本统计
    original_count = len(df_original)
    clean_count = len(df_clean)
    valid_rate = clean_count / original_count * 100 if original_count > 0 else 0

    print(f"\n📊 数据完整性")
    print(f"  原始数据总条数: {original_count}")
    print(f"  清洗后有效条数: {clean_count}")
    print(f"  有效样本率: {valid_rate:.2f}%")

    # 缺失值统计（原始数据）
    missing_content = df_original['review_content'].isna().sum() if 'review_content' in df_original.columns else 0
    missing_sentiment = df_original['sentiments'].isna().sum() if 'sentiments' in df_original.columns else 0
    print(f"  缺失评论数: {missing_content} 条")
    print(f"  缺失感情指标数: {missing_sentiment} 条")

    # 文本长度统计
    content_len = df_clean['content'].astype(str).str.len()
    clean_len = df_clean['clean_text'].str.len()
    print(f"\n📏 文本长度")
    print(f"  原始评论平均长度: {content_len.mean():.1f} 字符")
    print(f"  清洗后文本平均长度: {clean_len.mean():.1f} 字符")
    print(f"  评论长度范围: [{content_len.min()}, {content_len.max()}]")
    print(f"  清洗后文本长度范围: [{clean_len.min()}, {clean_len.max()}]")
    print(f"  清洗后文本中位数长度: {clean_len.median():.1f} 字符")

    # 情感分布
    sentiment = df_clean['sentiment']
    print(f"\n❤️ 情感分布（连续值）")
    print(f"  最小值: {sentiment.min():.4f}")
    print(f"  最大值: {sentiment.max():.4f}")
    print(f"  平均值: {sentiment.mean():.4f}")
    print(f"  中位数: {sentiment.median():.4f}")
    print(f"  标准差: {sentiment.std():.4f}")

    # 二分类标签分布（基于阈值0.5）
    pos = (sentiment > LABEL_THRESHOLD).sum()
    neg = (sentiment < LABEL_THRESHOLD).sum()
    neutral = (sentiment == LABEL_THRESHOLD).sum()
    print(f"\n🏷️ 二分类标签分布 (阈值={LABEL_THRESHOLD})")
    print(f"  正向样本 (>{LABEL_THRESHOLD}): {pos} ({pos/clean_count*100:.2f}%)")
    print(f"  负向样本 (<{LABEL_THRESHOLD}): {neg} ({neg/clean_count*100:.2f}%)")
    if neutral > 0:
        print(f"  中性样本 (=={LABEL_THRESHOLD}): {neutral} ({neutral/clean_count*100:.2f}%)")

    # 拆分后各集统计
    print(f"\n🔀 数据集拆分结果")
    print(f"  训练集样本数: {len(df_train)}")
    print(f"  验证集样本数: {len(df_val)}")
    print(f"  测试集样本数: {len(df_test)}")
    train_pos = (df_train['label'] == 1).sum()
    train_neg = (df_train['label'] == 0).sum()
    val_pos = (df_val['label'] == 1).sum()
    val_neg = (df_val['label'] == 0).sum()
    test_pos = (df_test['label'] == 1).sum()
    test_neg = (df_test['label'] == 0).sum()
    print(f"  训练集正负样本: {train_pos} / {train_neg} (比例 {train_neg/train_pos:.2f})" if train_pos>0 else "  训练集仅有负样本")
    print(f"  验证集正负样本: {val_pos} / {val_neg}")
    print(f"  测试集正负样本: {test_pos} / {test_neg}")

    print("\n✅ 报告生成完毕")
    print("="*60)

# ==================== 主程序 ====================
def main():
    # 0. 准备停用词表
    if not download_stopwords(STOPWORDS_FILE):
        print("警告：没有停用词表，将跳过停用词过滤步骤。")
        stopwords = set()
    else:
        stopwords = load_stopwords(STOPWORDS_FILE)
        print(f"成功加载 {len(stopwords)} 个停用词")

    # 1. 加载原始数据（用于报告统计）
    print("\n1. 加载原始数据...")
    df_raw = pd.read_csv(FILE_PATH)
    print(f"原始数据形状: {df_raw.shape}")

    # 2. 列映射与筛选
    col_mapping = {
        'review_content': 'content',
        'review_rating': 'score',
        'review_time': 'time',
        'review_helpful': 'useful_vote',
        'sentiments': 'sentiment'
    }
    # 只保留存在的列
    exist_cols = [c for c in col_mapping.keys() if c in df_raw.columns]
    df = df_raw[exist_cols].rename(columns=col_mapping)
    print(f"保留列: {list(df.columns)}")

    # 3. 缺失值处理（content和sentiment不能为空）
    init_len = len(df)
    df = df.dropna(subset=['content', 'sentiment'])
    print(f"删除缺失值: {init_len - len(df)} 条")

    # 4. 去重（可选）
    if ENABLE_DEDUPLICATION:
        init_len = len(df)
        df = df.drop_duplicates(subset=['content', 'sentiment'])
        print(f"删除重复评论: {init_len - len(df)} 条")

    # 5. 过滤感情指标范围 [0,1]
    init_len = len(df)
    df = df[(df['sentiment'] >= 0) & (df['sentiment'] <= 1)]
    print(f"感情指标超出范围: {init_len - len(df)} 条")

    # 6. 清洗文本得到 clean_text
    print("2. 对评论文本进行清洗（保留中文和常用标点）...")
    df['clean_text'] = df['content'].apply(clean_text_only_chinese)
    init_len = len(df)
    df = df[df['clean_text'].str.len() >= MIN_COMMENT_LEN]
    print(f"清洗后评论过短(长度<{MIN_COMMENT_LEN}): {init_len - len(df)} 条")
    df['clean_text'] = df['clean_text'].str[:MAX_CLEAN_LEN]

    # 7. 分词并去停用词
    print("3. 对清洗后文本进行分词并去除停用词...")
    df['tokenized'] = df['clean_text'].apply(lambda x: tokenize_and_remove_stopwords(x, stopwords))
    init_len = len(df)
    df = df[df['tokenized'].str.strip() != ""]
    print(f"分词后无有效词语: {init_len - len(df)} 条")

    # 8. 构造二分类标签
    df['label'] = df['sentiment'].apply(lambda x: 1 if x > LABEL_THRESHOLD else (0 if x < LABEL_THRESHOLD else None))
    init_len = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"删除中性样本 (sentiment=={LABEL_THRESHOLD}): {init_len - len(df)} 条")

    # 9. 保存完整清洗数据（可选）
    df.to_csv(OUTPUT_CSV_CLEAN, index=False, encoding='utf-8-sig')
    print(f"完整清洗数据已保存: {OUTPUT_CSV_CLEAN}")

    # 10. 分层拆分训练/验证/测试集
    print("\n4. 进行数据集拆分...")
    X = df['tokenized']   # 使用分词后的文本作为特征
    y = df['label']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VAL_SIZE,
        stratify=y, random_state=RANDOM_STATE
    )
    # 从临时集中分出验证集和测试集
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE+VAL_SIZE),
        stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"拆分结果: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")

    # 11. 训练集样本平衡（过采样）
    if DO_BALANCE:
        train_pos = (y_train == 1).sum()
        train_neg = (y_train == 0).sum()
        if train_pos > 0 and train_neg > 0:
            imbalance = max(train_pos, train_neg) / min(train_pos, train_neg)
            if imbalance > MAX_IMBALANCE_RATIO:
                print(f"训练集类别不平衡 ({train_pos}:{train_neg})，执行过采样...")
                try:
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=RANDOM_STATE)
                    X_train_resampled, y_train_resampled = ros.fit_resample(
                        X_train.values.reshape(-1, 1), y_train
                    )
                    X_train = pd.Series(X_train_resampled[:, 0], name='tokenized')
                    y_train = pd.Series(y_train_resampled, name='label')
                    print(f"过采样后: 正样本 {sum(y_train == 1)} 负样本 {sum(y_train == 0)}")
                    # 过采样后直接构造训练集（仅包含 tokenized 和 label）
                    df_train = pd.DataFrame({'tokenized': X_train, 'label': y_train})
                except ImportError:
                    print("警告：未安装 imbalanced-learn，跳过过采样。")
                    df_train = df.loc[X_train.index].copy()
            else:
                print("训练集比例可接受，不进行平衡处理")
                df_train = df.loc[X_train.index].copy()
        else:
            print("训练集只有一个类别，无法进行平衡")
            df_train = df.loc[X_train.index].copy()
    else:
        df_train = df.loc[X_train.index].copy()


    # 12. 构建最终数据集（保留原始字段和标签）
    # 在 df 中根据 tokenized 内容匹配回原始信息
    # 创建索引映射，因为拆分后tokenized是独立的Series
    df_train = df.loc[X_train.index].copy() if not isinstance(X_train, pd.Series) else df.loc[X_train.index].copy()
    df_val = df.loc[X_val.index].copy()
    df_test = df.loc[X_test.index].copy()
    # 确保标签一致（以防万一）
    df_train['label'] = y_train.values if isinstance(y_train, pd.Series) else y_train
    df_val['label'] = y_val.values if isinstance(y_val, pd.Series) else y_val
    df_test['label'] = y_test.values if isinstance(y_test, pd.Series) else y_test

    # 输出训练/验证/测试集 CSV
    df_train.to_csv(OUTPUT_TRAIN, index=False, encoding='utf-8-sig')
    df_val.to_csv(OUTPUT_VAL, index=False, encoding='utf-8-sig')
    df_test.to_csv(OUTPUT_TEST, index=False, encoding='utf-8-sig')
    print(f"\n数据集已保存:")
    print(f"  训练集: {OUTPUT_TRAIN} ({len(df_train)} 条)")
    print(f"  验证集: {OUTPUT_VAL} ({len(df_val)} 条)")
    print(f"  测试集: {OUTPUT_TEST} ({len(df_test)} 条)")

    # 13. 生成并打印报告
    generate_report(df_raw, df, df_train, df_val, df_test)

if __name__ == "__main__":
    main()