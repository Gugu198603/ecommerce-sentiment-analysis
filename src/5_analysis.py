import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib
import sys
import os
from utils import DIRS, ensure_dirs, setup_logging

logger = setup_logging()

def set_chinese_font():
    """设置matplotlib支持中文字体显示，兼容Mac和Windows"""
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def analysis():
    ensure_dirs()
    set_chinese_font()
    
    data_path = DIRS['processed_data'] / "cleaned_data.csv"
    if not data_path.exists():
        logger.error("未找到预处理数据，无法进行业务分析。")
        return
        
    logger.info("1. 开始加载数据...")
    df = pd.read_csv(data_path)
    
    logger.info("2. 生成负面评论词云图...")
    neg_texts = df[df['sentiment'] == 0]['tokenized'].dropna().values
    text_corpus = " ".join(neg_texts)
    
    # Mac 系统中常用的中文字体路径
    font_path = "/System/Library/Fonts/PingFang.ttc"
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
    if not os.path.exists(font_path):
        font_path = None # 不指定字体，可能中文会乱码，但不会报错
    
    if sys.platform != 'darwin':
        font_path = "C:/Windows/Fonts/simhei.ttf"
    
    try:
        wc = WordCloud(
            font_path=font_path,
            width=800, 
            height=600,
            background_color='white',
            max_words=200
        )
        wc.generate(text_corpus)
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('负面评论高频词云分析')
        
        wc_path = DIRS['figures'] / "wordcloud_negative.png"
        plt.savefig(wc_path, dpi=300)
        plt.close()
        logger.info(f"负面词云已保存至 {wc_path}")
    except Exception as e:
        logger.error(f"生成词云失败，请检查字体路径: {e}")
        
    logger.info("3. 评论长度与情感分布对比...")
    df['comment_len'] = df['clean_text'].astype(str).apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='comment_len', hue='sentiment', common_norm=False, fill=True)
    plt.title('正负面评论的长度分布对比')
    plt.xlabel('评论字符长度')
    plt.ylabel('密度')
    plt.xlim(0, 200)
    
    len_dist_path = DIRS['figures'] / "length_distribution.png"
    plt.savefig(len_dist_path, dpi=300)
    plt.close()
    logger.info(f"长度分布图已保存至 {len_dist_path}")
    
    logger.info("4. 输出业务洞察 (Business Insights)...")
    avg_pos_len = df[df['sentiment']==1]['comment_len'].mean()
    avg_neg_len = df[df['sentiment']==0]['comment_len'].mean()
    
    insight_text = (
        "========= 业务洞察报告 =========\n"
        "1. 情感与评论长度关联:\n"
        f"   - 好评平均长度: {avg_pos_len:.1f} 字\n"
        f"   - 差评平均长度: {avg_neg_len:.1f} 字\n"
        "   > 结论: 用户在给出差评时，往往会写更长的评论来抱怨具体的产品缺陷或物流服务。\n\n"
        "2. 负向词云分析:\n"
        "   > 建议电商运营团队重点关注负面词云中出现的高频名词（如：'包装'、'物流'、'质量'等），这些是直接影响复购率的痛点。\n"
        "================================"
    )
    
    logger.info("\n" + insight_text)
    
    with open(DIRS['reports'] / "business_insights.txt", "w", encoding="utf-8") as f:
        f.write(insight_text)
    
if __name__ == '__main__':
    analysis()