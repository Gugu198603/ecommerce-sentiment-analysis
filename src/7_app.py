import streamlit as st
import requests
import json
import pandas as pd
from utils import DIRS

# 页面配置
st.set_page_config(
    page_title="电商评论情感分析系统",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 修复了 API 地址，使用正确的 localhost
API_URL = "http://127.0.0.1:8000"

def get_bert_sentiment(text):
    try:
        response = requests.post(f"{API_URL}/predict/bert", json={"text": text}, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API 错误: {response.text}"}
    except Exception as e:
        return {"error": f"连接 FastAPI 服务失败: {e}。请确保后台服务已启动！"}

def get_absa_analysis(text, api_key=None, base_url=None, model=None):
    # 这里我们通过 Streamlit 界面接收用户的 API Key，并通过环境变量传给 FastAPI
    # 或者为了安全，真实生产中应在 FastAPI 侧写死或从 Vault 读取
    # 这里作为演示，我们修改环境变量让 FastAPI 读到 (不推荐生产环境这样做，但课程设计够用)
    import os
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    if base_url:
        os.environ['OPENAI_API_BASE'] = base_url
    if model:
        os.environ['LLM_MODEL'] = model
        
    try:
        response = requests.post(f"{API_URL}/predict/absa", json={"text": text}, timeout=30)
        if response.status_code == 200:
            return response.json()
        return {"error": f"LLM API 错误: {response.text}"}
    except Exception as e:
        return {"error": f"连接 FastAPI 服务失败: {e}。"}

# --- 侧边栏 ---
st.sidebar.title("⚙️ 系统设置")
st.sidebar.markdown("---")

st.sidebar.subheader("大语言模型配置 (可选)")
st.sidebar.info("配置 API Key 即可解锁细粒度属性级情感分析 (ABSA) 功能。支持兼容 OpenAI 接口的大模型。")

api_key = st.sidebar.text_input("API Key", type="password", placeholder="sk-...")
base_url = st.sidebar.text_input("Base URL", value="https://api.openai.com/v1", help="如果你使用国内模型（如 DeepSeek/智谱），请修改此地址")
model_name = st.sidebar.text_input("Model Name", value="gpt-3.5-turbo")

st.sidebar.markdown("---")
st.sidebar.markdown("### 项目信息")
st.sidebar.markdown("- 核心技术: BERT, Word2Vec, LLM")
st.sidebar.markdown("- 任务: 电商评论正负向分类 & ABSA")
st.sidebar.markdown("- 框架: PyTorch, FastAPI, Streamlit")


# --- 主界面 ---
st.title("🛍️ 电商评论情感智能分析系统")
st.markdown("通过深度学习与大语言模型，自动化挖掘用户评论背后的商业价值。")

tab1, tab2 = st.tabs(["💬 实时评论分析", "📊 历史业务报告"])

with tab1:
    st.subheader("输入单条用户评论")
    user_input = st.text_area("评论内容:", height=150, placeholder="请粘贴买家的真实评价，例如：手机手感很好，但是物流太慢了，客服态度也不好。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 基础情感打分 (BERT)", use_container_width=True):
            if not user_input.strip():
                st.warning("请输入评论内容！")
            else:
                with st.spinner("BERT 模型推理中..."):
                    result = get_bert_sentiment(user_input)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        sentiment = result.get("sentiment", "未知")
                        conf = result.get("confidence", 0.0)
                        
                        st.markdown("### BERT 预测结果")
                        
                        if sentiment == "正向":
                            st.success(f"**情感倾向**: {sentiment} 😃")
                        else:
                            st.error(f"**情感倾向**: {sentiment} 😡")
                            
                        st.progress(conf, text=f"置信度: {conf:.2%}")
                        
                        st.info("💡 **商业建议**: 基础情感打分能快速判断整体基调，适用于大盘数据的宏观舆情监控。")

    with col2:
        if st.button("🧠 细粒度属性分析 (LLM ABSA)", use_container_width=True):
            if not user_input.strip():
                st.warning("请输入评论内容！")
            else:
                with st.spinner("大模型深入分析中..."):
                    result = get_absa_analysis(user_input, api_key, base_url, model_name)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        analysis = result.get("analysis", "")
                        
                        st.markdown("### LLM 洞察报告")
                        st.markdown(f"```text\n{analysis}\n```")
                        
                        if not api_key:
                            st.warning("⚠️ 当前未配置 API Key，显示为模拟结果。请在左侧边栏配置以获取真实分析。")
                            
                        st.info("💡 **商业建议**: ABSA 能够精准定位用户抱怨或表扬的具体维度（如：物流慢、质量好），是指导产品迭代的利器。")

with tab2:
    st.subheader("业务数据可视化大盘")
    st.markdown("以下数据图表由历史评论数据离线生成（运行 `src/5_analysis.py`）。")
    
    fig_dir = DIRS['figures']
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 😡 负面情绪高频痛点词云")
        wc_path = fig_dir / "wordcloud_negative.png"
        if wc_path.exists():
            st.image(str(wc_path), use_container_width=True)
        else:
            st.info("暂未生成词云图，请先运行数据分析脚本。")
            
    with col4:
        st.markdown("#### 📏 评论长度与情感关联分布")
        dist_path = fig_dir / "length_distribution.png"
        if dist_path.exists():
            st.image(str(dist_path), use_container_width=True)
        else:
            st.info("暂未生成分布图，请先运行数据分析脚本。")
            
    st.markdown("---")
    st.markdown("#### 📈 深度业务洞察结论")
    report_path = DIRS['reports'] / "business_insights.txt"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f"```text\n{f.read()}\n```")
    else:
        st.info("暂无洞察报告。")