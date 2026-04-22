from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
import os
from utils import DIRS, setup_logging

logger = setup_logging()

app = FastAPI(
    title="电商评论情感分析 API",
    description="提供基于 BERT 的基础情感打分和基于 LLM 的细粒度属性级情感分析(ABSA)服务",
    version="1.0.0"
)

# --- 1. BERT 模型加载 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_path = DIRS['models']
tokenizer = None
model = None

try:
    if model_path.exists() and (model_path / "config.json").exists():
        logger.info(f"正在加载 BERT 模型 ({device})...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        logger.info("BERT 模型加载成功！")
    else:
        logger.warning("未找到预训练的 BERT 模型，/predict/bert 接口可能不可用。请先运行 src/3_train_bert.py")
except Exception as e:
    logger.error(f"加载 BERT 模型失败: {e}")


# --- 2. 接口数据结构定义 ---
class CommentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class ABSAResponse(BaseModel):
    text: str
    analysis: str


# --- 3. BERT 基础情感分析接口 ---
@app.post("/predict/bert", response_model=SentimentResponse, tags=["模型预测"])
async def predict_bert(request: CommentRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="BERT 模型未加载。")
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="评论内容不能为空。")

    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            # 假设 1 是正向，0 是负向
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            
            sentiment_label = "正向" if pred_class == 1 else "负向"
            
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment_label,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"BERT 预测出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 4. 大语言模型 ABSA 接口 ---
@app.post("/predict/absa", response_model=ABSAResponse, tags=["大模型分析"])
async def predict_absa(request: CommentRequest):
    """
    使用 LLM 进行细粒度属性级情感分析 (ABSA)
    你需要设置环境变量 OPENAI_API_KEY (或者兼容 OpenAI 接口的其他模型 API KEY)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # 为了方便展示，如果没有提供 key，返回模拟结果
        return ABSAResponse(
            text=request.text,
            analysis="[未配置 API_KEY, 此为模拟结果]\n- 物流: 负向 (速度太慢)\n- 质量: 正向 (做工不错)\n- 服务: 中性"
        )
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="评论内容不能为空。")

    # 配置客户端 (支持更换 Base URL 来使用国内大模型如 DeepSeek/Qwen 等)
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = f"""
你是一个专业的电商舆情分析专家。请对以下电商评论进行“细粒度属性级情感分析”(ABSA)。
提取出评论中提到的具体商品属性（如：外观、质量、物流、价格、客服等），并判断每个属性的情感倾向（正向、负向、中性），并给出简短理由。

评论内容："{request.text}"

请按以下格式输出分析结果：
- [属性1]: [情感倾向] ([理由])
- [属性2]: [情感倾向] ([理由])
...
"""
    
    try:
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=256
        )
        
        result = response.choices[0].message.content
        return ABSAResponse(
            text=request.text,
            analysis=result
        )
    except Exception as e:
        logger.error(f"LLM API 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"LLM 分析失败: {str(e)}")


@app.get("/", tags=["系统"])
async def root():
    return {"message": "欢迎使用电商评论情感分析 API 服务！访问 /docs 查看接口文档。"}

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 FastAPI 服务，端口 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)