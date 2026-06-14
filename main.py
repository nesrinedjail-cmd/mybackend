# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Gemini SDK
import google.generativeai as genai


from youtube_comment_downloader import (
    YoutubeCommentDownloader,
    SORT_BY_RECENT
)

import re
import joblib
import numpy as np
import torch
import os
import json
from datetime import datetime, timedelta, timezone
import time

from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModel
)

# =========================================================
# LOAD ENV
# =========================================================

DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

# FASTAPI

app = FastAPI()
print("MAIN FILE LOADED")
# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

# =========================================================
# =========================================================
# GEMINI
# =========================================================
# =========================================================

# GEMINI

# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY غير موجودة داخل .env")

genai.configure(api_key=GEMINI_API_KEY)

print("✅ Gemini Connected")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI Fallback Enabled")
else:
    openai_client = None

# =========================================================
# AI CALL
# =========================================================



def get_gemini_model_candidates() -> list[str]:
    candidates: list[str] = []

    try:
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(model, "name", "").replace("models/", "")
                if name and name not in candidates:
                    candidates.append(name)
    except Exception as e:
        print("❌ Error reading Gemini models:", e)

    for fallback in (
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-pro-latest",
    ):
        if fallback not in candidates:
            candidates.append(fallback)

    return candidates


def call_ai(prompt: str) -> str | None:
    # Gemini model name availability depends on your Gemini API/account configuration.
    # Try the models that are actually available for this API key.
    model_candidates = get_gemini_model_candidates()
    print("Using Gemini models:", model_candidates)

    last_error: Exception | None = None

    for model_name in model_candidates:
        try:
           
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.8,
                    "max_output_tokens": 2000,
                },
            )
            return (resp.text or "").strip() if resp else None
        except Exception as e:
            last_error = e
            # Keep trying other models
            print(f"Gemini Error ({model_name}):", e)

    # Fallback to OpenAI if configured
    if openai_client is None:
        return None

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "أنت محلل بيانات احترافي متخصص في تحليل تعليقات يوتيوب. أعد JSON فقط بدون أي نص إضافي."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e2:
        print("OpenAI Fallback Error:", e2)
        if last_error:
            print("Last Gemini Error:", last_error)
        return None




def generate_fallback_analysis(comments: list[str]) -> dict:
    positive = 0
    negative = 0
    neutral = 0
    for text in comments:
        sentiment, _ = predict_sentiment(text)
        if sentiment == "positive":
            positive += 1
        elif sentiment == "negative":
            negative += 1
        else:
            neutral += 1

    total = len(comments) or 1
    summary = (
        f"التعليقات تُظهر %s إيجابية، %s سلبية، و %s محايدة." % (
            positive, negative, neutral
        )
    )

    strengths = [
        "التفاعل جيد مع المحتوى الإيجابي." if positive >= negative else ""
    ]
    weaknesses = [
        "بعض التعليقات سلبية، يحتاج المحتوى إلى تحسين." if negative > positive else ""
    ]
    recommendations = [
        "ركز على تحسين جودة المحتوى وتقليل نقاط الجدل." if negative > 0 else "استمر بنفس النهج الحالي."
    ]

    return {
        "strengths": [s for s in strengths if s],
        "weaknesses": [w for w in weaknesses if w],
        "recommendations": recommendations,
        "summary": summary,
        "main_opportunity": "تحسين التفاعل من خلال الرد على التعليقات الإيجابية وبناء مزيد من الثقة.",
        "future_prediction": "من المتوقع أن يزداد التفاعل عند تقليل المحتوى المثير للجدل وزيادة المحتوى الملهم."
    }

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"✅ Device: {device}")

# =========================================================
# LOAD CLASSIFIER
# =========================================================

model = joblib.load(MODEL_PATH)

print("✅ Sentiment model loaded")

# =========================================================
# LOAD MARBERT
# =========================================================

try:

    tokenizer = AutoTokenizer.from_pretrained(
        "UBC-NLP/MARBERT"
    )

    bert_model = AutoModel.from_pretrained(
        "UBC-NLP/MARBERT"
    )

    bert_model.to(device)

    print("✅ MARBERT loaded")

except Exception as e:

    print(f"❌ Error loading MARBERT: {e}")

    raise e

# =========================================================
# RELIGIOUS WORDS
# =========================================================

RELIGIOUS_KEYWORDS = [

    'اللهم',
    'سبحان الله',
    'الحمد لله',
    'الله أكبر',
    'لا إله إلا الله',
    'استغفر الله',
    'ما شاء الله',
    'تبارك الله',
    'جزاك الله خير',
    'اللهم صلي',
    'اللهم صل',
    'صلى الله عليه',
    'عليه الصلاة والسلام',
    'اللهم بارك',
    'اللهم اغفر',
    'ربنا',
    'رسول الله',
    'النبي محمد',
    'آمين',
    'يا رب',
    'الدعاء',
    'الاستغفار'

]

# =========================================================
# QUESTION WORDS
# =========================================================

QUESTION_KEYWORDS = [

    "هل",
    "كم",
    "كيف",
    "متى",
    "أين",
    "لماذا",
    "ما",
    "ماذا",

    "شحال",
    "بشحال",
    "قداه",
    "واش",
    "وين",
    "علاه",
    "كيفاش",

    "السعر",
    "الثمن",
    "يتباع",
    "يباع",
    "بكم",
    "price"

]

# =========================================================
# RELIGIOUS CHECK
# =========================================================

def is_religious_comment(text: str) -> bool:

    if not text:
        return False

    text = text.lower()

    for keyword in RELIGIOUS_KEYWORDS:

        if keyword in text:
            return True

    return False

# =========================================================
# QUESTION CHECK
# =========================================================

def is_question_comment(text: str) -> bool:

    if not text:
        return False

    text = text.strip().lower()

    if text.endswith("?") or text.endswith("؟"):
        return True

    words = text.split()

    first_word = words[0] if words else ""

    if first_word in QUESTION_KEYWORDS:
        return True

    return False

# =========================================================
# GET EMBEDDING
# =========================================================

def get_embedding(text: str):

    try:

        if not text or text.strip() == "":
            return np.zeros((1, 768))

        inputs = tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=512

        )

        inputs = {

            key: value.to(device)

            for key, value in inputs.items()

        }

        with torch.no_grad():

            outputs = bert_model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]

        embedding = embedding.cpu().numpy()

        return embedding

    except Exception as e:

        print(f"❌ خطأ get_embedding: {e}")

        return np.zeros((1, 768))

# =========================================================
# PREDICT SENTIMENT
# =========================================================

def predict_sentiment(text: str, threshold=0.6):

    try:

        if not text or text.strip() == "":
            return "neutral", 0.0

        if is_religious_comment(text):
            return "neutral", 1.0

        if is_question_comment(text):
            return "neutral", 1.0

        emb = get_embedding(text)

        probs = model.predict_proba(emb)[0]

        max_prob = float(np.max(probs))

        pred_class = model.classes_[np.argmax(probs)]

        mapping = {

            "P": "positive",
            "N": "negative",
            "NEU": "neutral"

        }

        if max_prob < threshold:
            return "neutral", max_prob

        return mapping.get(
            pred_class,
            "neutral"
        ), max_prob

    except Exception as e:

        print(f"❌ خطأ predict_sentiment: {e}")

        return "neutral", 0.0

# =========================================================
# EXTRACT VIDEO ID
# =========================================================

def extract_video_id(url: str):

    patterns = [

        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})'

    ]

    for p in patterns:

        match = re.search(p, url)

        if match:
            return match.group(1)

    raise ValueError("رابط يوتيوب غير صحيح")

# =========================================================
# EXTRACT COMMENTS API
# =========================================================

@app.post("/api/extract")

def extract_comments(request: dict):

    try:

        url = request.get("url")

        if not url:

            raise HTTPException(
                status_code=400,
                detail="الرابط مطلوب"
            )

        max_results = int(
            request.get("max_results", 100)
        )

        video_id = extract_video_id(url)

        downloader = YoutubeCommentDownloader()

        generator = downloader.get_comments(

            youtube_id=video_id,

            sort_by=SORT_BY_RECENT

        )

        comments = []

        religious_count = 0
        question_count = 0

        for i, comment in enumerate(generator):

            if i >= max_results:
                break

            text = comment.get("text", "")

            if is_religious_comment(text):
                religious_count += 1

            if is_question_comment(text):
                question_count += 1

            sentiment, score = predict_sentiment(text)

            comments.append({

                "id": comment.get("cid", str(i)),

                "author": comment.get(
                    "author",
                    "Unknown"
                ),

                "text": text,

                "likes": comment.get(
                    "votes",
                    0
                ),

                "sentiment": sentiment,

                "confidence": round(score, 4)

            })

        stats = {

            "positive": len([
                c for c in comments
                if c["sentiment"] == "positive"
            ]),

            "negative": len([
                c for c in comments
                if c["sentiment"] == "negative"
            ]),

            "neutral": len([
                c for c in comments
                if c["sentiment"] == "neutral"
            ]),

            "total": len(comments),

            "religious_comments": religious_count,

            "question_comments": question_count

        }

        return {

            "comments": comments,

            "stats": stats,

            "total_extracted": len(comments),

            "videoTitle": f"تحليل فيديو {video_id}",

            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",

            "videoUrl": url

        }

    except Exception as e:

        print(f"❌ خطأ extract_comments: {e}")

        return {
            "comments": [],
            "stats": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "total": 0,
                "religious_comments": 0,
                "question_comments": 0,
            },
            "total_extracted": 0,
            "videoTitle": "تحليل فيديو غير متاح حالياً",
            "thumbnail": "",
            "videoUrl": request.get("url", ""),
            "error": "تعذر الوصول إلى YouTube الآن. يرجى المحاولة مرة أخرى لاحقاً." if "Read timed out" in str(e) or "timed out" in str(e).lower() else str(e),
            "fallback": True,
        }


# =========================================================
# AI REPORT ANALYSIS
# =========================================================

@app.post("/analyze")
def analyze(request: dict):
    try:
        comments = request.get("comments", []) or []
        print("COMMENTS RECEIVED:", len(comments))
        normalized_comments = [
            str(item).strip()
            for item in comments
            if isinstance(item, str) and str(item).strip()
        ]

        if not normalized_comments:
            raise ValueError("لا توجد تعليقات كافية للتحليل")

        comments_text = "\n".join(f"- {text}" for text in normalized_comments[:40])

        prompt = f"""
حلل التعليقات التالية وأعد تقريراً احترافياً بصيغة JSON فقط.

التعليمات:
- استخدم العربية فقط
- لا تكتب أي شيء خارج JSON
- كن دقيقاً وواقعياً
- استخرج نقاط القوة والضعف والتوصيات
- أضف ملخصاً واضحاً وتنبؤاً مستقبلياً
- تجاهل الأدعية والأسئلة

التعليقات:
{comments_text}

أعد النتيجة بهذا الشكل فقط:
{{
  "strengths": [],
  "weaknesses": [],
  "recommendations": [],
  "summary": "",
  "main_opportunity": "",
  "future_prediction": ""
}}
"""

        raw_text = call_ai(prompt)
        print("RAW GEMINI:", raw_text)

        if not raw_text:
            raise ValueError("لم يعُد Gemini أي نتيجة")

        match = re.search(r'\{[\s\S]*\}', raw_text, re.DOTALL)
        if not match:
            return generate_fallback_analysis(normalized_comments)

        try:
            result = json.loads(match.group(0))
            print("FINAL RESPONSE:", result)
        except json.JSONDecodeError:
            return generate_fallback_analysis(normalized_comments)

        has_meaningful_content = bool(
            (result.get("summary") or "").strip()
            or (result.get("main_opportunity") or "").strip()
            or (result.get("future_prediction") or "").strip()
            or (result.get("strengths") or [])
            or (result.get("weaknesses") or [])
            or (result.get("recommendations") or [])
        )

        if not has_meaningful_content:
            print("⚠️ Empty Gemini result detected, using fallback analysis")
            return generate_fallback_analysis(normalized_comments)

        return {
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
            "recommendations": result.get("recommendations", []),
            "summary": result.get("summary", ""),
            "main_opportunity": result.get("main_opportunity", ""),
            "future_prediction": result.get("future_prediction", ""),
            "insights": {
                "main_issue": result.get("main_opportunity", ""),
                "future_prediction": result.get("future_prediction", ""),
            },
        }

    except Exception as e:
        print(f"❌ خطأ analyze: {e}")
        return generate_fallback_analysis(normalized_comments if 'normalized_comments' in locals() else [])

# =========================================================
# OPENAI ANALYSIS
# =========================================================
# =========================================================
# RUN SERVER
# ل

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8001,


    )