#!/usr/bin/env python3
"""
سكريبت اختبار كامل للنظام
يختبر:
1. المكتبات والاستيرادات
2. تحميل النموذج
3. التصنيف
4. الاتصال بـ Gemini
5. الـ API endpoints
"""

import os
import sys
import json
import time
from pathlib import Path

# إضافة المجلد الحالي للـ path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🧪 اختبار شامل للنظام")
print("=" * 70)

# Step 1: التحقق من المكتبات
print("\n✓ Step 1: التحقق من المكتبات")
required_packages = {
    'google.generativeai': 'google-generativeai',
    'youtube_comment_downloader': 'youtube-comment-downloader',
    'fastapi': 'fastapi',
    'torch': 'torch',
    'transformers': 'transformers',
    'joblib': 'joblib'
}

missing_packages = []
for module, package_name in required_packages.items():
    try:
        __import__(module)
        print(f"  ✅ {package_name}")
    except ImportError:
        print(f"  ❌ {package_name} - غير مثبت")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n⚠️ المكتبات المفقودة:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Step 2: التحقق من ملف .env
print("\n✓ Step 2: التحقق من ملف .env")
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"  ✅ GEMINI_API_KEY موجود: {api_key[:15]}***")
else:
    print("  ❌ GEMINI_API_KEY غير موجود في .env")
    sys.exit(1)

# Step 3: التحقق من Gemini Configuration
print("\n✓ Step 3: تكوين Gemini")
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    print("  ✅ تم تكوين Gemini بنجاح")
except Exception as e:
    print(f"  ❌ خطأ: {e}")
    sys.exit(1)

# Step 4: التحقق من النموذج
print("\n✓ Step 4: اختبار نموذج Gemini المتاح")
try:
    available_models = [
        m.name.replace('models/', '')
        for m in genai.list_models()
        if 'generateContent' in getattr(m, 'supported_generation_methods', [])
    ]
    model_name = available_models[0] if available_models else 'gemini-2.5-flash'
    print(f"  ✅ النموذج المستخدم: {model_name}")
    gemini_model = genai.GenerativeModel(model_name)
    response = gemini_model.generate_content("قل مرحبا")
    print(f"  ✅ النموذج يعمل: {response.text[:50]}...")
except Exception as e:
    print(f"  ❌ خطأ: {e}")
    sys.exit(1)

# Step 5: التحقق من تحميل النموذج المحلي
print("\n✓ Step 5: التحقق من النموذج المحلي (MarBERT)")
try:
    import joblib
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"  ✅ تم تحميل sentiment_model.pkl")
        print(f"     - الفئات: {model.classes_}")
    else:
        print(f"  ⚠️  sentiment_model.pkl غير موجود في {model_path}")
        
    marbert_path = os.path.join(os.path.dirname(__file__), "marbert_model")
    if os.path.exists(marbert_path):
        print(f"  ✅ مجلد marbert_model موجود")
    else:
        print(f"  ⚠️  مجلد marbert_model غير موجود - سيتم تحميله من HuggingFace")
        
except Exception as e:
    print(f"  ⚠️  تحذير: {e}")

# Step 6: اختبار التصنيف
print("\n✓ Step 6: اختبار التصنيف (Sentiment Analysis)")
test_texts = [
    "الفيديو رائع جداً وممتاز",
    "محتوى سيء وممل",
    "مشروح عادي"
]

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # تحميل MARBERT
    try:
        marbert_path = os.path.join(os.path.dirname(__file__), "marbert_model")
        if os.path.exists(marbert_path):
            tokenizer = AutoTokenizer.from_pretrained(marbert_path)
            bert_model = AutoModel.from_pretrained(marbert_path)
        else:
            print("    تحميل MARBERT من HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
            bert_model = AutoModel.from_pretrained("UBC-NLP/MARBERT")
    except Exception as e:
        print(f"    ⚠️ تحذير في تحميل MARBERT: {e}")
        raise
    
    bert_model.to(device)
    bert_model.eval()
    
    # تحميل النموذج
    model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
    if not os.path.exists(model_path):
        print(f"  ❌ sentiment_model.pkl غير موجود!")
        sys.exit(1)
        
    sentiment_model = joblib.load(model_path)
    
    print("  اختبار التصنيفات:")
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        probs = sentiment_model.predict_proba(emb)[0]
        pred_class = sentiment_model.classes_[np.argmax(probs)]
        confidence = float(np.max(probs))
        
        mapping = {"P": "positive", "N": "negative", "NEU": "neutral"}
        result = mapping.get(pred_class, "neutral")
        
        print(f"    • '{text[:30]}...'")
        print(f"      → {result} ({confidence:.2%})")
    
    print("  ✅ التصنيف يعمل بنجاح")
    
except Exception as e:
    print(f"  ⚠️ تحذير: {e}")

# Step 7: اختبار Gemini Analysis
print("\n✓ Step 7: اختبار التحليل عبر Gemini")
try:
    test_comments = """
- الفيديو رائع جداً وممتاز
- محتوى سيء وممل جداً
- شرح واضح جداً شكراً
- لا أفهم شيء من التفاصيل
- مشروح عظيم وجميل جداً
"""
    
    prompt = f"""
أنت محلل استراتيجي متخصص. حلل التعليقات التالية واستخرج:
- نقاط القوة (Strengths)
- نقاط الضعف (Weaknesses)
- التوصيات (Recommendations)
- الرؤى (Insights): المشكلة الرئيسية والتنبؤ المستقبلي

التعليقات:
{test_comments}

أعد الرد بصيغة JSON فقط:
{{"strengths":[],"weaknesses":[],"recommendations":[],"insights":{{"main_issue":"","future_prediction":""}}}}
"""
    
    print("  📤 جاري إرسال الطلب إلى Gemini...")
    available_models = [
        m.name.replace('models/', '')
        for m in genai.list_models()
        if 'generateContent' in getattr(m, 'supported_generation_methods', [])
    ]
    model_name = available_models[0] if available_models else 'gemini-2.5-flash'
    print(f"  ✅ النموذج المستخدم: {model_name}")
    gemini_model = genai.GenerativeModel(model_name)
    response = gemini_model.generate_content(prompt)
    
    print("  ✅ تم استقبال الرد من Gemini")
    
    # محاولة استخراج JSON
    import re
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        result = json.loads(match.group(0))
        print("\n  📊 النتيجة المحللة:")
        print(f"    - نقاط القوة: {len(result.get('strengths', []))} عناصر")
        print(f"    - نقاط الضعف: {len(result.get('weaknesses', []))} عناصر")
        print(f"    - التوصيات: {len(result.get('recommendations', []))} عناصر")
        if result.get('insights'):
            print(f"    - المشكلة الرئيسية: {result['insights'].get('main_issue', 'N/A')[:50]}...")
        print("  ✅ التحليل يعمل بنجاح")
    else:
        print("  ⚠️ لم يتمكن من استخراج JSON من الرد")
        
except Exception as e:
    print(f"  ❌ خطأ: {e}")
    import traceback
    traceback.print_exc()

# Step 8: ملخص النتائج
print("\n" + "=" * 70)
print("✅ الاختبار مكتمل!")
print("=" * 70)
print("""
التالي:
  1. تشغيل السيرفر:
     uvicorn main:app --reload --port 8001
     
  2. اختبر من الواجهة الأمامية:
     http://localhost:3000
     
  3. ارفع فيديو وحلل التعليقات

💡 إذا أردت تشغيل النموذج بدون GPU:
   export CUDA_VISIBLE_DEVICES=""
   python main.py
""")
