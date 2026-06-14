#!/usr/bin/env python3
"""
سكريبت اختبار محسّن لـ Gemini
التحقق من:
1. تثبيت المكتبات
2. قراءة GEMINI_API_KEY من .env
3. الاتصال بـ Gemini API
4. اختبار التحليل
"""

import os
import sys
import json
import re
from dotenv import load_dotenv

print("=" * 60)
print("🔍 اختبار شامل لـ Gemini Integration")
print("=" * 60)

# Step 1: تحميل المفتاح
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("\n✓ Step 1: التحقق من GEMINI_API_KEY")
if api_key:
    print(f"  ✅ المفتاح موجود: {api_key[:20]}***")
else:
    print("  ❌ المفتاح غير موجود! أضفه في .env")
    sys.exit(1)

# Step 2: التحقق من المكتبة
print("\n✓ Step 2: التحقق من google-generativeai")
try:
    import google.generativeai as genai
    print("  ✅ المكتبة مثبتة بنجاح")
except ImportError:
    print("  ❌ المكتبة غير مثبتة!")
    print("  الحل: pip install google-generativeai")
    sys.exit(1)

# Step 3: تكوين Gemini
print("\n✓ Step 3: تكوين Gemini API")
try:
    genai.configure(api_key=api_key)
    print("  ✅ تم تكوين API بنجاح")
except Exception as e:
    print(f"  ❌ خطأ في التكوين: {e}")
    sys.exit(1)

# Step 4: اختبار النموذج
print("\n✓ Step 4: اختبار النموذج المتاح من Gemini")
try:
    available_models = [
        m.name.replace('models/', '')
        for m in genai.list_models()
        if 'generateContent' in getattr(m, 'supported_generation_methods', [])
    ]
    if not available_models:
        raise RuntimeError('لا توجد موديلات generateContent متاحة')
    model_name = available_models[0]
    print(f"  ✅ النموذج المستخدم: {model_name}")
    model = genai.GenerativeModel(model_name)
    print("  ✅ تم إنشاء النموذج بنجاح")
except Exception as e:
    print(f"  ❌ خطأ في إنشاء النموذج: {e}")
    sys.exit(1)

# Step 5: اختبار التحليل
print("\n✓ Step 5: اختبار التحليل باستخدام تعليقات تجريبية")

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

أعد الرد بصيغة JSON فقط بدون أي نص إضافي:
{{"strengths":[],"weaknesses":[],"recommendations":[],"insights":{{"main_issue":"","future_prediction":""}}}}
"""

try:
    print("  📤 جاري الإرسال للـ API...")
    response = model.generate_content(prompt)
    print("  ✅ تلقينا رد من Gemini")
    
    print("\n--- الرد الخام من Gemini ---")
    print(response.text)
    print("--- نهاية الرد الخام ---\n")
    
    # محاولة استخراج JSON
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if match:
        json_str = match.group(0)
        result = json.loads(json_str)
        print("✅ تم استخراج JSON بنجاح!")
        print("\n--- النتيجة المحللة ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("⚠️ لم يتمكن من استخراج JSON من الرد")
        
except Exception as e:
    print(f"  ❌ خطأ في الاستدعاء: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ جميع الاختبارات نجحت! النظام جاهز للعمل")
print("=" * 60)
print("\n💡 الخطوات التالية:")
print("1. تثبيت المكتبات: pip install -r requirements.txt")
print("2. تشغيل السيرفر: uvicorn main:app --reload --port 8001")
print("3. اختبار الـ API من الواجهة الأمامية")
