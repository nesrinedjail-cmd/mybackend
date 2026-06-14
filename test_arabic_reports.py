#!/usr/bin/env python3
"""
اختبار التقارير بالعربية
التأكد من أن Gemini يرسل النتائج بالعربية فقط وبصيغة مفصلة
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ لم يتم العثور على GEMINI_API_KEY")
    sys.exit(1)

genai.configure(api_key=api_key)

print("=" * 70)
print("🧪 اختبار التقارير بالعربية")
print("=" * 70)

# تعليقات تجريبية
test_comments = """
- الفيديو رائع جداً والشرح واضح جداً
- المحتوى ممل والسرعة بطيئة
- شكراً على الشرح المفصل والمفيد جداً
- لا أفهم بعض الأجزاء الصعبة
- فيديو عظيم ومشروح احترافي جداً
- الصوت غير واضح وهناك مشاكل تقنية
- محتوى قيّم جداً وسأطبقه فوراً
- لا توجد أمثلة عملية كافية
- شكراً لك يا أستاذ الشرح رائع جداً
- المادة قديمة وغير حديثة
"""

prompt = f"""أنت محلل استراتيجي متخصص في تحليل تعليقات الفيديوهات والمحتوى.

المهمة: حلل التعليقات التالية وأعد تقرير استراتيجي شامل بصيغة JSON.

⭐ متطلبات التحليل:
1. اكتب كل شيء باللغة العربية الفصحى فقط - لا تستخدم أي لغة أخرى
2. كن محدداً وواضحاً في كل نقطة
3. اذكر أمثلة من التعليقات عند الحاجة
4. ركز على الاستراتيجية والقيمة

📊 التعليقات المراد تحليلها:
{test_comments}

🎯 المخرجات المطلوبة (بصيغة JSON):
{{
  "strengths": [
    "نقطة قوة محددة وواضحة",
    "نقطة قوة أخرى"
  ],
  "weaknesses": [
    "نقطة ضعف محددة",
    "نقطة ضعف أخرى"
  ],
  "recommendations": [
    "توصية محددة وقابلة للتنفيذ",
    "توصية أخرى"
  ],
  "summary": "ملخص شامل للتقرير في 2-3 جمل",
  "main_opportunity": "أهم فرصة للتحسن",
  "future_prediction": "توقعات مستقبلية بناءً على الاتجاهات"
}}

⚠️ تعليمات مهمة:
- أجب بـ JSON فقط - بدون أي نص إضافي قبل أو بعد
- كل النصوص باللغة العربية فقط
- تأكد أن كل نقطة واضحة ومحددة وقيّمة
- اجعل الأمثلة محددة من التعليقات الفعلية
"""

print("\n🔄 جاري إرسال الطلب إلى Gemini...")
print("-" * 70)

try:
    available_models = [
        m.name.replace('models/', '')
        for m in genai.list_models()
        if 'generateContent' in getattr(m, 'supported_generation_methods', [])
    ]
    model_name = available_models[0] if available_models else 'gemini-2.5-flash'
    model = genai.GenerativeModel(model_name)
    print(f"  ✅ النموذج المستخدم: {model_name}")
    response = model.generate_content(prompt)
    
    raw_response = response.text
    print("\n📝 الرد الخام من Gemini:")
    print(raw_response)
    print("-" * 70)
    
    # محاولة استخراج JSON
    match = re.search(r'\{[\s\S]*\}', raw_response, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        result = json.loads(json_str)
        
        print("\n✅ تم استخراج JSON بنجاح!")
        print("\n" + "=" * 70)
        print("📊 التقرير النهائي (مفصل بالعربية):")
        print("=" * 70)
        
        # نقاط القوة
        print("\n💪 نقاط القوة:")
        for i, strength in enumerate(result.get("strengths", []), 1):
            print(f"   {i}. {strength}")
        
        # نقاط الضعف
        print("\n⚠️  نقاط الضعف:")
        for i, weakness in enumerate(result.get("weaknesses", []), 1):
            print(f"   {i}. {weakness}")
        
        # التوصيات
        print("\n💡 التوصيات:")
        for i, rec in enumerate(result.get("recommendations", []), 1):
            print(f"   {i}. {rec}")
        
        # الملخص
        if result.get("summary"):
            print(f"\n📋 الملخص:")
            print(f"   {result['summary']}")
        
        # الفرصة الرئيسية
        if result.get("main_opportunity"):
            print(f"\n🎯 أهم فرصة للتحسن:")
            print(f"   {result['main_opportunity']}")
        
        # التنبؤ المستقبلي
        if result.get("future_prediction"):
            print(f"\n🔮 التنبؤات المستقبلية:")
            print(f"   {result['future_prediction']}")
        
        print("\n" + "=" * 70)
        print("✅ التقرير كامل وجاهز للاستخدام")
        print("=" * 70)
        
        # طباعة JSON كاملاً للتحقق
        print("\n📄 JSON كاملاً:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    else:
        print("\n❌ لم يتمكن من استخراج JSON من الرد")
        print(f"الرد كان: {raw_response}")
        
except Exception as e:
    print(f"\n❌ خطأ: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ الاختبار انتهى")
print("=" * 70)
