import google.generativeai as genai
import os
from dotenv import load_dotenv

# تحميل المفتاح من ملف .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"--- فحص الإعدادات ---")
print(f"المفتاح المستخرج: {api_key[:10]}****" if api_key else "❌ لم يتم العثور على مفتاح في ملف .env")

genai.configure(api_key=api_key)

def check_model(model_name):
    try:
        print(f"محاولة الاتصال بموديل: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello")
        print(f"✅ نجح الاتصال بـ {model_name}: {response.text}")
        return True
    except Exception as e:
        print(f"❌ فشل {model_name}: {e}")
        return False

available_models = [
    m.name.replace('models/', '')
    for m in genai.list_models()
    if 'generateContent' in getattr(m, 'supported_generation_methods', [])
]

if available_models:
    success = any(check_model(name) for name in available_models[:5])
    if success:
        print("\n💡 استنتاج: كل شيء سليم! تم استخدام موديل متاح من API.")
    else:
        print("\n⚠️ استنتاج: الخلل غالباً في صلاحية المفتاح (API Key) أو إصدار المكتبة.")
else:
    print("\n⚠️ استنتاج: لا توجد موديلات generateContent متاحة من هذا المفتاح.")