import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # هذا الموديل هو الأضمن للاختبار
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("test")
    print("✅ المفتاح سليم والموديل يعمل!")
except Exception as e:
    print(f"❌ الخلل هنا: {e}")