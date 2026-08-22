import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
if not key:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env")
    exit(1)

print(f"✅ Found Key: {key[:5]}...{key[-5:]}")

# Configure and list
genai.configure(api_key=key)

try:
    print("Querying available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")