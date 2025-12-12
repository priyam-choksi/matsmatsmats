import os
import requests
from dotenv import load_dotenv

# Load env from your project root
load_dotenv(r"F:\DAMG 7374_GENAI\TradingAgent\.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"GROQ_API_KEY: {GROQ_API_KEY[:20] if GROQ_API_KEY else 'NOT FOUND'}...")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY[:20] if OPENAI_API_KEY else 'NOT FOUND'}...")

# Models to test
MODELS = {
    "GPT-4o-mini": ("openai", "gpt-4o-mini"),
    "Llama-3.3-70B": ("groq", "llama-3.3-70b-versatile"),
    "Llama-4-Maverick": ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"),
    "Kimi-K2": ("groq", "moonshotai/kimi-k2-instruct-0905"),
    "Qwen3-32B": ("groq", "qwen/qwen3-32b"),
    "GPT-OSS-120B": ("groq", "openai/gpt-oss-120b"),
    "Allam-2-7B": ("groq", "allam-2-7b"),  # Testing if this works
}

def test_model(name, provider, model):
    print(f"\n🧪 Testing {name} ({provider}/{model})...")
    
    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        else:  # groq
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
            "max_tokens": 10
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
            print(f"   ✅ SUCCESS: {reply}")
            return True
        else:
            print(f"   ❌ FAILED: {resp.status_code} - {resp.text[:100]}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

# Test all models
print("\n" + "="*50)
print("TESTING ALL MODELS")
print("="*50)

results = {}
for name, (provider, model) in MODELS.items():
    results[name] = test_model(name, provider, model)

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for name, passed in results.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {name}")