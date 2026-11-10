from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

'''print("Available models:")
for m in models.data:
    print("-", m.id)
    
 - whisper-large-v3-turbo
- openai/gpt-oss-120b
- canopylabs/orpheus-arabic-saudi
- canopylabs/orpheus-v1-english
- meta-llama/llama-prompt-guard-2-86m
- openai/gpt-oss-safeguard-20b
- meta-llama/llama-prompt-guard-2-22m
- qwen/qwen3.8-27b
- whisper-large-v3
- allam-2-7b
- openai/gpt-oss-20b'''


response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {"role": "user", "content": "Explain cloud computing to a beginner in three sentences."}
    ]
)

print(response.choices[0].message.content)


