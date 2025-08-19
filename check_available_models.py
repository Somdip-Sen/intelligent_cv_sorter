import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your environment variables to get the API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in your .env file.")

# Configure the library with your key
genai.configure(api_key=api_key)

print("--- Finding available models that support 'generateContent' ---")

# The core of the script: list_models()
for m in genai.list_models():
  # We filter for models that can be used for our text generation task
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

print("----------------------------------------------------------")