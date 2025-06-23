import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

def get_gemini_api_key():
    return os.getenv("Gemini_My_Tok_3")

