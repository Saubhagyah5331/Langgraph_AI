import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

def get_amazon_api_client():
    return os.getenv("RAPID_API_KEY")

