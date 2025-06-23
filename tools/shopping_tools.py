import os
import json
from typing import Type
from pydantic import BaseModel, Field
import requests
from api_key_enpoints.amazon_api_client import get_amazon_api_client
from langchain_core.tools import BaseTool
from utils.gemini import GeminiLLM
from prompts.shopping.recommend_prompt import recommend_prompt
from prompts.shopping.compare_prompt import compare_prompt


# --- Amazon Product Recommender Tool ---

class AmazonProductInput(BaseModel):
    query: str = Field(description="The search query for the product.")

class AmazonProductRecommender(BaseTool):
    name: str = "recommend_product"
    description: str = "Recommends products based on the search query using Amazon data."
    args_schema: Type[BaseModel] = AmazonProductInput
    

    def _run(self, query: str) -> str:
        try:
            api_key = get_amazon_api_client()
            if not api_key:
                return "API key not found. Set RAPID_API_KEY in .env"

            base_url = "https://real-time-amazon-data.p.rapidapi.com/search"
            headers = {
                "x-rapidapi-key": api_key,
                "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
            }
            params = {
                "query": query,
                "page": "1",
                "country": "IN",
                "sort_by": "RELEVANCE",
                "max_price": "20000",
                "product_condition": "ALL",
                "is_prime": "false",
                "deals_and_discounts": "NONE"
            }

            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code != 200:
                return f"Error fetching product data: {response.status_code}"

            results = response.json()
            products = self._parse_product_data(results)

            # Summarize via Gemini
            gemini = GeminiLLM()
            chain = recommend_prompt | gemini.get_llm_with_parser()
            return chain.invoke({"query": query, "products": products})

        except Exception as e:
            return f"Error in product recommendation: {str(e)}"

    def _parse_product_data(self, results: dict) -> list:
        """Parse product data from API response."""
        return [
            {
                "name": item.get('product_title', 'No Title'),
                "price": item.get('product_price', 'N/A'),
                "original_price": item.get('product_original_price', 'N/A'),
                "url": item.get('product_url', '#')
            }
            for item in results.get("data", {}).get("products", [])
        ]


# --- Product Comparator Tool ---

class ProductComparatorInput(BaseModel):
    query: str = Field(description="Format: 'Product A vs Product B")

class ProductComparator(BaseTool):
    name: str = "compare_products"
    description: str = "Compares two products. Use format: 'Product A vs Product B'."
    args_schema: Type[BaseModel] = ProductComparatorInput

    def _run(self, query: str) -> str:
        if " vs " not in query:
            return "Invalid format. Use 'Product A vs Product B'."

        try:
            p1_name, p2_name = query.split(" vs ")
            database_path = "database/product_data.json"
            with open(database_path, "r") as f:
                products = json.load(f)

            p1 = self._find_product(products, p1_name)
            p2 = self._find_product(products, p2_name)

            if not p1 or not p2:
                return "One or both products not found in the database."

            # Summarize comparison using Gemini
            gemini = GeminiLLM()
            chain = compare_prompt | gemini.get_llm_with_parser()
            return chain.invoke({"product1": p1, "product2": p2})

        except Exception as e:
            return f"Error comparing products: {str(e)}"

    def _find_product(self, products: list, name: str):
        return next((p for p in products if name.lower() in p["name"].lower()), None)
