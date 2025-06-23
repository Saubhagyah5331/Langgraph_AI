from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from utils.gemini import GeminiLLM
from prompts.general.general_prompt import general_tool_prompt

class GeneralQueryInput(BaseModel):
    query: str = Field(description="The general query or task to be processed.")

class GeneralQueryTool(BaseTool):
    name: str = "handle_general_query"
    description: str = "Handles general user queries or open-ended tasks."
    args_schema: Type[BaseModel] = GeneralQueryInput

    def _run(self, query: str) -> str:
        try:
            gemini_llm = GeminiLLM()
            chain = general_tool_prompt | gemini_llm.get_llm_with_parser()
            response = chain.invoke({"query": query})
            return response
        except Exception as e:
            return f"Error processing general query: {str(e)}"
