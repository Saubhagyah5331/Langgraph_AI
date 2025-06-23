from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from api_key_enpoints.gemini_client import get_gemini_api_key


class GeminiLLM:
    """Class-based Gemini LLM wrapper for LangChain + LangGraph agent/tool compatibility."""

    def __init__(self, model_name: str = 'gemini-2.0-flash'):
        self.api_key = get_gemini_api_key()
        self.model_name = model_name
        self.parser = StrOutputParser()
        self._llm = None
        self._llm_with_parser = None

    def get_llm(self):
        """
        Return Gemini LLM (ChatGoogleGenerativeAI) compatible with create_react_agent.
        This supports bind_tools, invoke, etc.
        """
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key
            )
        return self._llm

    def get_llm_with_parser(self):
        """
        Return Gemini LLM piped with output parser (for summarization etc).
        """
        if self._llm_with_parser is None:
            self._llm_with_parser = self.get_llm() | self.parser
        return self._llm_with_parser
