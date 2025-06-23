from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from tools.news_tools import NewsArticleSummarizerTool
from utils.gemini import GeminiLLM
from prompts.news.news_agent_prompt import news_agent_prompt  # This should be a ChatPromptTemplate


class NewsAgent:
    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.tools = [NewsArticleSummarizerTool()]
        # self.prompt = news_agent_prompt
        self.agent: Runnable = self._build_agent()

    def _build_agent(self) -> Runnable:
        return create_react_agent(
            tools=self.tools,
            model=self.llm,
            # prompt=self.prompt
        )

    def run(self, user_input: str) -> str:
        return self.agent.invoke({
            "messages": [{
                "role": "user",
                "content": user_input
            }]
        })
