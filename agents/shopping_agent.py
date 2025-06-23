from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from utils.gemini import GeminiLLM
from prompts.shopping.shopping_agent_prompt import shopping_agent_prompt
from tools.shopping_tools import AmazonProductRecommender, ProductComparator


class ShoppingAgent:
    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.tools = [AmazonProductRecommender(), ProductComparator()]
        # self.prompt = shopping_agent_prompt  # should be a ChatPromptTemplate
        self.agent: Runnable = self._build_agent()

    def _build_agent(self) -> Runnable:
        return create_react_agent(
            tools=self.tools,
            model=self.llm,
            # prompt=self.prompt  # ✅ ensure this is a valid ChatPromptTemplate
        )

    def run(self, user_input: str) -> str:
        return self.agent.invoke({
            "messages": [{
                "role": "user",
                "content": user_input
            }]
        })
