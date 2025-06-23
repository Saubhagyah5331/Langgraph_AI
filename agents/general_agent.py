from langgraph.prebuilt import create_react_agent
from tools.general_tools import GeneralQueryTool
from utils.gemini import GeminiLLM
from langchain_core.runnables import Runnable
from prompts.general.general_agent_prompt import prompt

class GeneralAgent:
    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.tools = [GeneralQueryTool()]
        # self.prompt = prompt
        self.agent: Runnable = self._build_agent()

    def _build_agent(self) -> Runnable:
        return create_react_agent(
            tools=self.tools,
            model=self.llm,
            # prompt= self.prompt
        )

    def run(self, user_input: str) -> str:
        return self.agent.invoke({
            "messages": [{
                "role": "user", 
                "content": user_input}]
        })
