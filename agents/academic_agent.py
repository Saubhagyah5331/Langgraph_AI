from utils.gemini import GeminiLLM
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent
from prompts.academic.academic_agent_prompt import system_message
from tools.academic_tools import AcademicNoteSummarizerTool, YouTubeVideoRecommenderTool


class AcademicAgent:
    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.tools = [
            AcademicNoteSummarizerTool(),
            YouTubeVideoRecommenderTool()
        ]
        # self.prompt = system_message
        self.agent: Runnable = self._build_agent()

    def _build_agent(self) -> Runnable:
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            # prompt=self.prompt
        )

    def run(self, user_input: str) -> str:
        return self.agent.invoke({
            "messages": [{
                "role": "user", 
                "content": user_input}]
        })
