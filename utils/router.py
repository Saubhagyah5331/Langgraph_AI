from langchain_core.messages import HumanMessage
from loguru import logger
from utils.gemini import GeminiLLM
from prompts.router.classification_prompt import classification_prompt


class AgentRouter:
    """
    Class-based agent router that uses Gemini LLM to classify a user query
    into one of the predefined agent categories.
    """

    def __init__(self):
        self.llm = GeminiLLM().get_llm()
        self.valid_agents = {"academic", "news", "shopping", "general"}
        self.prompt = classification_prompt

    def classify_agent(self, user_input: str) -> str:
        """
        Classify the user query into an agent category.

        Parameters:
            user_input (str): The user's input query.

        Returns:
            str: One of "academic", "news", "shopping", or "general".
        """
        try:
            messages = self.prompt.format_messages(messages=[HumanMessage(content=user_input)])
            logger.debug("Formatted classification prompt:\n{}", messages)

            response = self.llm.invoke(messages)
            result = response.content.strip().lower()
            logger.success(f"Classification result: {result}")

            if result not in self.valid_agents:
                logger.warning(f"Unrecognized category: '{result}', defaulting to 'general'")
                return "general"

            return result

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "general"
