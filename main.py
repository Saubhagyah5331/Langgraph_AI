from graph import MultiAgentGraph
from utils.state_types import AgentState
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage


def extract_final_response(messages: list[BaseMessage]) -> str:
    """Extracts final AI response and tool traces for display."""
    ai_content = ""
    tool_content = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            tool_content.append(msg.content.strip())
        elif isinstance(msg, AIMessage) and msg.content:
            ai_content = msg.content.strip()  # last AIMessage takes precedence

    # Combine tool outputs (if any) + AI final response
    combined = ""
    if tool_content:
        combined += "[bold yellow]🔧 Tool Outputs:[/bold yellow]\n" + "\n\n".join(tool_content) + "\n\n"
    if ai_content:
        combined += "[bold cyan]🤖 Final Answer:[/bold cyan]\n" + ai_content
    return combined if combined else "⚠️ No valid AI response or tool output found."


class ChatBotInterface:
    """Class-based interface for the multi-agent chatbot."""

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "langgraph"):
        self.console = Console()
        self.mongo_client = MongoClient(mongo_uri)
        self.saver = MongoDBSaver(self.mongo_client, db_name=db_name)
        self.thread_id = "1"
        self.chat_history = []
        self.graph_builder = MultiAgentGraph(self.saver)
        self.graph = None

    def initialize(self):
        """Initialize the chatbot graph."""
        logger.info("Building the LangGraph with Human-in-the-Loop...")
        self.graph = self.graph_builder.build_graph()

    def print_bot_message(self, message: str):
        """Display bot message with styling."""
        self.console.print(Panel.fit(
            message,
            title="🤖 Bot",
            border_style="cyan"
        ))

    def print_user_prompt(self) -> str:
        """Get user input with styling."""
        return self.console.input("[bold green]You[/bold green]: ").strip()

    def print_error_message(self, message: str):
        """Display error message with styling."""
        self.console.print(Panel.fit(
            f"[bold red]{message}[/bold red]",
            title="❌ Exception",
            border_style="red"
        ))

    def print_feedback_prompt(self, interrupt_data: dict) -> str:
        """Display feedback prompt based on interrupt type."""
        interrupt_type = interrupt_data.get("type", "unknown")

        if interrupt_type == "satisfaction_check":
            if interrupt_data.get("response"):
                response_messages = interrupt_data["response"].get("messages", [])
                formatted = extract_final_response(response_messages)
                self.print_bot_message(formatted)

            self.console.print(f"\n[bold magenta]🙏 {interrupt_data.get('message', 'Was the response helpful?')}[/bold magenta]")
            return self.console.input("[bold green]Feedback (yes/no):[/bold green] ").strip()

        elif interrupt_type == "clarification_request":
            self.console.print(f"\n[bold yellow]📝 {interrupt_data.get('message', 'Please provide clarification:')}[/bold yellow]")
            return self.console.input("[bold green]Your clarification:[/bold green] ").strip()

        else:
            self.console.print(f"\n[bold magenta]{interrupt_data.get('message', 'Please provide input:')}[/bold magenta]")
            return self.console.input("[bold green]Input:[/bold green] ").strip()

    def create_initial_state(self, query: str) -> AgentState:
        """Create initial state for the graph."""
        return {
            "query": query,
            "response": "",
            "next": "",
            "chat_history": self.chat_history,
            "feedback_received": False,
            "clarification_received": False,
        }

    def process_query(self, query: str):
        """Process a single query through the graph."""
        logger.debug(f"Received user input: {query}")
        state = self.create_initial_state(query)
        config = {"configurable": {"thread_id": self.thread_id}}

        logger.info("Invoking the graph...")
        result = self.graph.invoke(state, config=config)

        # Handle interrupts
        while result.get('__interrupt__'):
            interrupts = result['__interrupt__']
            logger.info(f"📋 Received {len(interrupts)} interrupt(s)")

            if interrupts:
                interrupt_data = interrupts[0].value
                feedback = self.print_feedback_prompt(interrupt_data)
                logger.info(f"🔄 Resuming with feedback: {feedback}")
                result = self.graph.invoke(Command(resume=feedback), config=config)

        if result.get("chat_history"):
            self.chat_history = result["chat_history"]
        else:
            self.chat_history.append(HumanMessage(content=query))

        # ✅ Extract and display combined AI + tool content
        final_response = ""

        if isinstance(result.get("response"), dict) and "messages" in result["response"]:
            final_response = extract_final_response(result["response"]["messages"])
        elif isinstance(result.get("messages"), list):
            final_response = extract_final_response(result["messages"])
        elif isinstance(result.get("response"), str):
            final_response = result["response"]

        final_response = final_response or "⚠️ No valid AI response found."
        self.print_bot_message(final_response)
        result["response"] = final_response  # store clean response for feedback loop

    def run(self):
        """Main chat loop."""
        if not self.graph:
            self.initialize()

        self.console.print(Panel.fit(
            "🤖 [bold cyan]Multi-Agent ChatBot with Human-in-the-Loop is Ready![/bold cyan]\nType [bold yellow]'exit'[/bold yellow] to quit.",
            title="LangGraph ChatBot",
            border_style="green"
        ))

        while True:
            try:
                query = self.print_user_prompt()

                if query.lower() == "exit":
                    self.console.print("[bold red]Exiting... Goodbye! 👋[/bold red]")
                    break

                self.process_query(query)

            except Exception as e:
                logger.exception("Exception during execution:")
                self.print_error_message(str(e))


if __name__ == "__main__":
    chatbot = ChatBotInterface()
    chatbot.run()
