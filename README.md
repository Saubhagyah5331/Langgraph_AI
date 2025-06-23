# LangGraph AI Multi-Agent Chatbot
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Saubhagyah5331/Langgraph_AI)

This repository contains a sophisticated multi-agent chatbot built with LangGraph. It features an intelligent routing system that directs user queries to specialized agents for tasks related to academics, news, shopping, and general conversation. A key feature is the "Human-in-the-Loop" (HITL) mechanism, which allows the chatbot to ask for feedback and clarification, improving its responses over time.

## Core Features
*   **Multi-Agent Architecture**: The system uses four distinct agents, each tailored for a specific domain.
*   **Intelligent Routing**: A router powered by Google's Gemini model classifies user queries and directs them to the appropriate agent.
*   **Human-in-the-Loop (HITL)**: The graph can pause and request user feedback (`yes`/`no`). If the user is unsatisfied, it asks for clarification to refine the query and generate a better response.
*   **Persistent Chat History**: Pymongo is used with `MongoDBSaver` to maintain conversation state, allowing for context-aware interactions.
*   **Rich CLI Interface**: An interactive and visually appealing command-line interface built with the `rich` library for a better user experience.
*   **Specialized Agents**:
    *   **Academic Agent**: Summarizes lecture notes and recommends educational YouTube videos.
    *   **News Agent**: Fetches the latest news on a given topic from the World News API and summarizes it.
    *   **Shopping Agent**: Recommends products from Amazon and provides detailed comparisons for products listed in a local database.
    *   **General Agent**: Handles any query that doesn't fit into the other categories.

## System Architecture
The application is orchestrated by a stateful graph defined in `graph.py` using LangGraph.

1.  **Entrypoint (`main.py`)**: Initializes the `ChatBotInterface` and the `MultiAgentGraph`. It handles the user interaction loop in the terminal.
2.  **Routing (`utils/router.py`)**: When a user submits a query, it first goes to the `router` node. The router uses a Gemini model to classify the query into one of four categories: `academic`, `news`, `shopping`, or `general`.
3.  **Agent Invocation (`agents/`)**: The graph transitions to the corresponding agent node (e.g., `academic_agent`). The selected agent processes the query using its specialized tools.
4.  **Tools (`tools/`)**: Each agent is equipped with tools that perform specific actions, such as calling an external API (World News API, Amazon API), searching YouTube, or querying a local JSON database.
5.  **Human Feedback (`graph.py`)**: After an agent generates a response, the graph transitions to the `human_feedback` node. This node uses `interrupt` to pause execution and asks the user if the response was helpful.
    *   If **"yes"**, the conversation ends for that turn.
    *   If **"no"**, the graph asks the user for clarification. The clarification is appended to the original query, and the process restarts from the `router` node with the updated query.
6.  **State Management (`utils/state_types.py`)**: The entire process is managed by a shared `AgentState`, which tracks the query, response, chat history, and feedback flags. The state is persisted in MongoDB across turns.

## Setup and Installation

### Prerequisites
*   Python 3.10+
*   Poetry for dependency management
*   A running MongoDB instance

### Steps
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saubhagyah5331/langgraph_ai.git
    cd langgraph_ai
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add the following API keys.

    ```env
    # Get from Google AI Studio
    Gemini_My_Tok_3="YOUR_GEMINI_API_KEY"

    # Get from https://worldnewsapi.com/
    news_api_key="YOUR_WORLD_NEWS_API_KEY"

    # Get from RapidAPI for the Real-Time Amazon Data API
    RAPID_API_KEY="YOUR_RAPIDAPI_KEY"
    ```

4.  **Ensure MongoDB is running:**
    The application connects to a local MongoDB instance by default (`mongodb://localhost:27017`). You can change the connection URI in `main.py` if needed.

## How to Run
Execute the main script to start the interactive chat session:

```bash
poetry run python main.py
```

The application will greet you, and you can start typing your queries.

### Example Interactions

*   **News Query**: `latest news about electric vehicles`
*   **Academic Query**: `Can you recommend YouTube videos on quantum mechanics?`
*   **Shopping Comparison Query**: `compare iPhone 15 vs Google Pixel 8`
*   **General Query**: `What is the capital of France?`

After a response is given, you will be prompted for feedback:

```
🙏 Was the response helpful? (yes/no)
Feedback (yes/no): no

📝 Please provide what you were expecting or how I can improve the response:
Your clarification: a more detailed comparison of the camera systems
```

The chatbot will then re-process the query with your added clarification.

## Project Structure
```
.
├── agents/             # Contains the logic for each specialized agent.
├── api_key_enpoints/   # Manages retrieval of API keys.
├── database/           # Contains local data files (e.g., product specs).
├── prompts/            # Stores all prompts for the LLM, organized by agent/purpose.
├── tools/              # Defines the tools each agent can use.
├── utils/              # Contains utility classes for the router, state, and LLM wrapper.
├── graph.py            # Defines the core multi-agent LangGraph structure and logic.
├── main.py             # The entry point for the CLI application.
└── pyproject.toml      # Project dependencies and configuration.