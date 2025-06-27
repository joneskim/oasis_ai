# Oasis AI - LangGraph Agent with OpenRouter

A basic LangGraph project demonstrating how to build an agent with OpenRouter integration and tool usage.

## Overview

This project implements a simple LangGraph agent that uses OpenRouter to access powerful language models. The agent is equipped with two tools:

1. **Web Search Tool**: Allows the agent to search the web for information (simulated in this demo)
2. **Calculator Tool**: Enables the agent to perform mathematical calculations

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory based on the `.env.example` template:
   ```
   cp .env.example .env
   ```
4. Add your OpenRouter API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Running the Agent

```
python app.py
```

This will start an interactive session where you can enter questions. The agent will:
1. Process your query
2. Decide whether to use one of its tools or respond directly
3. If a tool is used, execute it and incorporate the results
4. Provide a final response

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
└── app.py          # Main agent implementation with LangGraph
```

## How It Works

The agent uses a LangGraph workflow with three main nodes:

1. **agent_node**: Processes the user query and decides whether to use a tool
2. **tool_execution**: Executes the selected tool and captures the result
3. **output**: Formats and displays the final response

The workflow is dynamic and can loop back to the agent node after tool execution to process the results.

## Learning More

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [OpenRouter Documentation](https://openrouter.ai/docs)