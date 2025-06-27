#!/usr/bin/env python3
"""
Basic LangGraph Application using OpenRouter with two tools
"""
import os
import json
import math
import requests
from typing import TypedDict, List, Union, Literal, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.tools import tool, ToolException
from langchain_core.language_models import LanguageModelLike
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langgraph.graph import StateGraph, END

# Import configuration
from config import get_openrouter_llm

# Load environment variables
load_dotenv()

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information about a query."""
    try:
        # Using a mock search API for demonstration
        # In a real application, you would use a real search API
        print(f"Searching the web for: {query}")
        
        # For demonstration, we'll return a mock response
        return f"Search results for '{query}': Found information about {query}. This is a simulated search result."
    except Exception as e:
        raise ToolException(f"Error searching the web: {str(e)}")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Using a safe eval approach for simple calculations
        # This is a simplified version and not secure for production
        print(f"Calculating: {expression}")
        allowed_names = {
            "abs": abs,
            "float": float,
            "int": int,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
            "sum": sum,
            "math": math
        }
        
        # Replace common math functions with their math module equivalents
        for func in dir(math):
            if not func.startswith("__"):
                expression = expression.replace(f"math.{func}", f"math.{func}")
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        raise ToolException(f"Error evaluating expression: {str(e)}")

# Define the available tools
tools = [search_web, calculator]

# Define the state for LangGraph
class AgentState(TypedDict):
    question: str
    agent_outcome: Any
    intermediate_steps: List

# Define the nodes for our graph
def agent_node(state: AgentState) -> AgentState:
    """Agent node for the graph.
    
    This node runs the agent on the current input.
    """
    # Get the LLM from config
    llm = get_openrouter_llm(model_name="google/gemini-2.5-flash-preview-05-20")
    
    # Use the global tools variable
    # No need to redefine tools here
    
    # Define the prompt for the agent
    # The OpenAI tools agent requires a specific format with agent_scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. \
        You have access to the following tools:
        - search_web: Search the web for information
        - calculator: Evaluate mathematical expressions
        
        Use these tools to help answer the user's question.
        Always think step-by-step and explain your reasoning."""),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    
    # Create the agent using the global tools
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    # Run the agent on the current input
    result = agent_executor.invoke({"input": state["question"]})
    
    # Update the state with the result
    return {
        "question": state["question"],
        "agent_outcome": result,
        "intermediate_steps": state.get("intermediate_steps", []) + [{"action": "agent_execution", "result": result}]
    }

def output_node(state: AgentState) -> AgentState:
    """Format and output the final result"""
    result = state["agent_outcome"]
    
    print("\n=== FINAL RESPONSE ===\n")
    print(result["output"])
    print("\n=====================\n")
    
    return state

# Build the graph
def build_graph():
    """Build the LangGraph workflow"""
    # Initialize the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("output", output_node)
    
    # Add edges
    graph.add_edge("agent", "output")
    
    # Set the entry point
    graph.set_entry_point("agent")
    
    # Compile the graph
    return graph.compile()

# Example using LangChain directly (as provided in the request)
def run_simple_chain():
    """Run a simple LangChain with OpenRouter using modern LCEL syntax."""
    # Get the LLM from config
    llm = get_openrouter_llm(model_name="google/gemini-2.5-flash-preview-05-20")
    
    # Create a prompt template
    template = """Question: {question}
Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    # Create a chain using LCEL (LangChain Expression Language)
    llm_chain = prompt | llm
    
    # Run the chain
    question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
    print("\n=== SIMPLE CHAIN EXAMPLE ===\n")
    print(f"Question: {question}")
    
    # Use .invoke() instead of the deprecated .run()
    response = llm_chain.invoke({"question": question})
    
    # The response from `prompt | llm` is a message object, so we access its content
    print(f"Response: {response.content}")
    print("\n===========================\n")

def main():
    # """Main function to run the agent"""
    # # First run the simple chain example
    run_simple_chain()
    
    # Then run the LangGraph agent
    print("\nNow running the LangGraph agent with tools...\n")
    
    # Build the graph
    graph = build_graph()
    
    # Get user input
    while True:
        user_query = input("Enter your question: ")
        
        # Initialize the state
        initial_state = {
            "question": user_query,
            "agent_outcome": None,
            "intermediate_steps": []
        }
    
    # Run the graph
        for state in graph.stream(initial_state):
            pass  # The agent and output nodes will handle everything

if __name__ == "__main__":
    main()
