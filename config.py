"""
Configuration for OpenRouter API
"""
import openai
from os import getenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_openrouter_llm(model_name="anthropic/claude-3-opus:beta"):
    """
    Initialize and return an OpenRouter LLM instance.

    This function configures a ChatOpenAI instance to use OpenRouter
    with the necessary API key, base URL and headers.

    Args:
        model_name: The model to use from OpenRouter.

    Returns:
        A configured ChatOpenAI instance connected to OpenRouter.
    """
    # Get the API key and base URL from environment variables
    api_key = getenv("OPENROUTER_API_KEY")
    base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Define the headers for OpenRouter attribution
    default_headers = {
        "HTTP-Referer": getenv("YOUR_SITE_URL", "https://github.com/yourusername/oasis_ai"),
        "X-Title": getenv("YOUR_SITE_NAME", "Oasis AI"),
    }

    # Create the ChatOpenAI instance with all required parameters
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,  # This is needed for both sync and async clients
        openai_api_base=base_url,
        default_headers=default_headers,
    )
    return llm
