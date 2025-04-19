import json
from google import genai as google_genai
import openai
import os

def get_model(model_name: str = 'gemini-2.0-flash-lite', model_type: str = 'gemini'):
    """
    Retrieves the specified model.

    Args:
        model_name (str): The name of the model to use (default: 'gemini-2.0-flash-lite').
        model_type (str): The type of model to use ('gemini' or 'openai').

    Returns:
        genai.Client or openai.OpenAI: The requested model.

    Raises:
        ValueError: If the API key for the model is not found in the JSON file or if an invalid model_type is specified.
        FileNotFoundError: If the api_keys.json file is not found.
        json.JSONDecodeError: If the api_keys.json file contains invalid JSON.
    """

    try:
        # Get the absolute path to the api_keys.json file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys_path = os.path.join(script_dir, "api_keys.json")
        
        with open(api_keys_path, "r") as f:
            api_keys = json.load(f)

        if model_type == 'gemini':
            api_key = api_keys.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in api_keys.json")

            # Create and return a Gemini client
            client = google_genai.Client(api_key=api_key)
            return client
        elif model_type == 'openai':
            api_key = api_keys.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in api_keys.json")
            openai.api_key = api_key
            return openai
        else:
            raise ValueError("Invalid model_type. Choose 'gemini' or 'openai'.")

    except FileNotFoundError:
        raise FileNotFoundError(f"api_keys.json file not found at {api_keys_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in api_keys.json")
