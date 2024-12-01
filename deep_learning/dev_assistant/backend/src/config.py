import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_var(name, cast=None, required=True, error_message=None):
    """
    Fetches an environment variable and optionally casts its value.

    Args:
        name (str): Name of the environment variable.
        cast (callable, optional): A function to cast the variable (e.g., int, float).
        required (bool): Whether the variable is required.
        error_message (str): Custom error message for missing/invalid variables.

    Returns:
        The value of the environment variable, cast to the desired type if specified.

    Raises:
        ValueError: If the variable is required but not set or casting fails.
    """
    value = os.getenv(name)
    if required and not value:
        raise ValueError(error_message or f"{name} is not set in the environment variables.")

    if cast:
        try:
            return cast(value)
        except (ValueError, TypeError):
            raise ValueError(error_message or f"{name} must be a valid {cast.__name__}.")

    return value


# Required OpenAI API credentials
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", required=True)
OPENAI_PROJECT_ID = get_env_var("OPENAI_PROJECT_ID", required=True)

# Required OpenAI API settings
OPENAI_MODEL = get_env_var("OPENAI_MODEL", required=True)
OPENAI_TEMPERATURE = get_env_var(
    "OPENAI_TEMPERATURE", cast=float, error_message="OPENAI_TEMPERATURE must be a valid float."
)
OPENAI_MAX_TOKENS = get_env_var(
    "OPENAI_MAX_TOKENS", cast=int, error_message="OPENAI_MAX_TOKENS must be a valid integer."
)

USE_REAL_OPENAI_API = (
    get_env_var(
        "USE_REAL_OPENAI_API",
        cast=lambda x: x.lower() in ["true", "1", "yes"],  # Proper boolean casting
        required=False,
        error_message="USE_REAL_OPENAI_API must be a boolean.",
    )
    or False
)

# Dummy response for testing
DUMMY_RESPONSE = "This is a dummy response for testing purposes."
