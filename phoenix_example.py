import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import openai
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Phoenix tracer
tracer_provider = register(
    project_name="openai-demo",  # Your project name
    endpoint="http://localhost:4317",  # Default Phoenix gRPC endpoint
)

# Initialize OpenAI instrumentation
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

def generate_haiku():
    """Generate a haiku using OpenAI."""
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment
    
    try:
        logger.info("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4",  # Fixed model name (was gpt-4o)
            messages=[
                {"role": "system", "content": "You are a haiku expert."},
                {"role": "user", "content": "Write a nature-themed haiku."}
            ],
            temperature=0.7,
            max_tokens=50  # Added token limit for efficiency
        )
        logger.info("Successfully received response from OpenAI")
        return response.choices[0].message.content
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def main():
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return
    
    logger.info("Starting haiku generation...")
    haiku = generate_haiku()
    if haiku:
        print("\nGenerated Haiku:")
        print(haiku)
    else:
        print("\nFailed to generate haiku. Please check the logs for details.")

if __name__ == "__main__":
    main() 