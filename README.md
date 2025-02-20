# Phoenix OpenAI Tracing Example

This example demonstrates how to use Arize Phoenix to trace OpenAI API calls in your application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Start Phoenix server:
```bash
phoenix serve
```
This will start the Phoenix server on http://localhost:6006 with the gRPC endpoint on port 4317.

## Running the Example

1. Run the example script:
```bash
python phoenix_example.py
```

2. View traces:
- Open http://localhost:6006 in your browser
- Navigate to the Traces section
- You should see your OpenAI API calls being traced

## Features

- Automatic tracing of OpenAI API calls
- Error handling and logging
- Example of generating a haiku using GPT-4
- Integration with Phoenix for monitoring and debugging

## Notes

- Make sure the Phoenix server is running before executing the script
- The example uses GPT-4, but you can modify the model in the code
- All API calls are automatically traced and visible in the Phoenix UI

