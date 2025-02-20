# Weaviate RAG Example

This example demonstrates how to use Weaviate Cloud for Retrieval Augmented Generation (RAG) with OpenAI integration. The example shows how to:

1. Set up a Weaviate Cloud client
2. Create a collection with vector search and generative capabilities
3. Import and chunk text data
4. Perform RAG operations (single object and grouped generation)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Sign up for Weaviate Cloud:
   - Go to [Weaviate Cloud](https://console.weaviate.cloud/)
   - Create an account and set up a cluster
   - Get your cluster URL and API key

3. Create a `.env` file with your API keys:
```bash
WEAVIATE_API_KEY='your-weaviate-cloud-api-key'    # From Weaviate Cloud Console
WEAVIATE_URL='your-cluster-url'                   # Your Weaviate Cloud cluster URL (e.g., https://your-cluster.weaviate.network)
OPENAI_API_KEY='your-openai-api-key'              # From OpenAI platform
```

## Running the Example

1. Run the example script:
```bash
python weaviate_rag_example.py
```

2. The script will:
   - Download sample text about Git
   - Create a Weaviate collection
   - Import chunked text data
   - Demonstrate RAG capabilities:
     - Generate a haiku from a single chunk
     - Perform semantic search and generate a tweet

## Features

- Text chunking with overlap
- Vector search using OpenAI embeddings
- Single object text generation
- Grouped text generation
- Semantic search integration

## Notes

- The example uses GPT-4 through OpenAI's API
- Text is chunked into 150-word pieces with 25-word overlap
- The collection is recreated each time for demonstration purposes
- All operations are performed using Weaviate's Python client v4
- Uses Weaviate Cloud for deployment (no local setup needed)

## Additional Resources

- [Weaviate Cloud Documentation](https://weaviate.io/developers/weaviate/installation/weaviate-cloud)
- [OpenAI Documentation](https://platform.openai.com/docs)
- [RAG Best Practices](https://weaviate.io/developers/weaviate/starter-guides/retrieval-augmented-generation)
- [Weaviate Cloud Console](https://console.weaviate.cloud/)

