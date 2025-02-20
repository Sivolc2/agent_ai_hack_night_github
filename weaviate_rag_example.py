import os
import logging
from typing import List, Optional
import requests
import re
from dataclasses import dataclass
from pathlib import Path

import weaviate
from weaviate.classes.init import Auth
from weaviate.collections import Collection
from weaviate.exceptions import WeaviateQueryError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeaviateConfig:
    """Configuration for Weaviate client."""
    url: str
    api_key: str
    openai_api_key: str
    collection_name: str = "DocumentChunks"
    chunk_size: int = 150
    chunk_overlap: int = 25
    batch_size: int = 100

class WeaviateClient:
    """Manages Weaviate operations with improved error handling and configuration."""
    
    def __init__(self, config: WeaviateConfig):
        self.config = config
        self.client = self._setup_client()
        self.collection = self._get_or_create_collection()

    def _setup_client(self) -> weaviate.WeaviateClient:
        """Initialize and return a Weaviate client with proper error handling."""
        try:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.config.url,
                auth_credentials=Auth.api_key(self.config.api_key),
                headers={
                    "X-OpenAI-Api-Key": self.config.openai_api_key
                }
            )
            
            # Verify connection
            client.collections.get("_schema")
            logger.info("Successfully connected to Weaviate Cloud!")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise ConnectionError(f"Weaviate connection failed: {str(e)}")

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create a new one with proper configuration."""
        try:
            if self.client.collections.exists(self.config.collection_name):
                logger.info(f"Using existing collection: {self.config.collection_name}")
                return self.client.collections.get(self.config.collection_name)
            
            logger.info(f"Creating new collection: {self.config.collection_name}")
            collection = self.client.collections.create(
                name=self.config.collection_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="content",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="title",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="chunk_index",
                        data_type=weaviate.classes.config.DataType.INT
                    ),
                    weaviate.classes.config.Property(
                        name="metadata",
                        data_type=weaviate.classes.config.DataType.TEXT
                    )
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),  # Use OpenAI as vectorizer
                generative_config=weaviate.classes.config.Configure.Generative.openai()  # Use OpenAI for generation
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to create/get collection: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Improved text chunking with better handling of sentence boundaries."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences (basic implementation - can be improved with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_size = len(sentence_words)
            
            if current_size + sentence_size <= self.config.chunk_size:
                current_chunk.extend(sentence_words)
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = sentence_words
                current_size = sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def import_data(self, chunks: List[str], title: str, metadata: Optional[dict] = None):
        """Import data with batching and progress tracking."""
        try:
            batch = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                data_object = {
                    "content": chunk,
                    "title": title,
                    "chunk_index": i,
                    "metadata": str(metadata or {})
                }
                batch.append(data_object)
                
                if len(batch) >= self.config.batch_size or i == total_chunks - 1:
                    with self.collection.batch.dynamic() as batch_writer:
                        for obj in batch:
                            batch_writer.add_object(obj)
                    
                    logger.info(f"Imported batch: {i+1}/{total_chunks} chunks")
                    batch = []
                    
        except Exception as e:
            logger.error(f"Failed to import data: {str(e)}")
            raise

    def search_and_generate(self, query: str, limit: int = 5) -> dict:
        """Enhanced semantic search with generation capabilities."""
        try:
            # Use the simpler direct generation approach
            response = self.collection.generate.near_text(
                query=query,
                limit=limit,
                grouped_task="""
                Synthesize a comprehensive answer based on these passages.
                Include relevant details while maintaining accuracy.
                If uncertain, acknowledge limitations in the available information.
                """
            )
            
            return {
                'generated_response': response.generated,
                'matching_chunks': [
                    {
                        'content': obj.properties.get('content', ''),
                        'title': obj.properties.get('title', ''),
                    }
                    for obj in response.objects
                ] if hasattr(response, 'objects') else []
            }
            
        except Exception as e:
            logger.error(f"Search query failed: {str(e)}")
            raise

def load_environment() -> WeaviateConfig:
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = {
        "WEAVIATE_URL": os.getenv("WCD_URL"),  # Updated environment variable name
        "WEAVIATE_API_KEY": os.getenv("WCD_API_KEY"),  # Updated environment variable name
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    return WeaviateConfig(
        url=required_vars["WEAVIATE_URL"],
        api_key=required_vars["WEAVIATE_API_KEY"],
        openai_api_key=required_vars["OPENAI_API_KEY"]
    )

def main():
    """Main execution function with improved error handling and demo capabilities."""
    client = None
    try:
        # Initialize configuration and client
        config = load_environment()
        client = WeaviateClient(config)
        
        # Demo: Download and process sample text
        sample_url = "https://raw.githubusercontent.com/progit/progit2/main/book/01-introduction/sections/what-is-git.asc"
        response = requests.get(sample_url)
        response.raise_for_status()
        
        # Process and import content
        chunks = client.chunk_text(response.text)
        client.import_data(
            chunks=chunks,
            title="Git Introduction",
            metadata={"source": sample_url, "type": "documentation"}
        )
        
        # Demonstrate search and generation
        demo_queries = [
            "What is Git and why was it created?",
            "Explain the main benefits of using Git",
            "How does Git handle version control differently from other systems?"
        ]
        
        for query in demo_queries:
            logger.info(f"\nQuery: {query}")
            result = client.search_and_generate(query)
            logger.info(f"Generated Response:\n{result['generated_response']}\n")
            logger.info("Top matching chunks:")
            for i, chunk in enumerate(result['matching_chunks'], 1):
                logger.info(f"{i}. {chunk['content'][:200]}...")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    finally:
        if client and hasattr(client, 'client'):
            client.client.close()
            logger.info("Weaviate client connection closed.")

if __name__ == "__main__":
    main() 