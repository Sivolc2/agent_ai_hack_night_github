from brain_edge_interaction import BrainEdgeSystem, Colors
from weaviate_rag_example import WeaviateClient, WeaviateConfig, load_environment
from typing import Dict, List, Optional, Union
import json
import yaml
import requests
from pathlib import Path

class DataLoader:
    """Handles loading data from local files and URLs."""
    
    @staticmethod
    def load_text_file(file_path: Union[str, Path]) -> str:
        """Load text from a local file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    @staticmethod
    def load_yaml_file(file_path: Union[str, Path]) -> List[Dict]:
        """Load YAML data from a local file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    @staticmethod
    def load_from_url(url: str) -> str:
        """Load text data from a URL."""
        response = requests.get(url)
        response.raise_for_status()
        return response.text

class RAGBrainEdgeSystem:
    def __init__(self, verbose: bool = True):
        """Initialize both Weaviate and Brain-Edge systems."""
        self.brain_edge = BrainEdgeSystem(verbose=verbose)
        self.weaviate_config = load_environment()
        self.weaviate_client = WeaviateClient(self.weaviate_config)
        self.verbose = verbose
        self.data_loader = DataLoader()
        
        # Load local data
        self.load_local_data()
        
    def load_local_data(self):
        """Load data from local files."""
        try:
            # Load user profiles
            self.user_profiles = self.data_loader.load_yaml_file("data/user_profile.yaml")
            
            # Load apartment listings
            self.apartment_data = self.data_loader.load_text_file("data/raw_text.txt")
            
            # Process apartment data into structured format
            self.apartment_listings = self._process_apartment_data(self.apartment_data)
            
            self._log("Successfully loaded local data files", Colors.HEADER)
        except Exception as e:
            self._log(f"Error loading local data: {str(e)}", Colors.HEADER)
            raise

    def _process_apartment_data(self, raw_data: str) -> List[Dict]:
        """Process raw apartment data into structured format."""
        # Split the raw text into individual listings
        # This is a simple implementation - you might want to enhance this based on your data structure
        listings = []
        current_listing = {}
        lines = raw_data.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_listing:
                    listings.append(current_listing.copy())
                    current_listing = {}
                continue
                
            # Basic parsing of key information
            if line.startswith('$'):
                current_listing['price'] = line
            elif 'bed' in line.lower():
                current_listing['bedrooms'] = line
            elif 'bath' in line.lower():
                current_listing['bathrooms'] = line
            elif 'San Francisco, CA' in line:
                current_listing['location'] = line
            elif 'PET FRIENDLY' in line:
                current_listing['pet_friendly'] = True
            elif line.startswith('Description'):
                current_listing['description'] = line
                
        if current_listing:
            listings.append(current_listing)
            
        return listings

    def _log(self, message: str, color: str = Colors.RESPONSE):
        """Helper method for logging if verbose is enabled."""
        if self.verbose:
            print(Colors.wrap(color, message))

    def get_user_context(self, user_email: str) -> Optional[Dict]:
        """Get user profile context for personalization."""
        for profile in self.user_profiles:
            if profile["email"] == user_email:
                return profile
        return None

    def process_query_with_context(self, user_query: str, user_email: Optional[str] = None, num_results: int = 3) -> Dict:
        """
        Process a query using local apartment data and optional user context.
        """
        # Get user context if email provided
        user_context = self.get_user_context(user_email) if user_email else None
        
        # Filter relevant listings based on the query
        relevant_listings = self._filter_listings(user_query, num_results)
        
        # Format listings for the brain
        context_str = "\n\n".join([
            f"Listing {i+1}:\n{json.dumps(listing, indent=2)}"
            for i, listing in enumerate(relevant_listings)
        ])

        # Create situation prompt for the brain
        situation = f"""Given this user query: "{user_query}"

Available apartment listings from our local database:

{context_str}

{"User Profile Information:" if user_context else ""}
{json.dumps(user_context, indent=2) if user_context else ""}

Based on this information, analyze the context and create specific tasks for our edge instances.
The first edge instance should focus on analyzing and extracting key information from the listings, including:
- Prices and availability
- Location details
- Amenities and features
- Pet policies
- Any special offers

The second edge instance should focus on:
{"- Matching the listings with the user's preferences and providing personalized recommendations" if user_context else "- Generating a comprehensive summary and recommendations based on the listings"}

Remember: You are the brain (R1) coordinating two edge instances (V3).
"""

        # Process with brain-edge system
        self._log("\nProcessing with Brain-Edge system...", Colors.HEADER)
        result = self.brain_edge.process_situation(situation)

        # Add context to result
        result["listings"] = relevant_listings
        if user_context:
            result["user_context"] = user_context

        return result

    def _filter_listings(self, query: str, limit: int) -> List[Dict]:
        """
        Filter apartment listings based on the query.
        This is a simple implementation - you might want to enhance this with better search logic.
        """
        query = query.lower()
        filtered = []
        
        for listing in self.apartment_listings:
            score = 0
            listing_str = json.dumps(listing).lower()
            
            # Simple scoring based on query terms
            for term in query.split():
                if term in listing_str:
                    score += 1
                    
            if score > 0:
                filtered.append((score, listing))
                
        # Sort by score and return top results
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [listing for score, listing in filtered[:limit]]

    def add_url_data(self, url: str):
        """Add data from a URL to the existing dataset."""
        try:
            new_data = self.data_loader.load_from_url(url)
            new_listings = self._process_apartment_data(new_data)
            self.apartment_listings.extend(new_listings)
            self._log(f"Successfully added {len(new_listings)} listings from URL", Colors.HEADER)
        except Exception as e:
            self._log(f"Error loading data from URL: {str(e)}", Colors.HEADER)
            raise

def main():
    # Initialize the combined system
    system = RAGBrainEdgeSystem(verbose=True)
    
    # Display available user profiles
    print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Available User Profiles:')}")
    for i, profile in enumerate(system.user_profiles, 1):
        print(f"{i}. {profile['name']} ({profile['email']})")
        print(f"   Interests: {', '.join(profile['interests'])}")
        print(f"   Skills: {', '.join(profile['skills'])}\n")
    
    # Get user selection
    while True:
        try:
            profile_idx = int(input("\nSelect a user profile (enter number): ")) - 1
            if 0 <= profile_idx < len(system.user_profiles):
                selected_profile = system.user_profiles[profile_idx]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    profile_name = selected_profile['name']
    print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Selected Profile: ' + profile_name)}")
    
    while True:
        # Get user query
        print("\nExample questions you can ask:")
        print("1. Would I like the apartment at 2 Townsend St?")
        print("2. Find apartments that match my interests")
        print("3. Which apartments are close to tech meetups?")
        print("4. Show pet-friendly apartments near my preferred locations")
        print("\nType 'quit' to exit")
        
        query = input("\nEnter your question: ").strip()
        if query.lower() == 'quit':
            break
            
        print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, '='*50)}")
        print(Colors.wrap(Colors.HEADER + Colors.BOLD, f"Processing Query: {query}"))
        print(Colors.wrap(Colors.HEADER + Colors.BOLD, f"For User: {selected_profile['name']}"))
        print(Colors.wrap(Colors.HEADER + Colors.BOLD, '='*50))

        result = system.process_query_with_context(
            user_query=query,
            user_email=selected_profile['email']
        )

        # Display results
        print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Retrieved Listings:')}")
        for i, listing in enumerate(result["listings"], 1):
            print(f"\n{Colors.wrap(Colors.THINKING, f'Listing {i}:')}")
            print(Colors.wrap(Colors.RESPONSE, json.dumps(listing, indent=2)))

        print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Brain Analysis:')}")
        if "thinking" in result["brain_decisions"]:
            print(Colors.wrap(Colors.BRAIN, result["brain_decisions"]["thinking"]))

        print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Edge Responses:')}")
        for i, response in enumerate(result["edge_responses"], 1):
            edge_color = Colors.EDGE1 if i == 1 else Colors.EDGE2
            print(f"\n{Colors.wrap(edge_color + Colors.BOLD, f'Edge Instance {i}:')}")
            print(Colors.wrap(Colors.OUTPUT, response))
            
        print("\n" + Colors.wrap(Colors.DIVIDER, "-"*50))

if __name__ == "__main__":
    main() 