from test_fireworks import FireworksAPI
from typing import List, Dict, Optional
import json
import re
from datetime import datetime

# ANSI color codes
class Colors:
    HEADER = '\033[95m'      # Purple
    BRAIN = '\033[94m'       # Blue
    EDGE1 = '\033[92m'       # Green
    EDGE2 = '\033[93m'       # Yellow
    THINKING = '\033[96m'    # Cyan
    RESPONSE = '\033[97m'    # White
    DIVIDER = '\033[90m'     # Gray
    ENDC = '\033[0m'         # Reset
    BOLD = '\033[1m'         # Bold
    
    @staticmethod
    def wrap(color: str, text: str) -> str:
        return f"{color}{text}{Colors.ENDC}"

class BrainEdgeSystem:
    def __init__(self, verbose: bool = False):
        self.api = FireworksAPI()
        self.brain = self.api.models["deepseek"]  # R1 model
        self.edge_instances = [
            self.api.models["deepseek-v3"],  # First V3 instance
            self.api.models["deepseek-v3"]    # Second V3 instance
        ]
        self.verbose = verbose
        self.thought_log = []
        
    def log_thought(self, agent: str, thought: str):
        """
        Log a thought from an agent if verbose mode is enabled
        """
        if self.verbose:
            timestamp = datetime.now().isoformat()
            thought_entry = {
                "timestamp": timestamp,
                "agent": agent,
                "thought": thought
            }
            self.thought_log.append(thought_entry)
            
            # Color-coded agent names and thoughts
            if "Brain" in agent:
                agent_color = Colors.BRAIN
            elif "Edge1" in agent:
                agent_color = Colors.EDGE1
            else:
                agent_color = Colors.EDGE2
                
            colored_timestamp = Colors.wrap(Colors.DIVIDER, f"[{timestamp}]")
            colored_agent = Colors.wrap(agent_color + Colors.BOLD, agent)
            colored_thinking = Colors.wrap(Colors.THINKING, "thinking: ")
            colored_thought = Colors.wrap(Colors.RESPONSE, thought)
            
            print(f"\n{colored_timestamp} {colored_agent} {colored_thinking}{colored_thought}")
        
    def brain_decide(self, situation: str) -> str:
        """
        Use R1 to analyze situation and decide what commands to send to V3 instances
        """
        self.log_thought("Brain (R1)", f"Analyzing situation: {situation}")
        
        prompt = f"""Given this situation: {situation}

        Analyze the situation and provide two separate commands for our edge instances to execute.
        Use HTML-style tags to structure your response as follows:

        <thinking>
        Share your step-by-step thought process here about how you're approaching this task
        </thinking>

        <reasoning>
        Explain your final decision-making process here
        </reasoning>

        <edge1>
        Write the specific command for the first edge instance here
        </edge1>

        <edge2>
        Write the specific command for the second edge instance here
        </edge2>

        Make sure each command is clear, specific, and self-contained within its tags.
        Do not include any additional tags or thinking process in the edge commands.
        """
        
        response = self.api.get_completion(self.brain, prompt) or ""
        self.log_thought("Brain (R1)", "Generated response with commands for edge instances")
        return response
        
    def parse_brain_response(self, response: str) -> Dict:
        """
        Parse the brain's HTML-tagged response into a structured format
        """
        commands = {}
        
        # Extract content between tags using regex
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        edge1_match = re.search(r'<edge1>(.*?)</edge1>', response, re.DOTALL)
        edge2_match = re.search(r'<edge2>(.*?)</edge2>', response, re.DOTALL)
        
        if thinking_match:
            commands["thinking"] = thinking_match.group(1).strip()
            self.log_thought("Brain (R1)", f"Thought process: {commands['thinking']}")
            
        if reasoning_match:
            commands["reasoning"] = reasoning_match.group(1).strip()
        if edge1_match:
            commands["edge1_command"] = edge1_match.group(1).strip()
        if edge2_match:
            commands["edge2_command"] = edge2_match.group(1).strip()
            
        return commands
        
    def execute_edge_commands(self, commands: Dict) -> List[str]:
        """
        Send commands to V3 edge instances and get their responses
        """
        responses = []
        
        if "edge1_command" in commands:
            self.log_thought("Edge1 (V3)", f"Executing command: {commands['edge1_command']}")
            edge1_prompt = f"""You are a V3 edge instance. Execute this command directly:

            {commands['edge1_command']}

            Before providing your response, briefly explain your approach:
            <thinking>Your approach</thinking>

            Then give your response:
            <response>Your actual output</response>
            """
            response1 = self.api.get_completion(self.edge_instances[0], edge1_prompt)
            
            # Parse edge1 thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response1, re.DOTALL)
            response_match = re.search(r'<response>(.*?)</response>', response1, re.DOTALL)
            
            if thinking_match:
                self.log_thought("Edge1 (V3)", f"Approach: {thinking_match.group(1).strip()}")
            responses.append(response_match.group(1).strip() if response_match else response1)
            
        if "edge2_command" in commands:
            self.log_thought("Edge2 (V3)", f"Executing command: {commands['edge2_command']}")
            edge2_prompt = f"""You are a V3 edge instance. Execute this command directly:

            {commands['edge2_command']}

            Before providing your response, briefly explain your approach:
            <thinking>Your approach</thinking>

            Then give your response:
            <response>Your actual output</response>
            """
            response2 = self.api.get_completion(self.edge_instances[1], edge2_prompt)
            
            # Parse edge2 thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response2, re.DOTALL)
            response_match = re.search(r'<response>(.*?)</response>', response2, re.DOTALL)
            
            if thinking_match:
                self.log_thought("Edge2 (V3)", f"Approach: {thinking_match.group(1).strip()}")
            responses.append(response_match.group(1).strip() if response_match else response2)
            
        return responses
    
    def process_situation(self, situation: str) -> Dict:
        """
        Main method to process a situation using the brain-edge system
        """
        # Brain (R1) decides what to do
        brain_response = self.brain_decide(situation)
        
        # Parse the brain's response
        decisions = self.parse_brain_response(brain_response)
        
        # Execute commands on edge instances (V3)
        edge_responses = self.execute_edge_commands(decisions)
        
        result = {
            "brain_decisions": decisions,
            "edge_responses": edge_responses,
            "raw_brain_response": brain_response
        }
        
        if self.verbose:
            result["thought_log"] = self.thought_log
            
        return result

def main():
    system = BrainEdgeSystem(verbose=True)
    
    # Example usage
    test_situation = "We need to write a haiku on 2 different topics"
    result = system.process_situation(test_situation)
    
    # Color-coded section headers and content
    print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Brain (R1) Raw Response:')}")
    print(Colors.wrap(Colors.DIVIDER, "-" * 50))
    print(Colors.wrap(Colors.BRAIN, result["raw_brain_response"]))
    print(Colors.wrap(Colors.DIVIDER, "-" * 50))
    
    print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Parsed Brain Decisions:')}")
    parsed_decisions = json.dumps(result["brain_decisions"], indent=2)
    print(Colors.wrap(Colors.BRAIN, parsed_decisions))
    
    print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Edge (V3) Responses:')}")
    for i, response in enumerate(result["edge_responses"], 1):
        edge_color = Colors.EDGE1 if i == 1 else Colors.EDGE2
        print(f"\n{Colors.wrap(edge_color + Colors.BOLD, f'Edge Instance {i}:')}")
        print(Colors.wrap(edge_color, response))
        
    if "thought_log" in result:
        print(f"\n{Colors.wrap(Colors.HEADER + Colors.BOLD, 'Complete Thought Log:')}")
        print(Colors.wrap(Colors.DIVIDER, "-" * 50))
        for entry in result["thought_log"]:
            # Color-code based on agent
            if "Brain" in entry["agent"]:
                agent_color = Colors.BRAIN
            elif "Edge1" in entry["agent"]:
                agent_color = Colors.EDGE1
            else:
                agent_color = Colors.EDGE2
                
            timestamp = Colors.wrap(Colors.DIVIDER, f"[{entry['timestamp']}]")
            agent = Colors.wrap(agent_color + Colors.BOLD, entry["agent"])
            thought = Colors.wrap(Colors.RESPONSE, entry["thought"])
            
            print(f"{timestamp} {agent}:")
            print(f"  {thought}\n")

if __name__ == "__main__":
    main() 