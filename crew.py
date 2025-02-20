from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, crew
from crewai_tools import SerperDevTool
from fireworks_llm import FireworksLLM
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Fireworks models for different roles
researcher_llm = FireworksLLM(
    model_id="accounts/fireworks/models/deepseek-v3",  # Using DeepSeek for research
    temperature=0.7
)

analyst_llm = FireworksLLM(
    model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",  # Using Llama for analysis
    temperature=0.6
)

@CrewBase
class LatestAiDevelopmentCrew():
    """LatestAiDevelopment crew"""

    agents_config = "config/agents.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            llm=researcher_llm
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm=analyst_llm
        )

    @crew
    def crew(self) -> Crew:
        """Create the crew for AI development research"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

def main():
    # Create an instance of the crew
    ai_crew = LatestAiDevelopmentCrew()
    
    # Get the crew
    crew_instance = ai_crew.crew()
    
    # Kick off the crew's work
    result = crew_instance.kickoff()
    print("\nCrew Work Result:")
    print(result)

if __name__ == "__main__":
    main() 