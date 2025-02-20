from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# from crewai_tools import QdrantVectorSearchTool
import os
from dotenv import load_dotenv


load_dotenv()


from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class cogeneeInput(BaseModel):
    """Input schema for cognee search tool"""
    query: str = Field(..., description="Base cognee search query")

class CogneeTool(BaseTool):
    name: str = "cognee"
    description: str = "Searches AGENT MEMORY USING GRAPHS AND VECTOR STORES"
    args_schema: Type[BaseModel] = cogeneeInput

    async def _run(self, query: str) -> str:
        import cognee
        result = await cognee.search(query_text=query)
        return str(result)


@CrewBase
class AnalyzingApartmentHoodCrew:
    """AnalyzingApartmentHoodCrew crew"""

    # vector_search_tool = QdrantVectorSearchTool(
    #     collection_name="contracts_business_5",
    #     qdrant_url=os.getenv("QDRANT_URL"),
    #     qdrant_api_key=os.getenv("QDRANT_API_KEY"),

    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    # )
    cognee_tool = CogneeTool()

    @agent
    def manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["manager_agent"],
            tools=[self.search_tool, self.web_rag_tool, self.cognee_tool],
            allow_delegation=True
        )

    @agent
    def location_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["location_analysis_specialist"],
            tools=[self.search_tool, self.web_rag_tool, self.cognee_tool],
        )

    @agent
    def home_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["home_analysis_specialist"],
            tools=[self.search_tool, self.web_rag_tool, self.cognee_tool],
        )


    # @task
    # def manager_investigation_task(self, inputs: dict) -> dict:
    #     """
    #     Simple example task that uses the manager_agent.
    #     This task must return a dict so the crew can produce an output.
    #     """
    #     query = inputs.get("query", "N/A")
    #
    #     # Here is where you'd call the agent with the query
    #     # For a trivial example, we do a plain LLM call:
    #     response = self.manager_agent().run(f"Analyze this query: {query}")
    #
    #     # Return the task output as a dict
    #     return {"analysis": response}

    @task
    def home_analysis_specialist_task(self) -> Task:
        return Task(
            config=self.tasks_config["home_analysis_specialist_task"],
        )

    @task
    def location_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["location_analysis_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalyzingApartmentHoodCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )



def run():
    """
    Run the crew.
    """
    inputs = {
        "query": "What are the the apartments i should buy",
    }
    AnalyzingApartmentHoodCrew().crew().kickoff(
        inputs=inputs
    )

if __name__ == "__main__":
    run()
