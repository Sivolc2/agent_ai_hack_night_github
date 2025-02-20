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


@CrewBase
class AnalyzingContractClausesForConflictsAndSimilaritiesCrew:
    """AnalyzingContractClausesForConflictsAndSimilarities crew"""

    # vector_search_tool = QdrantVectorSearchTool(
    #     collection_name="contracts_business_5",
    #     qdrant_url=os.getenv("QDRANT_URL"),
    #     qdrant_api_key=os.getenv("QDRANT_API_KEY"),

    search_tool = SerperDevTool()
    web_rag_tool = WebsiteSearchTool()
    # )

    @agent
    def manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["manager_agent"],
            tools=[self.search_tool, self.web_rag_tool],
            allow_delegation=True
        )

    @agent
    def location_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["location_analysis_specialist"],
            tools=[self.search_tool, self.web_rag_tool],
        )

    @agent
    def home_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["home_analysis_specialist"],
            tools=[self.search_tool, self.web_rag_tool],
        )




    @crew
    def crew(self) -> Crew:
        """Creates the AnalyzingContractClausesForConflictsAndSimilarities crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
