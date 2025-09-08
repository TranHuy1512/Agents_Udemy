from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool

from .tools.push_tool import PushNotificationTool
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

import os
from typing import List
from pydantic import BaseModel, Field


class PopularDestination(BaseModel):
    name: str = Field(description="Name of the popular destination")
    reason: str = Field(description="Reason why this destination is popular")


class PopularDestinationList(BaseModel):
    destinations: List[PopularDestination] = Field(
        description="List of popular destinations"
    )


class PopularDestinationAnalysis(BaseModel):
    name: str = Field(description="Name of the popular destination")
    accessibility: str = Field(
        description="Description of how accessible the destination is (e.g., transportation, distance, cost)",
    )
    activities: List[str] = Field(
        description="List of key activities available at the destination (e.g., sightseeing, adventure, food experiences)",
    )
    uniqueness: str = Field(
        description="What makes this destination unique compared to others"
    )
    seasonality: str = Field(
        description="Best season to visit (e.g., peak, off-peak, mixed), and any impact of seasons on accessibility or experience",
    )
    budget: str = Field(
        description="Estimated budget level (e.g., budget, mid-range, luxury)",
    )


class PopularDestinationAnalysisList(BaseModel):
    analysis_list: List[PopularDestinationAnalysis] = Field(
        description="List of destinations analysis"
    )


@CrewBase
class DanangTravel:
    """DanangTravel crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def destination_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["destination_finder"],
            tools=[SerperDevTool()],
            # type: ignore[index]
            verbose=True,
        )

    @agent
    def destination_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["destination_analyst"],
            tools=[SerperDevTool()],
            # type: ignore[index]
            verbose=True,
        )

    @agent
    def destination_picker(self) -> Agent:
        return Agent(
            config=self.agents_config["destination_picker"],
            tools=[PushNotificationTool()],
            # type: ignore[index]
            verbose=True,
        )

    @task
    def find_popular_destinations(self) -> Task:
        return Task(
            config=self.tasks_config["find_popular_destinations"],
            output_pydantic=PopularDestinationList,
            # type: ignore[index]
        )

    @task
    def analyze_popular_destinations(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_popular_destinations"],
            output_pydantic=PopularDestinationAnalysisList,
            # type: ignore[index]
            output_file="report.md",
        )

    @task
    def pick_best_destinations(self) -> Task:
        return Task(
            config=self.tasks_config["pick_best_destinations"],  # type: ignore[index]
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DanangTravel crew"""
        manager = Agent(config=self.agents_config["manager"], allow_delegation=True)

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            # memory=True,
            # long_term_memory=LongTermMemory(
            #     storage=LTMSQLiteStorage(db_path="./memory/long_term_memory_storage.db")
            # ),
            # short_term_memory=ShortTermMemory(
            #     storage=RAGStorage(type="short_term", path="./memory/")
            # ),
            # entity_memory=EntityMemory(storage=RAGStorage(type="short_term", path="./memory/"))
        )
