from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_vision_tool.tools.OllamaLLavaTool import OllamaLLavaTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class CrewaiVisionTool():
    """CrewaiVisionTool crew"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def image_information_extracter(self) -> Agent:
        # Now using the patched Ollama configuration
        llm = LLM(
            model="ollama/llama3:latest",
            api_base="http://localhost:11434"
        )
        return Agent(
            config=self.agents_config['image_information_extracter'], # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=[OllamaLLavaTool()]
        )

    @agent
    def mermaid_diagram_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['mermaid_diagram_creator'], # type: ignore[index]
            verbose=True,
            llm=LLM(
                model="ollama/codellama:latest",
                api_base="http://localhost:11434"
            )
        )
    
    @agent
    def mermaid_diagram_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config['mermaid_diagram_verifier'], # type: ignore[index]
            verbose=True,
            llm=LLM(
                model="ollama/qwen3:latest",
                api_base="http://localhost:11434"
            )
        )


    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def info_extractor_task(self) -> Task:
        return Task(
            config=self.tasks_config['info_extractor_task'], # type: ignore[index]
            output_file = "analyzed.txt"
        )

    @task
    def mermaid_code_creator_task(self) -> Task:
        return Task(
            config=self.tasks_config['mermaid_code_creator_task'], # type: ignore[index]
            output_file = "diagram.mmd"
        )
    
    @task
    def mermaid_code_verifier_task(self) -> Task:
        return Task(
            config=self.tasks_config['mermaid_code_verifier_task'], # type: ignore[index]
            output_file = "verification.txt"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiVisionTool crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            planning=False,  # Disable planning to avoid OpenAI dependency (since I don't have any tokens left)
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
