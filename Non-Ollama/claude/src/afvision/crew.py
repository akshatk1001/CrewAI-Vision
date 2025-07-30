from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import streamlit as st
import json
from langchain_core.agents import AgentFinish
from afvision.tools.custom_tool import ImageTool  # Assuming custom_tool.py is in the same directory
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Afvision():
    """Afvision crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    agents: List[BaseAgent]
    tasks: List[Task]

    def step_callback_crewai(self, agent_output, agent_name, *args):
        with st.chat_message("AI"):
            if isinstance(agent_output, str):
                try:
                    agent_output = json.loads(agent_output)
                except json.JSONDecodeError:
                    pass
            
            if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
                for action, desc in agent_output:
                    st.write(f"Agent Name: {agent_name}")
                    st.write(f"Tool used (if any): {getattr(action, 'tool', 'N/A')}")
                    st.write(f"Tool input:  {getattr(action, 'tool_input', 'N/A')}")
                    st.write(f"{getattr(action, 'log', 'Unknown')}")
                    with st.expander("Show observation"):
                        st.markdown(f"Observation\n\n{desc}")
            
            elif isinstance(agent_output, AgentFinish):
                st.write(f"Agent Name: {agent_name}")
                st.write(f"I've finished my task:\n {agent_output.return_values['output']}")
            
            else:
                st.write(f"Agent Name: {agent_name}")
                st.write(type(agent_output))
                st.write(f"Output: {agent_output}")



    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def diagram_info_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['diagram_info_extractor'], # type: ignore[index]
            max_iter=3,
            step_callback = lambda step: self.step_callback_crewai(step, "Info Extractor Agent"),
        )

    @agent
    def mermaid_code_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['mermaid_code_generator'], # type: ignore[index]
            step_callback = lambda step: self.step_callback_crewai(step, "Mermaid Code Generator Agent")
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['extraction_task'], # type: ignore[index]
            tools=[ImageTool()],  # Registering the custom tool
        )

    @task
    def code_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_generation_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Afvision crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
