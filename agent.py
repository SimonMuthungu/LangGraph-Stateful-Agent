from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load LLM
llm = OpenAI(temperature=0)

# Tool: Search
search = SerpAPIWrapper()

# Define tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events or facts"
    )
]

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent something
query = input("Ask something: ")
response = agent.run(query)
print("\nAgent Response:", response)
