from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from function_calls import summary_tool
from web_search import search_tool

def build_agent(vector_store):
    tools = [summary_tool, search_tool]
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent
