from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

search_tool = DuckDuckGoSearchRun()

def web_search(query):
    return search_tool.run(query)

search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Searches the web for relevant information."
)
