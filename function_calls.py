import json
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

def arithmetic_function_calling(call):
    try:
        function_name = call['name']
        arguments = json.loads(call['arguments'])

        if function_name == "add":
            return arguments['a'] + arguments['b']
        elif function_name == "subtract":
            return arguments['a'] - arguments['b']
        elif function_name == "multiply":
            return arguments['a'] * arguments['b']
        elif function_name == "divide":
            return arguments['a'] / arguments['b'] if arguments['b'] != 0 else "Error: Division by zero"
        else:
            return "Unknown function"
    except Exception as e:
        return f"Error: {e}"

def generate_summary(text):
    llm = ChatOpenAI(model_name="gpt-4")
    return llm.predict(f"Summarize this content: {text}")

summary_tool = Tool(
    name="generate_summary",
    func=generate_summary,
    description="Generates a concise summary of the provided text."
)
