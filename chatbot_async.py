from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_graph.prebuilt import ToolNode, tools_condition  # Note: standard import is langgraph.prebuilt
from langchain_core.tools import tool
import asyncio

# Load environment variables (like OPENAI_API_KEY) from a local .env file
load_dotenv()  

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-5")

# Define a custom tool using the @tool decorator for arithmetic operations
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# Register the tool and bind it to the LLM so it knows it can call it
tools = [calculator]
llm_with_tools = llm.bind_tools(tools)

# Define the state structure for the graph, using add_messages to append new messages to history
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    """Builds and compiles the LangGraph workflow."""

    # Node: Handles conversation and decides whether to invoke a tool or respond
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    # Prebuilt node to execute the tools requested by the LLM
    tool_node = ToolNode(tools)

    # Initialize the StateGraph with our ChatState schema
    graph = StateGraph(ChatState)

    # Add nodes to the graph
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Define graph routing/connections
    # 1. Start execution at the chat_node
    graph.add_edge(START, "chat_node")
    
    # 2. Use tools_condition to check if the LLM wants to call a tool; if yes, route to "tools", else finish
    graph.add_conditional_edges("chat_node", tools_condition)
    
    # 3. After the tool executes, feed the result back to the chat_node
    graph.add_edge("tools", "chat_node")

    # Compile the graph into a runnable application
    chatbot = graph.compile()
    return chatbot


async def main():
    # Build the compiled graph
    chatbot = build_graph()

    # Invoke the graph asynchronously with a prompt requesting a specific persona and task
    result = await chatbot.ainvoke({
        "messages": [
            HumanMessage(content="Find the modulus of 132354 and 23 and give answer like a cricket commentator.")
        ]
    })

    # Print the final output message from the LLM
    print(result['messages'][-1].content)


if __name__ == '__main__':
    # Run the asynchronous main function
    asyncio.run(main())