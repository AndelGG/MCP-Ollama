from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
import asyncio

load_dotenv()
class AgentState(TypedDict):
    """Состояние агента, содержащее последовательность сообщений."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
async def add(a: int, b: int) -> int:
    """Складывает два целых числа и возвращает результат."""
    return a + b


@tool
async def list_files() -> list:
    """Возвращает список файлов в текущей папке."""
    return os.listdir(".")

tools = [add, list_files]
llm = ChatOllama(model="llama3.1").bind_tools(tools)

async def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="Ты моя система. Ответь на мой вопрос исходя из доступных для тебя инструментов"
    )
    messages = [system_prompt] + list(state["messages"])
    response = await llm.ainvoke(messages)
    return {"messages": [response]}

async def should_continue(state: AgentState) -> str:
    """Проверяет, нужно ли продолжить выполнение или закончить."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"

    # Иначе заканчиваем
    return "end"

async def main():
    # Создание графа
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)
    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    # Настройка потока
    graph.add_edge(START, "our_agent")
    graph.add_conditional_edges(
        "our_agent", should_continue, {"continue": "tools", "end": END}
    )
    graph.add_edge("tools", "our_agent")

    # Компиляция и запуск
    app = graph.compile()
    result = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Посчитай общее количество файлов в этой директории и прибавь к этому значению 10"
                )
            ]
        }
    )

if __name__ == "__main__":
    asyncio.run(main())