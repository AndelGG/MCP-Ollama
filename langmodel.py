from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")

llm.invoke("tell me a joke about bears")