import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.tools import search_guidelines, search_differential_diagnosis, search_pubmed
from langchain_core.messages import HumanMessage
from  langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [
    search_guidelines,
    search_differential_diagnosis,
    search_pubmed,
]


internet_search_agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,
)

response = internet_search_agent.invoke(
    {"messages": [HumanMessage(content="What is the treatment for glioblastoma for elderly patients?")]},
)

print(response["messages"][-1].content)


# RAG Agent — uncomment when RAG pipeline is ready

# rag_agent = create_react_agent(
#     model=agent_llm,
#     tools=[rag_tool],
# )