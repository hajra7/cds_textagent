from langchain_core.documents.base import Document


import sys
from pathlib import Path

# Add parent directory to path so we can import config and other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate

from config import PROCESSED_DATA_DIR


def test():
    # Load the existing ChromaDB vectorstore
    persist_dir = PROCESSED_DATA_DIR / "chroma_openai_text_embedding_3_small"
    
    print(f"Looking for ChromaDB at: {persist_dir}")
    print(f"Directory exists: {persist_dir.exists()}")
    
    if not persist_dir.exists():
        print("ERROR: ChromaDB directory not found!")
        print(f"Expected location: {persist_dir}")
        return
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="cases_openai",
    )
    
    # Check collection info
    collection = vectorstore._collection
    count = collection.count()
    print(f"\nCollection '{collection.name}' has {count} documents")
    
    if count == 0:
        print("ERROR: Collection is empty! You need to run chroma_with_openai.py first to build the index.")
        return
    
    # Create retriever   
    retriever = vectorstore.as_retriever()
    
    # RAG chain
    
    # Load llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    template = """
You are a helpful assistant that answers questions based on the provided context.
Use the provided context to answer the question.
Question: {input}
Context: {context}
Answer:
"""
    prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    response = rag_chain.invoke("What is the treatment for dyspnea?")
    print(f"\n=== RAG Response ===")
    print(response)



if __name__ == "__main__":
    test()
