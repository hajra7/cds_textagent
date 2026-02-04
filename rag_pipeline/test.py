import sys
from pathlib import Path

# Add parent directory to path so we can import config and other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import torch
from sentence_transformers import CrossEncoder
from config import PROCESSED_DATA_DIR




def format_docs_with_scores(docs_and_scores):
    """Format documents with their similarity scores for the prompt"""
    formatted = []
    for i, (doc, similarity, rerank_score) in enumerate(docs_and_scores, 1):
        formatted.append(
            f"[Doc {i} | EmbeddingSim: {similarity:.4f} ({similarity*100:.1f}%) | "
            f"RerankScore: {rerank_score:.4f}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def rerank_with_cross_encoder(cross_encoder, question: str, docs_with_similarity, top_k: int = 5):
    """
    docs_with_similarity: List[(Document, similarity_float_0_to_1)]
    Returns: List[(Document, similarity, rerank_score)]
    """
    if not docs_with_similarity:
        return []

    pairs = [(question, doc.page_content[:1200]) for doc, _sim in docs_with_similarity]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for (doc, sim), ce_score in zip(docs_with_similarity, scores):
        reranked.append((doc, sim, float(ce_score)))

    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked[:top_k]


def retrieve_rerank_and_format_docs(
    vectorstore,
    cross_encoder,
    question,
    retrieve_k=30,
    final_k=5,
    min_similarity=None,
    verbose=True,
):
    """
    1) Retrieve top_k via embeddings (Chroma)
    2) Convert distance->similarity if cosine
    3) (optional) filter by min_similarity
    4) Cross-encoder rerank
    5) Return formatted context + structured sources
    """
    if verbose:
        print(f"\nRetrieving top {retrieve_k} via embeddings, then reranking to top {final_k}...")

    collection = vectorstore._collection
    space = (collection.metadata or {}).get("hnsw:space", None)

    docs_with_scores = vectorstore.similarity_search_with_score(question, k=retrieve_k)

    def to_similarity(score):
        if space == "cosine":
            sim = 1.0 - float(score)
        else:
            sim = float(score)
        return max(0.0, min(1.0, sim))

    docs_with_similarity = [(doc, to_similarity(score)) for doc, score in docs_with_scores]
    docs_with_similarity.sort(key=lambda x: x[1], reverse=True)

    if min_similarity is not None:
        docs_with_similarity = [(d, s) for d, s in docs_with_similarity if s >= min_similarity]
        if verbose:
            print(f"After min_similarity filter: {len(docs_with_similarity)} docs remain")

    if not docs_with_similarity:
        return "No relevant context found.", []

    top_reranked = rerank_with_cross_encoder(
        cross_encoder, question, docs_with_similarity, top_k=final_k
    )

    formatted_chunks = []
    for i, (doc, sim, ce) in enumerate(top_reranked, 1):
        formatted_chunks.append(
            f"[Doc {i} | EmbeddingSim: {sim:.4f} ({sim*100:.1f}%) | RerankScore: {ce:.4f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks), top_reranked


def test():
    # Load the existing ChromaDB vectorstore
    persist_dir = PROCESSED_DATA_DIR / "chroma_openai_text_embedding_3_small"
    
    print(f"Loading ChromaDB from: {persist_dir}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="cases_openai",
    )
    
    # Check collection info
    collection = vectorstore._collection
    space = (collection.metadata or {}).get("hnsw:space", None)
    print("Chroma space:", space)

    count = collection.count()
    print(f"Collection '{collection.name}' has {count} documents\n")
    
    # Load llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Cross-encoder reranker (fast, good baseline)
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(
        reranker_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Loaded reranker: {reranker_model}")
    
    # Set up chat history
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Create chat prompt template with memory
    system_template = """You are a clinical decision-support assistant for radiology.

Answer at a clinical-professional level using ONLY the provided context and conversation history.

Guidelines for answering:
- Prioritize clinical relevance over general explanations
- Use rerank scores as the primary relevance signal; embedding similarity is only a coarse filter
- Avoid oversimplifications and absolute statements
- Distinguish clearly between:
  • newly diagnosed vs recurrent disease
  • fit vs poor performance status
  • curative vs palliative intent
- When age, performance status, or comorbidities are relevant, explicitly factor them into recommendations
- Do NOT recommend aggressive treatment when performance status is poor unless clearly supported by context
- Clearly label treatments that are palliative rather than disease-modifying
- If evidence is limited or supportive rather than definitive, state this explicitly
- Do not introduce facts, treatments, or guidelines not supported by the provided context

When answering treatment questions:
- Present options in order of clinical appropriateness
- Emphasize quality-of-life considerations when relevant
- Support recommendations with reasoning grounded in the retrieved context


Context:
{context}"""
    
    # Create prompt with memory using MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Create RAG chain
    rag_chain = prompt | llm | StrOutputParser()
    
    # Wrap with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # Session ID for this conversation
    session_id = "user_session_1"
    
    print("\n" + "="*70)
    print(" Medical Assistant RAG Chatbot with Memory")
    print("="*70)
    print("Ask questions about medical cases and diagnoses.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("="*70 + "\n")
    
    # Chat loop
    while True:
        # Get user input
        try:
            question = input("\n You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Stay healthy! ")
            break
        
        if not question:
            continue
            
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye! Stay healthy! ")
            break
        
        # Retrieve + rerank documents
        context_with_scores, reranked_docs = retrieve_rerank_and_format_docs(
            vectorstore,
            cross_encoder,
            question,
            retrieve_k=30,
            final_k=5,
            min_similarity=None,
            verbose=False,
        )
        
        # Get response with context and history
        try:
            response = conversational_rag_chain.invoke(
                {
                    "context": context_with_scores,
                    "input": question
                },
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"\n Assistant: {response}")
            
            # Display sources with similarity, rerank scores, and summary
            print("\nSources:")
            for i, (doc, similarity) in enumerate(reranked_docs, 1):
                case_id = doc.metadata.get("case_id", "N/A")
                article_id = doc.metadata.get("article_id", "N/A")
                doc_type = doc.metadata.get("doc_type", "N/A")
                region_granular = doc.metadata.get("radiology_region_granular", "N/A")
                summary = " ".join(doc.page_content.split())[:240]
                print(
                    f"  {i}. Case ID: {case_id} | Article: {article_id} | Type: {doc_type} "
                    f"| Region: {region_granular} "
                    f"| Similarity: {similarity:.4f} ({similarity*100:.1f}%) "
                )
                print(f"     Summary: {summary}...")
            
            print("\n" + "-"*70)
        except Exception as e:
            print(f"\n Error: {str(e)}")
            print("-"*70)



if __name__ == "__main__":
    test()
