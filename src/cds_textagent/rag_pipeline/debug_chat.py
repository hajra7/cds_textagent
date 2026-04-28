from langchain_mistralai import ChatMistralAI
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from cds_textagent.rag_pipeline.retriever import retrieve_rerank_and_format_docs


def test():
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    system_template = """You are a clinical decision-support assistant for radiology.

Answer at a clinical-professional level using ONLY the provided context and conversation history.

Guidelines for answering:
- Prioritize clinical relevance over general explanations
- Use rerank scores as the primary relevance signal; embedding similarity is only a coarse filter
- Avoid oversimplifications and absolute statements
- Distinguish clearly between:
  \u2022 newly diagnosed vs recurrent disease
  \u2022 fit vs poor performance status
  \u2022 curative vs palliative intent
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

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    rag_chain = prompt | llm | StrOutputParser()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    session_id = "user_session_1"

    print("\n" + "=" * 70)
    print(" Medical Assistant RAG Chatbot with Memory")
    print("=" * 70)
    print("Ask questions about medical cases and diagnoses.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("\n You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Stay healthy!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! Stay healthy!")
            break

        context_with_scores, reranked_docs = retrieve_rerank_and_format_docs(
            question,
            retrieve_k=30,
            final_k=5,
        )

        try:
            response = conversational_rag_chain.invoke(
                {"context": context_with_scores, "input": question},
                config={"configurable": {"session_id": session_id}},
            )

            print(f"\n Assistant: {response}")

            print("\nSources:")
            for i, (doc, similarity, rerank_score) in enumerate(reranked_docs, 1):
                case_id = doc.metadata.get("case_id", "N/A")
                article_id = doc.metadata.get("article_id", "N/A")
                doc_type = doc.metadata.get("doc_type", "N/A")
                region = doc.metadata.get("radiology_region_granular", "N/A")
                summary = " ".join(doc.page_content.split())[:240]
                print(
                    f"  {i}. Case: {case_id} | Article: {article_id} | Type: {doc_type} "
                    f"| Region: {region} "
                    f"| Sim: {similarity:.4f} ({similarity*100:.1f}%) "
                    f"| Rerank: {rerank_score:.4f}"
                )
                print(f"     {summary}...")

            print("\n" + "-" * 70)
        except Exception as e:
            print(f"\n Error: {e}")
            print("-" * 70)


if __name__ == "__main__":
    test()
