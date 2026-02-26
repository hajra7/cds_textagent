"""
test_agents.py — Test the internet search agent in isolation
Run: python test_agents.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents import internet_search_agent


def test_internet_search_agent(query: str):
    print(f"\n{'='*60}")
    print(f"  Agent Query: {query}")
    print(f"{'='*60}")

    result = internet_search_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Print each message step so you can see tool calls + responses
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n[Human]   {msg.content[:200]}")

        elif isinstance(msg, AIMessage):
            # Show tool calls if any
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[AI → Tool Call]  {tc['name']}({tc['args']})")
            # Show final text response
            if msg.content:
                print(f"\n[AI Final]\n{msg.content}")

        elif isinstance(msg, ToolMessage):
            # Truncate long tool outputs for readability
            preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            print(f"\n[Tool Result ← {msg.name}]\n{preview}")


if __name__ == "__main__":
    print("\n>>> Testing Internet Search Agent\n")

    test_internet_search_agent(
        "What is the treatment for glioblastoma for elderly patients?"
    )

    print("\n>>> Agent test done.\n")