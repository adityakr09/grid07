"""
Phase 2: The Autonomous Content Engine (LangGraph)
Uses Groq (llama3-70b) as LLM — free tier.
"""

import os
import json
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Groq LLM ─────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9,
    api_key=os.environ["GROQ_API_KEY"],
)

# ── Mock SearXNG Search Tool ──────────────────────────────────────────
MOCK_NEWS_DB = {
    "crypto":      "Bitcoin hits new all-time high of $105K amid regulatory ETF approvals from SEC.",
    "bitcoin":     "MicroStrategy doubles down — buys another 10,000 BTC as institutional demand surges.",
    "ai":          "OpenAI's GPT-5 reportedly passes the bar exam and beats radiologists at diagnosis.",
    "openai":      "OpenAI valued at $300B after latest funding round; Altman hints at AGI by 2026.",
    "elon":        "Elon Musk's xAI launches Grok-3, claims it outperforms GPT-4 on every benchmark.",
    "space":       "SpaceX Starship successfully completes first crewed lunar flyby mission.",
    "regulation":  "EU AI Act enforcement begins — tech giants face €30M fines for non-compliance.",
    "privacy":     "Meta fined €1.2B for GDPR violations; whistleblower leaks internal surveillance docs.",
    "social media":"TikTok algorithm exposed: internal docs show deliberate teen addiction design.",
    "billionaire": "Top 10 billionaires wealth grew by $500B in 2024 while real wages stagnated.",
    "market":      "S&P 500 breaks 6,000 for the first time as Fed signals two rate cuts in 2025.",
    "fed":         "Federal Reserve holds rates at 4.5%; bond traders price in 60bps of cuts by Q3.",
    "trading":     "Quant funds post 34% returns in 2024; momentum strategies dominate discretionary.",
    "rates":       "10-year Treasury yield spikes to 5.1% — mortgage rates hit 7.8%, housing stalls.",
    "default":     "Breaking: Tech stocks surge as AI infrastructure spending hits record levels.",
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search. Returns a hardcoded recent headline
    based on keywords found in the query.
    """
    q = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in q:
            return headline
    return MOCK_NEWS_DB["default"]


# ── LangGraph State ───────────────────────────────────────────────────
class PostState(TypedDict):
    bot_id:        str
    persona:       str
    search_query:  Optional[str]
    search_result: Optional[str]
    topic:         Optional[str]
    post_content:  Optional[str]
    final_output:  Optional[dict]


# ── Node 1: Decide Search ─────────────────────────────────────────────
def node_decide_search(state: PostState) -> PostState:
    print(f"\n[Node 1 — decide_search] Bot: {state['bot_id']}")

    messages = [
        SystemMessage(content=(
            f"You are {state['bot_id']} with this persona: {state['persona']}\n\n"
            "Decide ONE topic to post about today that fits your worldview. "
            "Then produce a search query (3-5 words) to find a recent news headline.\n\n"
            "Respond in this EXACT JSON format (no markdown, no extra text):\n"
            '{"topic": "...", "search_query": "..."}'
        )),
        HumanMessage(content="What do you want to post about today?"),
    ]

    response = llm.invoke(messages)
    raw = re.sub(r"```(?:json)?", "", response.content.strip()).strip("` \n")
    parsed = json.loads(raw)

    print(f"  Topic decided: {parsed['topic']}")
    print(f"  Search query:  {parsed['search_query']}")
    return {**state, "topic": parsed["topic"], "search_query": parsed["search_query"]}


# ── Node 2: Web Search ────────────────────────────────────────────────
def node_web_search(state: PostState) -> PostState:
    print(f"\n[Node 2 — web_search] Query: \"{state['search_query']}\"")
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  Result: {result}")
    return {**state, "search_result": result}


# ── Node 3: Draft Post ────────────────────────────────────────────────
def node_draft_post(state: PostState) -> PostState:
    print(f"\n[Node 3 — draft_post] Generating post for {state['bot_id']}...")

    messages = [
        SystemMessage(content=(
            f"You are {state['bot_id']} with this persona: {state['persona']}\n\n"
            f"News headline: {state['search_result']}\n\n"
            "Write a social media post (MAX 280 characters) that:\n"
            "1. Is highly opinionated and matches your persona.\n"
            "2. Incorporates the headline as context.\n"
            "3. Sounds authentic, not corporate.\n\n"
            "Respond ONLY with this JSON (no markdown, no extra text):\n"
            '{"bot_id": "...", "topic": "...", "post_content": "..."}\n\n'
            "CRITICAL: post_content must be 280 characters or less."
        )),
        HumanMessage(content=f"Topic: {state['topic']}\nWrite the post now."),
    ]

    response = llm.invoke(messages)
    raw = re.sub(r"```(?:json)?", "", response.content.strip()).strip("` \n")
    parsed = json.loads(raw)
    parsed["bot_id"] = state["bot_id"]
    parsed["topic"]  = state["topic"]

    print(f"  ✅ Post ({len(parsed['post_content'])} chars): {parsed['post_content']}")
    return {**state, "post_content": parsed["post_content"], "final_output": parsed}


# ── Build Graph ───────────────────────────────────────────────────────
def build_content_graph():
    graph = StateGraph(PostState)
    graph.add_node("decide_search", node_decide_search)
    graph.add_node("web_search",    node_web_search)
    graph.add_node("draft_post",    node_draft_post)
    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)
    return graph.compile()


def run_content_engine(bot_id: str, persona: str) -> dict:
    app = build_content_graph()
    initial_state: PostState = {
        "bot_id": bot_id, "persona": persona,
        "search_query": None, "search_result": None,
        "topic": None, "post_content": None, "final_output": None,
    }
    result = app.invoke(initial_state)
    return result["final_output"]


if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS
    for bot_id, persona_data in BOT_PERSONAS.items():
        print("\n" + "═" * 60)
        output = run_content_engine(bot_id, persona_data["description"])
        print(f"\n📦 Final JSON:\n{json.dumps(output, indent=2)}")
