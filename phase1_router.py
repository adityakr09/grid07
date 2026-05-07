"""
Phase 1: Vector-Based Persona Matching (The Router)
Uses ChromaDB + sentence-transformers (FREE, local) for embeddings
and Groq for LLM calls.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Tuple

BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}


def build_persona_store() -> chromadb.Collection:
    """
    Creates an in-memory ChromaDB collection with bot persona embeddings.
    Uses sentence-transformers (all-MiniLM-L6-v2) — completely free, runs locally.
    """
    client = chromadb.Client()

    # Free local embeddings — no API key needed
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection("bot_personas")
    except Exception:
        pass

    collection = client.create_collection(
        name="bot_personas",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=list(BOT_PERSONAS.keys()),
        documents=[p["description"] for p in BOT_PERSONAS.values()],
        metadatas=[{"name": p["name"]} for p in BOT_PERSONAS.values()],
    )

    print("[Phase 1] Persona store built. Bots indexed:")
    for bot_id, persona in BOT_PERSONAS.items():
        print(f"  • {bot_id} → {persona['name']}")

    return collection


def route_post_to_bots(
    post_content: str,
    collection: chromadb.Collection,
    threshold: float = 0.75,
) -> List[Tuple[str, str, float]]:
    """
    Embeds the post and finds matching bots via cosine similarity.
    threshold = max cosine distance (0.35 ≈ 0.65 similarity).
    """
    results = collection.query(
        query_texts=[post_content],
        n_results=len(BOT_PERSONAS),
        include=["distances", "metadatas"],
    )

    matched = []
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids       = results["ids"][0]

    print(f"\n[Phase 1] Routing post: \"{post_content}\"")
    print(f"{'Bot':<10} {'Name':<22} {'Distance':>10} {'Similarity':>12} {'Match?':>8}")
    print("─" * 68)

    for bot_id, meta, dist in zip(ids, metadatas, distances):
        similarity = 1.0 - dist
        is_match   = dist <= threshold
        marker     = "✅" if is_match else "❌"
        print(f"{bot_id:<10} {meta['name']:<22} {dist:>10.4f} {similarity:>11.4f}  {marker}")
        if is_match:
            matched.append((bot_id, meta["name"], similarity))

    if matched:
        print(f"\n→ Matched {len(matched)} bot(s): {[m[0] for m in matched]}")
    else:
        print("\n→ No bots matched this post.")

    return matched


if __name__ == "__main__":
    store = build_persona_store()
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high as ETFs flood the market with liquidity.",
        "Meta is collecting your data and selling it to hedge funds — wake up.",
    ]
    for post in test_posts:
        route_post_to_bots(post, store)
        print()
