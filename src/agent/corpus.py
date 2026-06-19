"""Local, structured crisis precedents for constrained RAG retrieval."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv


class Precedent(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_name: str
    date: str
    volume_signature: str
    volatility_signature: str
    breadth_signature: str
    event_type: str
    operational_outcome: str

    @property
    def identifier(self) -> str:
        return f"{self.event_name} ({self.date})"

    def document(self) -> str:
        return " | ".join([
            f"event_name: {self.event_name}", f"date: {self.date}",
            f"volume_signature: {self.volume_signature}", f"volatility_signature: {self.volatility_signature}",
            f"breadth_signature: {self.breadth_signature}", f"event_type: {self.event_type}",
            f"operational_outcome: {self.operational_outcome}",
        ])


# Concise, public-event precedents. Operational outcomes are written as SRE
# lessons, not assertions that a particular firm suffered a specific outage.
_EVENTS = [
    ("Taper tantrum", "2013-06-20", "elevated volume", "rate-volatility jump", "broad risk-off", "volatility-shock"),
    ("China devaluation selloff", "2015-08-24", "extreme market-wide turnover", "volatility surge", "broad declines", "volume-driven"),
    ("Black Monday 2015", "2015-08-24", "extreme market-wide turnover", "volatility surge", "broad declines", "volume-driven"),
    ("January 2016 selloff", "2016-01-15", "sustained elevated turnover", "high volatility", "broad weakness", "volatility-shock"),
    ("Brexit referendum", "2016-06-24", "event-day volume surge", "overnight volatility repricing", "global risk-off breadth", "overnight-event"),
    ("US election 2016", "2016-11-09", "event-day turnover surge", "overnight futures volatility", "rapid breadth rotation", "overnight-event"),
    ("Volmageddon VIX shock", "2018-02-05", "elevated turnover", "volatility-product dislocation", "broad equity selling", "volatility-shock"),
    ("December 2018 selloff", "2018-12-24", "thin-session stress volume", "elevated volatility", "broad risk-off", "volume-driven"),
    ("COVID market shock", "2020-03-09", "extraordinary turnover", "extreme volatility", "near-universal stress breadth", "volume-driven"),
    ("COVID circuit-breaker week", "2020-03-16", "sustained record turnover", "extreme volatility", "near-universal stress breadth", "volatility-shock"),
    ("GameStop concentration", "2021-01-28", "concentrated retail volume", "single-name volatility", "mixed market breadth", "volume-driven"),
    ("Fed tightening repricing", "2022-06-13", "elevated turnover", "rates-driven volatility", "broad risk-off", "volatility-shock"),
    ("UK gilt stress", "2022-09-26", "risk-off volume", "cross-asset volatility", "global breadth weakness", "overnight-event"),
    ("SVB failure", "2023-03-10", "banking-sector volume spike", "financial-sector volatility", "sector-led weak breadth", "volume-driven"),
    ("US regional-bank stress", "2023-05-04", "financial-sector turnover", "bank volatility", "sector-led stress", "volume-driven"),
    ("US debt-ceiling repricing", "2023-05-24", "elevated turnover", "policy uncertainty volatility", "mixed breadth", "overnight-event"),
    ("Treasury-yield shock", "2023-10-20", "elevated volume", "rates volatility", "broad de-risking", "volatility-shock"),
    ("Yen carry unwind", "2024-08-05", "market-wide turnover", "cross-asset volatility spike", "broad selling", "volatility-shock"),
    ("Fidelity service disruption", "2024-08-05", "high customer trading demand", "volatile-session conditions", "broad client attention", "volume-driven"),
    ("Fidelity access disruption", "2025-05-12", "elevated client activity", "service-access incident", "client access impact", "overnight-event"),
    ("Flash crash", "2010-05-06", "abrupt volume dislocation", "extreme intraday volatility", "rapidly deteriorating breadth", "volume-driven"),
    ("European debt crisis selloff", "2011-08-08", "high turnover", "volatility spike", "broad risk-off", "volatility-shock"),
    ("US debt downgrade", "2011-08-08", "high turnover", "volatility spike", "broad risk-off", "overnight-event"),
    ("FOMC taper communication", "2013-05-22", "elevated turnover", "rate volatility", "broad repricing", "overnight-event"),
    ("Oil-price selloff", "2014-12-15", "energy-led volume", "commodity volatility", "sector-led weakness", "volume-driven"),
    ("Trade-war escalation", "2019-08-05", "elevated turnover", "policy volatility", "broad risk-off", "overnight-event"),
    ("COVID vaccine rotation", "2020-11-09", "rotation volume", "cross-sector volatility", "sharp breadth rotation", "overnight-event"),
    ("Russia invasion shock", "2022-02-24", "high turnover", "geopolitical volatility", "broad risk-off", "overnight-event"),
    ("US bank downgrade volatility", "2023-08-02", "financial-sector volume", "credit volatility", "sector-led weakness", "volatility-shock"),
    ("Japan market selloff", "2024-08-05", "high turnover", "global volatility shock", "broad selling", "volatility-shock"),
]

PRECEDENTS = [
    Precedent(
        event_name=name, date=date, volume_signature=volume, volatility_signature=volatility,
        breadth_signature=breadth, event_type=event_type,
        operational_outcome="Activate incident readiness: monitor order-entry latency, capacity, vendor dependencies, and customer-access channels.",
    )
    for name, date, volume, volatility, breadth, event_type in _EVENTS
]


class HashEmbeddingFunction:
    """Deterministic local embedding fallback when Gemini/MiniLM is unavailable."""

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma's required parameter name is ``input``.
        vectors = []
        for text in input:
            vector = np.zeros(256, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
                vector[int.from_bytes(digest, "little") % len(vector)] += 1
            norm = np.linalg.norm(vector)
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors

    @staticmethod
    def name() -> str:
        return "voltex_hash_embedding_v1"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


class GeminiEmbeddingFunction:
    """Gemini embedding-001 adapter for Chroma when a key is available."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(model="gemini-embedding-001", contents=input)
        return [list(item.values) for item in response.embeddings]

    @staticmethod
    def name() -> str:
        return "gemini_embedding_001"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


class MiniLMEmbeddingFunction:
    """all-MiniLM-L6-v2 local fallback; model loading is lazy and offline-safe."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.model.encode(input, normalize_embeddings=True).tolist()

    @staticmethod
    def name() -> str:
        return "all_minilm_l6_v2"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def select_embedding_function():
    """Gemini first, all-MiniLM offline fallback, then a local deterministic guard."""

    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        try:
            return GeminiEmbeddingFunction(key)
        except Exception:
            pass
    try:
        return MiniLMEmbeddingFunction()
    except Exception:
        # Keeps alert availability intact when neither package/model cache nor
        # network is present; retrieval remains containment-safe.
        return HashEmbeddingFunction()


def ingest_corpus(persist_path: str | Path = "data/chroma"):
    """Create a local persisted Chroma collection using no network dependency."""

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        "voltex_precedents", metadata={"hnsw:space": "cosine"}, embedding_function=select_embedding_function()
    )
    ids = [f"precedent-{index:02d}" for index in range(len(PRECEDENTS))]
    collection.upsert(
        ids=ids, documents=[item.document() for item in PRECEDENTS],
        metadatas=[{"event_name": item.event_name, "date": item.date, "event_type": item.event_type, "identifier": item.identifier}
                   for item in PRECEDENTS],
    )
    return collection
