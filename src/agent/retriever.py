"""Constrained precedent retrieval for VOLTEX alerts."""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Precedent, ingest_corpus
from .signal import Signal


@dataclass(frozen=True)
class RetrievedPrecedent:
    identifier: str
    event_name: str
    date: str
    event_type: str
    cosine_similarity: float
    document: str


class PrecedentRetriever:
    def __init__(self, persist_path: str = "data/chroma", top_k: int = 3, threshold: float = 0.65) -> None:
        self.persist_path, self.top_k, self.threshold = persist_path, top_k, threshold
        self._collection = None

    def _collection_or_create(self):
        if self._collection is None:
            self._collection = ingest_corpus(self.persist_path)
        return self._collection

    @staticmethod
    def query_text(signal: Signal) -> str:
        # A large forecast surprise without preceding-session volume escalation
        # is the pre-open signature of a scheduled/overnight shock (e.g.,
        # referendum or election). This remains derived solely from Signal.
        event_type = "overnight-event" if (
            any(signal.event_flags.values())
            or (abs(signal.vol_zscore) < 1 and signal.forecast_surprise_zscore >= 2)
        ) else ("volatility-shock" if signal.vix_level >= 25 else "volume-driven")
        return (
            f"{signal.risk_tier} {event_type} volume zscore {signal.vol_zscore:.2f} "
            f"VIX {signal.vix_level:.2f} breadth {signal.stress_breadth:.2f} "
            f"forecast surprise {signal.forecast_surprise_zscore:.2f}"
        )

    def retrieve(self, signal: Signal) -> list[RetrievedPrecedent]:
        result = self._collection_or_create().query(
            query_texts=[self.query_text(signal)], n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        matches: list[RetrievedPrecedent] = []
        for document, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            similarity = 1.0 - float(distance)
            if similarity >= self.threshold:
                matches.append(RetrievedPrecedent(
                    identifier=metadata["identifier"], event_name=metadata["event_name"], date=metadata["date"],
                    event_type=metadata["event_type"], cosine_similarity=similarity, document=document,
                ))
        return matches
