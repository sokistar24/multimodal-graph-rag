"""Evaluation pipelines: system runners and the configured evaluation loop."""

from enum import StrEnum


class EvidenceControl(StrEnum):
    """Context manipulations that isolate where a correct answer came from."""

    NORMAL = "normal"
    CLOSED_BOOK = "closed-book"
    SHUFFLED = "shuffled"
    ORACLE = "oracle"
    PARTIAL_GOLD = "partial-gold"

    @property
    def system_name(self) -> str:
        return f"control:{self.value}"
