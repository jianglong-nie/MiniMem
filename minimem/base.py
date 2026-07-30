"""Data structures used by MiniMem."""

import datetime
from dataclasses import asdict, dataclass


@dataclass
class MemoryItem:
    """One factual memory extracted from a conversation."""

    observed_at: datetime.datetime
    fact: str

    def to_dict(self) -> dict:
        """Convert the memory to a JSON-serializable dictionary."""

        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        return data
