from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import settings
from .config_store import get_active_config
from .rag import RAGAgent


@dataclass
class _AgentEntry:
    agent: RAGAgent
    updated_at: Optional[str]


class AgentManager:
    def __init__(self) -> None:
        self._cache: Dict[str, _AgentEntry] = {}

    def reset(self) -> None:
        self._cache.clear()

    def get_agent(self) -> RAGAgent:
        active_config = get_active_config()
        config_id = active_config.get("id") if active_config else "default"
        updated_at = active_config.get("updated_at") if active_config else None
        entry = self._cache.get(config_id)
        if entry and entry.updated_at == updated_at:
            return entry.agent
        settings.reload()
        agent = RAGAgent()
        self._cache[config_id] = _AgentEntry(agent=agent, updated_at=updated_at)
        return agent
