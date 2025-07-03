from __future__ import annotations
from typing import Literal, List, Dict
from pydantic import BaseModel, Field

class Step(BaseModel):
    id: str
    action: Literal["spawn", "tool", "merge", "work"]
    worker: str | None = None
    tool: str | None = None
    args: Dict | None = None
    task: str | None = None  # Task description for work steps
    depends_on: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    steps: List[Step]