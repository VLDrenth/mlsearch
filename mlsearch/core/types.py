from __future__ import annotations
from typing import Literal, List, Dict
from pydantic import BaseModel, Field

class Step(BaseModel):
    id: str
    action: Literal["spawn", "tool", "merge"]
    worker: str | None = None
    tool: str | None = None
    args: Dict | None = None
    depends_on: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    steps: List[Step]