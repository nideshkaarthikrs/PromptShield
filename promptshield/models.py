"""
Pydantic v2 data models for the PromptShield OpenEnv environment.

- PromptShieldObservation: What the agent sees each step (the prompt to evaluate + context)
- PromptShieldAction: What the agent does (classification verdict + reasoning)
- PromptShieldReward: What the grader returns (reward value + breakdown)
- PromptShieldState: Current episode state summary
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, List


class PromptShieldObservation(BaseModel):
    prompt_text: str
    context: str
    step_number: int
    history: List[dict] = []
    task_name: str
    total_steps: int


class PromptShieldAction(BaseModel):
    classification: Literal["injection", "safe"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)
    sanitized_prompt: Optional[str] = None


class PromptShieldReward(BaseModel):
    value: float
    breakdown: dict
    is_correct: bool


class PromptShieldState(BaseModel):
    task_name: str
    current_step: int
    total_steps: int
    done: bool
    cumulative_reward: float
    score: float
