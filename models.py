"""
EvalForge — LLM Output Evaluation Environment
models.py — Typed Action, Observation, State extending OpenEnv base classes.
"""

from typing import Optional, Dict, Any, List
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class EvalAction(Action):
    """Agent's evaluation verdict on an LLM output."""

    factual_accuracy: float = Field(
        ..., ge=0.0, le=1.0,
        description="How factually correct is the LLM output? (0=wrong, 1=correct)",
    )
    instruction_following: float = Field(
        ..., ge=0.0, le=1.0,
        description="Did the LLM follow the instruction? (0=ignored, 1=perfectly)",
    )
    identified_failure: str = Field(
        ...,
        description=(
            "Failure mode detected. One of: factual_error, "
            "hallucinated_citation, sycophancy, confident_when_uncertain, none"
        ),
    )
    reasoning: str = Field(
        ..., min_length=20,
        description="Explanation for the scores and identified failure.",
    )
    overall_verdict: str = Field(
        ...,
        description="Quality verdict: pass, fail, or flag_for_review",
    )


class EvalObservation(Observation):
    """What the agent sees: the LLM response to evaluate plus feedback."""

    task_id: str = ""
    difficulty: str = ""
    original_prompt: str = ""
    llm_response: str = ""
    context: Optional[str] = None
    evaluation_instructions: str = ""
    step: int = 0
    feedback: str = ""


class EvalState(State):
    """Internal episode state (no ground truth exposed)."""

    task_id: str = ""
    max_steps: int = 3
    current_score: float = 0.0
