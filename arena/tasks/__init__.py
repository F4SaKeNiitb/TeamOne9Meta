from .research import RESEARCH_TASKS
from .consumer_drift import CONSUMER_TASKS

ALL_TASKS = {**RESEARCH_TASKS, **CONSUMER_TASKS}

__all__ = ["RESEARCH_TASKS", "CONSUMER_TASKS", "ALL_TASKS"]
