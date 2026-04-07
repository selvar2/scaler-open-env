"""Email Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, State]
):
    """
    Client for the Email Triage Environment.

    Example:
        >>> with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id="classification")
        ...     print(result.observation.email.subject)
        ...     result = env.step(EmailTriageAction(
        ...         priority="high", category="bug_report"
        ...     ))
        ...     print(result.reward)
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        return {
            "priority": action.priority,
            "category": action.category,
            "department": action.department,
            "response": action.response,
            "confidence": action.confidence,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        obs_data = payload.get("observation", {})
        feedback_data = obs_data.get("feedback")
        feedback = None
        if feedback_data:
            from models import FeedbackDetail
            feedback = FeedbackDetail(**feedback_data)

        observation = EmailTriageObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            feedback=feedback,
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 1),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
