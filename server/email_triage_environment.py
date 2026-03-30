"""
Email Triage Environment Implementation.

A real-world environment where AI agents classify, route, and respond
to customer support emails.
"""

import random
from uuid import uuid4
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageAction, EmailTriageObservation, EmailContent, FeedbackDetail, Priority, Category, Department
    from ..data.emails import EMAILS, get_emails_by_difficulty, get_email_by_id
    from ..tasks.graders import GRADERS, TASK_DEFINITIONS
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation, EmailContent, FeedbackDetail, Priority, Category, Department
    from data.emails import EMAILS, get_emails_by_difficulty, get_email_by_id
    from tasks.graders import GRADERS, TASK_DEFINITIONS


class EmailTriageEnvironment(Environment):
    """
    Email triage environment for AI agents.

    Tasks:
      - classification (easy): Classify priority + category
      - routing (medium): Classify + route to department
      - full_triage (hard): Classify + route + draft response
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email = None
        self._current_task = None
        self._done = False
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> EmailTriageObservation:
        task_id = kwargs.get("task_id", "classification")
        email_id = kwargs.get("email_id", None)

        if seed is not None:
            self._rng = random.Random(seed)

        # Find task
        task_def = TASK_DEFINITIONS[0]
        for td in TASK_DEFINITIONS:
            if td["task_id"] == task_id:
                task_def = td
                break

        self._current_task = task_def

        # Select email
        if email_id:
            self._current_email = get_email_by_id(email_id)
        else:
            pool = get_emails_by_difficulty(task_def["difficulty"]) or EMAILS
            self._current_email = self._rng.choice(pool)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._done = False

        email = self._current_email
        return EmailTriageObservation(
            done=False,
            reward=0.0,
            email=EmailContent(
                sender=email["sender"],
                subject=email["subject"],
                body=email["body"],
                timestamp=email["timestamp"],
                has_attachment=email.get("has_attachment", False),
                thread_length=email.get("thread_length", 1),
                sender_history=email.get("sender_history", {}),
            ),
            task_id=task_def["task_id"],
            task_difficulty=task_def["difficulty"],
            step_number=0,
            max_steps=task_def["max_steps"],
            available_priorities=[p.value for p in Priority],
            available_categories=[c.value for c in Category],
            available_departments=[d.value for d in Department],
        )

    def step(self, action: EmailTriageAction, timeout_s: Optional[float] = None, **kwargs) -> EmailTriageObservation:
        # Auto-reset if step called without reset (HTTP stateless mode)
        if self._current_email is None or self._current_task is None:
            self.reset()

        if self._done:
            return EmailTriageObservation(
                done=True,
                reward=0.0,
                feedback=FeedbackDetail(message="Episode already complete. Call reset()."),
                step_number=self._state.step_count,
                max_steps=self._current_task["max_steps"] if self._current_task else 1,
            )

        self._state.step_count += 1
        self._done = True

        # Grade
        ground_truth = self._current_email["ground_truth"]
        grader_fn = GRADERS[self._current_task["grader"]]
        action_dict = {
            "priority": action.priority,
            "category": action.category,
            "department": action.department,
            "response": action.response,
            "confidence": action.confidence,
        }
        result = grader_fn(action_dict, ground_truth)
        score = result["score"]

        return EmailTriageObservation(
            done=True,
            reward=score,
            feedback=FeedbackDetail(
                priority_correct=result["details"].get("priority") == "correct",
                category_correct=result["details"].get("category") == "correct",
                department_correct=result["details"].get("department") == "correct",
                response_quality=score,
                message=f"Score: {score:.3f}. Details: {result['details']}",
            ),
            step_number=self._state.step_count,
            max_steps=self._current_task["max_steps"],
        )

    @property
    def state(self) -> State:
        return self._state
