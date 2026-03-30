"""
Email Triage Environment — Standalone (dict-based) Implementation.

Used by baseline.py for offline evaluation without the OpenEnv server.
For the OpenEnv-compatible version, see email_triage_environment.py.
"""

import random
import uuid
from typing import Optional, Dict, Any

from email_triage_env.models import (
    EmailTriageAction, EmailTriageObservation, EmailTriageState,
    EmailContent, FeedbackDetail,
    Priority, Category, Department,
)
from email_triage_env.data.emails import EMAILS, get_emails_by_difficulty, get_email_by_id
from email_triage_env.tasks.graders import GRADERS, TASK_DEFINITIONS


class EmailTriageEnvironment:
    """
    A real-world email triage environment for AI agents.

    Tasks:
      - classification (easy): Classify priority + category
      - routing (medium): Classify + route to department
      - full_triage (hard): Classify + route + draft response

    Each episode presents one email. The agent triages it in one step.
    Graders score 0.0–1.0 with partial credit.
    """

    def __init__(self):
        self._state = EmailTriageState()
        self._current_email: Optional[Dict[str, Any]] = None
        self._current_task: Optional[Dict[str, Any]] = None
        self._done = False

    def reset(
        self,
        task_id: str = "classification",
        email_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Start a new episode.

        Args:
            task_id: Which task to run (classification/routing/full_triage)
            email_id: Specific email to use (random if not specified)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Find task definition
        task_def = None
        for td in TASK_DEFINITIONS:
            if td["task_id"] == task_id:
                task_def = td
                break
        if task_def is None:
            task_def = TASK_DEFINITIONS[0]

        self._current_task = task_def

        # Select email
        if email_id:
            self._current_email = get_email_by_id(email_id)
        else:
            difficulty = task_def["difficulty"]
            pool = get_emails_by_difficulty(difficulty)
            if not pool:
                pool = EMAILS
            self._current_email = random.choice(pool)

        # Initialize state
        self._state = EmailTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_def["task_id"],
            task_difficulty=task_def["difficulty"],
            total_reward=0.0,
            emails_processed=0,
            correct_classifications=0,
        )
        self._done = False

        email = self._current_email
        obs = {
            "done": False,
            "reward": 0.0,
            "observation": {
                "email": {
                    "sender": email["sender"],
                    "subject": email["subject"],
                    "body": email["body"],
                    "timestamp": email["timestamp"],
                    "has_attachment": email.get("has_attachment", False),
                    "thread_length": email.get("thread_length", 1),
                    "sender_history": email.get("sender_history", {}),
                },
                "task_id": task_def["task_id"],
                "task_difficulty": task_def["difficulty"],
                "task_description": task_def["description"],
                "required_fields": task_def["required_fields"],
                "step_number": 0,
                "max_steps": task_def["max_steps"],
                "available_priorities": [p.value for p in Priority],
                "available_categories": [c.value for c in Category],
                "available_departments": [d.value for d in Department],
            },
        }
        return obs

    def step(self, action_data: Dict[str, Any]) -> dict:
        """
        Process the agent's triage action.

        Args:
            action_data: Dict with priority, category, department, response fields
        """
        if self._done:
            return {
                "done": True,
                "reward": 0.0,
                "observation": {
                    "feedback": {"message": "Episode already complete. Call reset()."},
                    "step_number": self._state.step_count,
                    "max_steps": self._current_task["max_steps"] if self._current_task else 1,
                },
            }

        self._state.step_count += 1
        self._done = True

        # Grade the action
        ground_truth = self._current_email["ground_truth"]
        grader_name = self._current_task["grader"]
        grader_fn = GRADERS[grader_name]
        result = grader_fn(action_data, ground_truth)

        score = result["score"]
        self._state.total_reward = score
        self._state.emails_processed = 1
        if score >= 0.8:
            self._state.correct_classifications = 1

        feedback = FeedbackDetail(
            priority_correct=result["details"].get("priority") == "correct",
            category_correct=result["details"].get("category") == "correct",
            department_correct=result["details"].get("department") == "correct",
            response_quality=score,
            message=f"Score: {score:.3f}. Details: {result['details']}",
        )

        return {
            "done": True,
            "reward": score,
            "observation": {
                "feedback": feedback.model_dump(),
                "step_number": self._state.step_count,
                "max_steps": self._current_task["max_steps"],
                "score_breakdown": result["details"],
            },
        }

    @property
    def state(self) -> dict:
        return self._state.model_dump()
