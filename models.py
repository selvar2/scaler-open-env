"""
Typed models for the Email Triage Environment.

Defines the data contracts between client and server:
  - EmailTriageAction: what the agent sends (classification, routing, response)
  - EmailTriageObservation: what the agent sees (email content, feedback)
  - EmailTriageState: episode metadata
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

from openenv.core.env_server.types import Action, Observation


# ─── Enums ───────────────────────────────────────────────────

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM = "spam"


class Department(str, Enum):
    ENGINEERING = "engineering"
    PRODUCT = "product"
    BILLING = "billing"
    SECURITY = "security"
    SUPPORT = "support"
    SPAM_FILTER = "spam_filter"


class Sentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


# ─── Action ──────────────────────────────────────────────────

class EmailTriageAction(Action):
    """Action the agent takes to triage an email."""
    priority: str = Field(..., max_length=20, description="Priority level: low, medium, high, critical")
    category: str = Field(..., max_length=30, description="Email category: bug_report, feature_request, billing, account_access, general_inquiry, spam")
    department: str = Field(default="", max_length=30, description="Department to route to (required for medium/hard tasks)")
    response: str = Field(default="", max_length=10000, description="Draft response text (required for hard task)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence in classification")


# ─── Observation ─────────────────────────────────────────────

class EmailContent(BaseModel):
    """The email the agent needs to triage."""
    sender: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1
    sender_history: Dict[str, Any] = Field(default_factory=dict)


class FeedbackDetail(BaseModel):
    """Detailed feedback on agent's action."""
    priority_correct: Optional[bool] = None
    category_correct: Optional[bool] = None
    department_correct: Optional[bool] = None
    response_quality: Optional[float] = None
    message: str = ""


class EmailTriageObservation(Observation):
    """What the agent observes after each step."""
    email: Optional[EmailContent] = None
    task_id: str = ""
    task_difficulty: str = ""
    step_number: int = 0
    max_steps: int = 1
    feedback: Optional[FeedbackDetail] = None
    available_departments: List[str] = Field(default_factory=list)
    available_priorities: List[str] = Field(default_factory=list)
    available_categories: List[str] = Field(default_factory=list)


# ─── State (use openenv State directly) ─────────────────────

class EmailTriageState(BaseModel):
    """Extended episode metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    task_difficulty: str = ""
    total_reward: float = 0.0
    emails_processed: int = 0
    correct_classifications: int = 0
