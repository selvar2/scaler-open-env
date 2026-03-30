"""
FastAPI application for the Email Triage Environment.

Uses openenv create_app for standard endpoints (reset/step/state/ws)
plus custom endpoints for tasks, grader, and baseline.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import EmailTriageAction, EmailTriageObservation
    from .email_triage_environment import EmailTriageEnvironment
except (ImportError, ModuleNotFoundError):
    from models import EmailTriageAction, EmailTriageObservation
    from server.email_triage_environment import EmailTriageEnvironment

try:
    from ..tasks.graders import TASK_DEFINITIONS, GRADERS
    from ..data.emails import EMAILS, get_emails_by_difficulty, get_email_by_id
except (ImportError, ModuleNotFoundError):
    from tasks.graders import TASK_DEFINITIONS, GRADERS
    from data.emails import EMAILS, get_emails_by_difficulty, get_email_by_id

# Create the standard OpenEnv app (reset/step/state/ws/health/schema/docs)
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage_env",
    max_concurrent_envs=100,
)


# ─── Custom endpoints for competition requirements ───────────

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Any, Dict


class GraderRequest(BaseModel):
    task_id: str
    email_id: str
    action: Dict[str, Any]


@app.get("/tasks")
def tasks():
    """Return list of tasks and the action schema."""
    result = []
    for td in TASK_DEFINITIONS:
        result.append({
            "task_id": td["task_id"],
            "name": td["name"],
            "difficulty": td["difficulty"],
            "description": td["description"],
            "required_fields": td["required_fields"],
            "max_steps": td["max_steps"],
            "num_emails": len(get_emails_by_difficulty(td["difficulty"])),
            "action_schema": {
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "category": {"type": "string", "enum": ["bug_report", "feature_request", "billing", "account_access", "general_inquiry", "spam"]},
                "department": {"type": "string", "enum": ["engineering", "product", "billing", "security", "support", "spam_filter"]},
                "response": {"type": "string", "description": "Draft response text"},
            },
        })
    return {"tasks": result}


@app.post("/grader")
def grader(req: GraderRequest):
    """Score a specific email+action combination."""
    email = get_email_by_id(req.email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    grader_fn = GRADERS.get(req.task_id)
    if not grader_fn:
        raise HTTPException(status_code=400, detail="Unknown task_id")
    result = grader_fn(req.action, email["ground_truth"])
    return {"email_id": req.email_id, "task_id": req.task_id, "score": result["score"], "details": result["details"]}


@app.post("/baseline")
def baseline():
    """Run heuristic baseline on all tasks."""
    results = {}
    for task_def in TASK_DEFINITIONS:
        task_id = task_def["task_id"]
        emails = get_emails_by_difficulty(task_def["difficulty"]) or EMAILS
        scores = []
        for email in emails:
            action = _heuristic(email)
            result = GRADERS[task_id](action, email["ground_truth"])
            scores.append({"email_id": email["id"], "score": result["score"]})
        avg = sum(s["score"] for s in scores) / len(scores) if scores else 0
        results[task_id] = {"difficulty": task_def["difficulty"], "average_score": round(avg, 3), "scores": scores}
    return {"baseline_results": results}


def _heuristic(email):
    text = (email["subject"] + " " + email["body"]).lower()
    if any(w in text for w in ["urgent", "critical", "asap", "losing revenue", "breach"]):
        priority = "critical"
    elif any(w in text for w in ["crash", "blocked", "cannot", "locked out", "incorrect", "cancellation"]):
        priority = "high"
    elif any(w in text for w in ["issue", "problem", "error", "missing", "downgrade"]):
        priority = "medium"
    else:
        priority = "low"
    if any(w in text for w in ["lottery", "claim", "scam"]):
        cat = "spam"
    elif any(w in text for w in ["crash", "bug", "502", "failing"]):
        cat = "bug_report"
    elif any(w in text for w in ["invoice", "billing", "charged", "downgrade"]):
        cat = "billing"
    elif any(w in text for w in ["password", "login", "locked"]):
        cat = "account_access"
    elif any(w in text for w in ["feature", "request", "dark mode"]):
        cat = "feature_request"
    else:
        cat = "general_inquiry"
    dept_map = {"bug_report": "engineering", "feature_request": "product", "billing": "billing",
                "account_access": "security", "spam": "spam_filter", "general_inquiry": "support"}
    dept = dept_map.get(cat, "support")
    resp = f"Thank you for contacting us about: {email['subject']}. Our {dept} team will investigate. We acknowledge the urgency and will prioritize accordingly."
    return {"priority": priority, "category": cat, "department": dept, "response": resp}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
