"""
Baseline inference script for the Email Triage Environment.

Runs a keyword-based heuristic agent against all 3 tasks
and reports reproducible scores.

Usage:
  # Against local server:
  python baseline.py --url http://127.0.0.1:8000

  # Against HF Space:
  python baseline.py --url https://YOUR-USERNAME-email-triage-env.hf.space

  # Standalone (no server needed):
  python baseline.py --standalone
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def heuristic_agent(email_obs: dict) -> dict:
    """Simple keyword-based agent. Returns action dict."""
    email = email_obs.get("email", {})
    subject = email.get("subject", "").lower()
    body = email.get("body", "").lower()
    text = subject + " " + body

    # Priority
    if any(w in text for w in ["urgent", "critical", "asap", "immediately", "losing revenue", "breach"]):
        priority = "critical"
    elif any(w in text for w in ["crash", "blocked", "cannot", "locked out", "incorrect amount", "cancellation"]):
        priority = "high"
    elif any(w in text for w in ["issue", "problem", "error", "missing", "downgrade"]):
        priority = "medium"
    else:
        priority = "low"

    # Category
    if any(w in text for w in ["won", "lottery", "claim", "scam"]):
        category = "spam"
    elif any(w in text for w in ["crash", "bug", "error", "502", "failing", "broken"]):
        category = "bug_report"
    elif any(w in text for w in ["invoice", "billing", "charged", "downgrade", "payment"]):
        category = "billing"
    elif any(w in text for w in ["password", "login", "locked", "access"]):
        category = "account_access"
    elif any(w in text for w in ["feature", "request", "add", "dark mode"]):
        category = "feature_request"
    else:
        category = "general_inquiry"

    # Department
    dept_map = {
        "bug_report": "engineering", "feature_request": "product",
        "billing": "billing", "account_access": "security",
        "spam": "spam_filter", "general_inquiry": "support",
    }
    department = dept_map.get(category, "support")

    # Response
    response = (
        f"Thank you for contacting us regarding your {category.replace('_', ' ')}. "
        f"We acknowledge your concern about: {email.get('subject', '')}. "
        f"Our {department} team will investigate this matter. "
        f"We understand the urgency and will prioritize accordingly. "
        f"You can expect an update within 24 hours."
    )

    return {
        "priority": priority,
        "category": category,
        "department": department,
        "response": response,
    }


def run_standalone():
    """Run baseline without a server (uses environment directly)."""
    from email_triage_env.server.environment import EmailTriageEnvironment
    from email_triage_env.tasks.graders import TASK_DEFINITIONS
    from email_triage_env.data.emails import get_emails_by_difficulty, EMAILS

    env = EmailTriageEnvironment()
    all_results = {}

    for task_def in TASK_DEFINITIONS:
        task_id = task_def["task_id"]
        difficulty = task_def["difficulty"]
        emails = get_emails_by_difficulty(difficulty)
        if not emails:
            emails = EMAILS

        scores = []
        for email in emails:
            obs = env.reset(task_id=task_id, email_id=email["id"], seed=42)
            email_obs = obs.get("observation", obs)
            action = heuristic_agent(email_obs)
            result = env.step(action)
            score = result.get("reward", 0.0)
            scores.append(score)
            print(f"  [{task_id}] {email['id']:12s} -> score={score:.3f}")

        avg = sum(scores) / len(scores) if scores else 0
        all_results[task_id] = {"difficulty": difficulty, "avg_score": avg, "scores": scores}
        print(f"  {task_id} average: {avg:.3f}")
        print()

    return all_results


def run_server(base_url: str):
    """Run baseline against a running server."""
    import requests

    all_results = {}

    # Get tasks
    tasks_resp = requests.get(f"{base_url}/tasks", timeout=10).json()

    for task in tasks_resp["tasks"]:
        task_id = task["task_id"]
        scores = []

        for i in range(task["num_emails"]):
            # Reset with seed for reproducibility
            reset_resp = requests.post(
                f"{base_url}/reset",
                json={"task_id": task_id, "seed": 42 + i},
                timeout=10,
            ).json()

            email_obs = reset_resp.get("observation", reset_resp)
            action = heuristic_agent(email_obs)

            step_resp = requests.post(
                f"{base_url}/step",
                json={"action": action},
                timeout=10,
            ).json()

            score = step_resp.get("reward", 0.0)
            scores.append(score)
            print(f"  [{task_id}] email {i+1} -> score={score:.3f}")

        avg = sum(scores) / len(scores) if scores else 0
        all_results[task_id] = {"difficulty": task["difficulty"], "avg_score": avg, "scores": scores}
        print(f"  {task_id} average: {avg:.3f}")
        print()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Triage Baseline")
    parser.add_argument("--url", type=str, default=None, help="Server URL")
    parser.add_argument("--standalone", action="store_true", help="Run without server")
    args = parser.parse_args()

    print("=" * 60)
    print("  Email Triage Environment — Baseline Inference")
    print("=" * 60)
    print()

    if args.standalone or args.url is None:
        print("  Mode: standalone (no server)")
        print()
        results = run_standalone()
    else:
        print(f"  Mode: server ({args.url})")
        print()
        results = run_server(args.url)

    print("=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    for task_id, data in results.items():
        print(f"  {task_id:20s} ({data['difficulty']:6s}) -> {data['avg_score']:.3f}")
    print()

    overall = sum(d["avg_score"] for d in results.values()) / len(results) if results else 0
    print(f"  Overall average: {overall:.3f}")
    print()
