#!/usr/bin/env python3
"""
Inference Script for the Email Triage Environment.
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import json
import os
import re
import sys
from typing import Any, Dict, List

from openai import OpenAI

from client import EmailTriageEnv
from models import EmailTriageAction

# ---------------------------------------------------------------------------
# Configuration (mandatory env vars per hackathon rules)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "https://selva12-email-triage-env.hf.space")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a customer support triage agent. You will receive a customer email and must classify it.

You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no extra text). The JSON must have these exact fields:

{
  "priority": "<low|medium|high|critical>",
  "category": "<bug_report|feature_request|billing|account_access|general_inquiry|spam>",
  "department": "<engineering|product|billing|security|support|spam_filter>",
  "response": "<your draft response to the customer>",
  "confidence": <0.0 to 1.0>
}

Classification guidelines:
- priority: How urgent is this?
  - critical: Revenue loss, security breach, data loss, system-wide outage
  - high: User blocked, crashes, billing errors, account lockout
  - medium: Degraded experience, missing data, plan changes
  - low: Feature requests, spam, general questions
- category: What type of issue?
  - bug_report: Software crashes, errors, data loss, broken features
  - feature_request: New feature suggestions
  - billing: Invoice issues, plan changes, charges
  - account_access: Login problems, password resets, lockouts
  - general_inquiry: Questions, compliance, vendor assessments
  - spam: Scam emails, lottery, phishing
- department: Who should handle it?
  - engineering: Bug fixes, technical issues
  - product: Feature requests, roadmap
  - billing: Payment, invoices, plan changes
  - security: Account access, compliance, data breaches
  - support: General inquiries, escalations
  - spam_filter: Spam emails
- response: Write a professional reply that:
  - Acknowledges the customer's concern
  - Addresses their specific issues
  - Provides next steps or timeline
  - Matches the urgency level

Remember: Output ONLY the JSON object, nothing else."""


# ---------------------------------------------------------------------------
# Heuristic fallback (used when LLM response cannot be parsed)
# ---------------------------------------------------------------------------

def heuristic_agent(email: Dict[str, Any]) -> Dict[str, Any]:
    """Simple keyword-based agent. Returns action dict."""
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
        "bug_report": "engineering",
        "feature_request": "product",
        "billing": "billing",
        "account_access": "security",
        "spam": "spam_filter",
        "general_inquiry": "support",
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
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# LLM-based agent (uses OpenAI client as required)
# ---------------------------------------------------------------------------

def llm_agent(email: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    """Use an LLM to triage the email. Returns action dict."""
    email_text = (
        f"From: {email.get('sender', 'unknown')}\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Body:\n{email.get('body', '')}\n"
    )

    sender_history = email.get("sender_history", {})
    if sender_history:
        email_text += (
            f"\nSender History:\n"
            f"  Previous tickets: {sender_history.get('previous_tickets', 'unknown')}\n"
            f"  Account age: {sender_history.get('account_age_days', 'unknown')} days\n"
            f"  Plan: {sender_history.get('plan', 'unknown')}\n"
        )

    if email.get("has_attachment"):
        email_text += "\n[This email has attachments]\n"

    if email.get("thread_length", 1) > 1:
        email_text += f"\n[This is part of a thread with {email['thread_length']} messages]\n"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": email_text},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        content = completion.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [WARN] LLM request failed ({exc}), falling back to heuristic")
        return heuristic_agent(email)

    # Parse JSON from response (handle markdown code blocks)
    if "```" in content:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if blocks:
            content = blocks[0].strip()

    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse LLM response, falling back to heuristic")
        print(f"  LLM output: {content[:200]}")
        return heuristic_agent(email)

    # Ensure required fields exist with defaults
    return {
        "priority": action.get("priority", "medium"),
        "category": action.get("category", "general_inquiry"),
        "department": action.get("department", "support"),
        "response": action.get("response", ""),
        "confidence": action.get("confidence", 0.8),
    }


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN must be set for LLM inference.")
        sys.exit(1)

    # Initialize OpenAI client (mandatory per hackathon rules)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to the Email Triage Environment
    env = EmailTriageEnv(base_url=ENV_URL)

    print(f"START inference.py | model={MODEL_NAME} | env={ENV_URL}")
    print(f"  API:   {API_BASE_URL}")
    print()

    # Task IDs and their configurations
    tasks = [
        {"task_id": "classification", "difficulty": "easy"},
        {"task_id": "routing", "difficulty": "medium"},
        {"task_id": "full_triage", "difficulty": "hard"},
    ]

    all_results: Dict[str, Any] = {}

    for task in tasks:
        task_id = task["task_id"]
        difficulty = task["difficulty"]

        print(f"[START] task={task_id}", flush=True)
        print(f"  Task: {task_id} ({difficulty})")
        print(f"  {'-' * 50}")

        scores: List[float] = []

        # Run multiple episodes per task with different seeds
        num_episodes = {"easy": 5, "medium": 4, "hard": 3}[difficulty]

        for i in range(num_episodes):
            # Reset environment for this task/episode
            result = env.reset(task_id=task_id, seed=42 + i)

            # Extract email from observation
            obs = result.observation if hasattr(result, "observation") else result
            email_data: Dict[str, Any] = {}

            if hasattr(obs, "email") and obs.email is not None:
                # Typed observation from EnvClient
                email_obj = obs.email
                email_data = {
                    "sender": getattr(email_obj, "sender", ""),
                    "subject": getattr(email_obj, "subject", ""),
                    "body": getattr(email_obj, "body", ""),
                    "timestamp": getattr(email_obj, "timestamp", ""),
                    "has_attachment": getattr(email_obj, "has_attachment", False),
                    "thread_length": getattr(email_obj, "thread_length", 1),
                    "sender_history": getattr(email_obj, "sender_history", {}),
                }
            elif isinstance(obs, dict):
                # Dict-based observation (HTTP fallback)
                obs_inner = obs.get("observation", obs)
                email_data = obs_inner.get("email", {})
            else:
                # Try to extract from result directly
                if isinstance(result, dict):
                    obs_inner = result.get("observation", result)
                    email_data = obs_inner.get("email", {})

            # Run LLM agent on the email
            action_dict = llm_agent(email_data, client)

            # Step with the action
            action = EmailTriageAction(
                priority=action_dict["priority"],
                category=action_dict["category"],
                department=action_dict.get("department", ""),
                response=action_dict.get("response", ""),
                confidence=action_dict.get("confidence", 0.8),
            )

            step_result = env.step(action)

            # Extract reward and clamp to (0, 1) open interval
            if hasattr(step_result, "reward"):
                score = step_result.reward or 0.01
            elif isinstance(step_result, dict):
                score = step_result.get("reward", 0.01)
            else:
                score = 0.01
            score = max(0.01, min(float(score), 0.99))

            scores.append(score)

            print(f"[STEP] step={i+1} reward={score}", flush=True)
            subject = email_data.get("subject", "unknown")[:50]
            print(f"    [{i+1}/{num_episodes}] {subject} | score={score:.3f}")

        avg = sum(scores) / len(scores) if scores else 0.01
        avg = max(0.01, min(avg, 0.99))
        all_results[task_id] = {
            "difficulty": difficulty,
            "avg_score": round(avg, 3),
            "scores": scores,
        }
        print(f"    Average: {avg:.3f}\n")
        print(f"[END] task={task_id} score={avg:.2f} steps={len(scores)}", flush=True)

    # Summary
    overall = sum(d["avg_score"] for d in all_results.values()) / len(all_results) if all_results else 0
    for task_id, data in all_results.items():
        print(f"  {task_id:20s} ({data['difficulty']:6s}) -> {data['avg_score']:.3f}")

    print(f"\n  Overall average: {overall:.3f}")


if __name__ == "__main__":
    main()
