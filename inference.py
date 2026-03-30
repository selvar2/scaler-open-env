#!/usr/bin/env python3
"""
Inference script for the Email Triage Environment.

Uses an OpenAI-compatible LLM (via HuggingFace Inference Providers)
to classify, route, and respond to customer support emails.

Prerequisites
-------------
1. Deploy the environment to HF Spaces (or run locally)::

       openenv push --repo-id YOUR_USERNAME/email-triage-env

2. Set your API key::

       export HF_TOKEN=your_token_here

3. Run this script::

       python inference.py

   Or against a local server::

       python inference.py --url http://localhost:8000

   Or run the heuristic baseline (no LLM needed)::

       python inference.py --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ENV_URL = "https://selva12-email-triage-env.hf.space"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

VERBOSE = True

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
# Heuristic baseline (no LLM needed)
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
# LLM-based agent
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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    if "```" in content:
        import re
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if blocks:
            content = blocks[0].strip()

    try:
        action = json.loads(content)
    except json.JSONDecodeError:
        if VERBOSE:
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
# Runner
# ---------------------------------------------------------------------------

def run_inference(
    env_url: str,
    agent_fn,
    agent_name: str = "agent",
) -> Dict[str, Any]:
    """Run the agent against all tasks and emails via the grader endpoint."""
    print(f"\n{'=' * 60}")
    print(f"  Email Triage Inference — {agent_name}")
    print(f"  Environment: {env_url}")
    print(f"{'=' * 60}\n")

    # Get task definitions
    tasks_resp = requests.get(f"{env_url}/tasks", timeout=15)
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json()["tasks"]

    all_results = {}

    for task in tasks:
        task_id = task["task_id"]
        difficulty = task["difficulty"]
        num_emails = task["num_emails"]

        print(f"  Task: {task_id} ({difficulty}) — {num_emails} emails")
        print(f"  {'-' * 50}")

        scores = []

        for i in range(num_emails):
            # Reset to get an email for this task
            reset_resp = requests.post(
                f"{env_url}/reset",
                json={"task_id": task_id, "seed": 42 + i},
                timeout=15,
            )
            reset_resp.raise_for_status()
            reset_data = reset_resp.json()

            obs = reset_data.get("observation", reset_data)
            email = obs.get("email", {})

            # Run agent
            action = agent_fn(email)

            # Score via grader (deterministic, works with HTTP stateless mode)
            step_resp = requests.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=15,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            score = step_data.get("reward", 0.0)
            scores.append(score)

            if VERBOSE:
                subject = email.get("subject", "unknown")[:50]
                print(f"    [{i+1}/{num_emails}] {subject:50s} → {score:.3f}")

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[task_id] = {
            "difficulty": difficulty,
            "avg_score": round(avg, 3),
            "scores": scores,
        }
        print(f"    Average: {avg:.3f}\n")

    # Summary
    print(f"{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for task_id, data in all_results.items():
        print(f"  {task_id:20s} ({data['difficulty']:6s}) → {data['avg_score']:.3f}")

    overall = sum(d["avg_score"] for d in all_results.values()) / len(all_results) if all_results else 0
    print(f"\n  Overall average: {overall:.3f}")
    print()

    return all_results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Email Triage Inference")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_ENV_URL,
        help=f"Environment URL (default: {DEFAULT_ENV_URL})",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use heuristic baseline instead of LLM",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model name",
    )
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model

    if args.baseline:
        run_inference(
            env_url=args.url,
            agent_fn=heuristic_agent,
            agent_name="Heuristic Baseline",
        )
    else:
        if not API_KEY:
            print("ERROR: Set HF_TOKEN or API_KEY environment variable for LLM inference.")
            print("       Or use --baseline for the heuristic agent (no API key needed).")
            sys.exit(1)

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        run_inference(
            env_url=args.url,
            agent_fn=lambda email: llm_agent(email, client),
            agent_name=f"LLM ({MODEL})",
        )


if __name__ == "__main__":
    main()
