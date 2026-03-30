---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# Email Triage Environment

A real-world OpenEnv environment where AI agents classify, prioritize, route, and respond to customer support emails.

## Environment Description

Customer support teams process hundreds of emails daily. Each email must be:
1. **Classified** by priority (low/medium/high/critical) and category (bug/feature/billing/etc.)
2. **Routed** to the correct department (engineering/product/billing/security/support)
3. **Responded to** with an appropriate draft addressing key concerns

This environment simulates that workflow with 13 realistic emails spanning easy, medium, and hard difficulty levels.

## Action Space

```python
{
    "priority": str,     # "low" | "medium" | "high" | "critical"
    "category": str,     # "bug_report" | "feature_request" | "billing" | "account_access" | "general_inquiry" | "spam"
    "department": str,   # "engineering" | "product" | "billing" | "security" | "support" | "spam_filter"
    "response": str,     # Draft response text (required for hard task)
    "confidence": float  # 0.0-1.0 agent confidence
}
```

## Observation Space

```python
{
    "email": {
        "sender": str,
        "subject": str,
        "body": str,
        "timestamp": str,
        "has_attachment": bool,
        "thread_length": int,
        "sender_history": {
            "previous_tickets": int,
            "account_age_days": int,
            "plan": str
        }
    },
    "task_id": str,
    "task_difficulty": str,
    "required_fields": list[str],
    "available_priorities": list[str],
    "available_categories": list[str],
    "available_departments": list[str]
}
```

## Tasks

| Task | Difficulty | Description | Scoring |
|------|-----------|-------------|---------|
| `classification` | Easy | Classify priority + category | 0.5 priority + 0.5 category |
| `routing` | Medium | Classify + route to department | 0.3 priority + 0.3 category + 0.4 department |
| `full_triage` | Hard | Classify + route + draft response | 0.15 priority + 0.15 category + 0.2 department + 0.5 response quality |

All graders provide **partial credit**:
- Adjacent priority levels get partial points (e.g., "high" when answer is "critical")
- Response quality measured by coverage of key points

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
PYTHONPATH=. uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t email-triage-env .
docker run -d -p 8000:8000 email-triage-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (params: task_id, email_id, seed) |
| `/step` | POST | Submit triage action |
| `/state` | GET | Get episode metadata |
| `/health` | GET | Server health check |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | POST | Score a specific email+action |
| `/baseline` | POST | Run heuristic baseline on all tasks |

## Baseline Scores

```bash
# Run standalone (no server needed)
python baseline.py --standalone

# Run against server
python baseline.py --url http://localhost:8000
```

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| classification | Easy | ~0.75 |
| routing | Medium | ~0.70 |
| full_triage | Hard | ~0.45 |

## Reward Function

The reward function provides **meaningful partial progress signals**:

- **Binary correctness** for each field (priority, category, department)
- **Adjacent-level credit** for priority (e.g., "high" vs "critical" gets partial points)
- **Response quality** measured by keyword coverage of expected key points
- **Length penalties** for responses that are too short (<10 words) or too long (>500 words)

This ensures agents receive useful gradient signal even when they don't get a perfect score.

## Deployment

```bash
# Deploy to Hugging Face Spaces
openenv push --repo-id YOUR_USERNAME/email-triage-env
```
