"""
Synthetic email dataset for the Email Triage Environment.

Each email has a ground-truth classification used by graders.
Emails range from obvious (easy) to ambiguous (hard).
"""

from typing import Dict, List, Any, Optional

# ─── Email Templates with Ground Truth ────────────────────────

EMAILS: List[Dict[str, Any]] = [
    # ── EASY: Clear signals, obvious classification ──
    {
        "id": "easy_001",
        "difficulty": "easy",
        "sender": "john.doe@company.com",
        "subject": "App crashes when uploading files larger than 10MB",
        "body": "Hi Support,\n\nEvery time I try to upload a file larger than 10MB, the application crashes immediately with error code 500. This started after the latest update (v3.2.1). I've tried clearing cache and reinstalling but the issue persists.\n\nSteps to reproduce:\n1. Open the app\n2. Go to Upload section\n3. Select any file > 10MB\n4. App crashes\n\nThis is blocking our team's workflow.\n\nThanks,\nJohn",
        "timestamp": "2025-03-15T09:30:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 2, "account_age_days": 365, "plan": "enterprise"},
        "ground_truth": {
            "priority": "high",
            "category": "bug_report",
            "department": "engineering",
            "sentiment": "frustrated",
            "key_response_points": ["acknowledge crash", "mention investigating", "ask for logs", "provide workaround"],
        },
    },
    {
        "id": "easy_002",
        "difficulty": "easy",
        "sender": "spam@lottery-winner.xyz",
        "subject": "YOU WON $1,000,000!!! CLAIM NOW!!!",
        "body": "CONGRATULATIONS!!!\n\nYou have been selected as the WINNER of our international lottery!\nClaim your $1,000,000 prize NOW by clicking the link below and providing your bank details.\n\nACT FAST - This offer expires in 24 hours!\n\nhttp://totally-not-a-scam.xyz/claim",
        "timestamp": "2025-03-15T03:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 0, "account_age_days": 0, "plan": "none"},
        "ground_truth": {
            "priority": "low",
            "category": "spam",
            "department": "spam_filter",
            "sentiment": "neutral",
            "key_response_points": ["mark as spam", "no response needed"],
        },
    },
    {
        "id": "easy_003",
        "difficulty": "easy",
        "sender": "sarah.m@startup.io",
        "subject": "Cannot log in - password reset not working",
        "body": "Hello,\n\nI've been locked out of my account for 2 days now. I tried the password reset but never received the email. I've checked spam folders. My email is sarah.m@startup.io.\n\nI need access urgently as I have a presentation tomorrow.\n\nPlease help!\nSarah",
        "timestamp": "2025-03-14T16:45:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 1, "account_age_days": 180, "plan": "pro"},
        "ground_truth": {
            "priority": "high",
            "category": "account_access",
            "department": "security",
            "sentiment": "frustrated",
            "key_response_points": ["acknowledge urgency", "verify identity", "manual reset steps", "timeline"],
        },
    },
    {
        "id": "easy_004",
        "difficulty": "easy",
        "sender": "mike.chen@bigcorp.com",
        "subject": "Request: Add dark mode to dashboard",
        "body": "Hi team,\n\nOur team uses the dashboard 8+ hours daily and the bright theme causes eye strain. Could you please add a dark mode option?\n\nThis would be a great quality-of-life improvement for power users.\n\nThanks,\nMike",
        "timestamp": "2025-03-15T11:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 5, "account_age_days": 730, "plan": "enterprise"},
        "ground_truth": {
            "priority": "low",
            "category": "feature_request",
            "department": "product",
            "sentiment": "neutral",
            "key_response_points": ["acknowledge request", "add to roadmap", "thank for feedback"],
        },
    },
    {
        "id": "easy_005",
        "difficulty": "easy",
        "sender": "accounting@partner.org",
        "subject": "Invoice #4521 - Incorrect amount charged",
        "body": "Dear Billing Department,\n\nWe received invoice #4521 dated March 10, 2025, but the amount of $2,500 does not match our agreement of $1,800/month. Please review and issue a corrected invoice.\n\nOur contract reference: CT-2024-0892\n\nRegards,\nAccounting Team",
        "timestamp": "2025-03-15T08:15:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 3, "account_age_days": 400, "plan": "enterprise"},
        "ground_truth": {
            "priority": "high",
            "category": "billing",
            "department": "billing",
            "sentiment": "neutral",
            "key_response_points": ["acknowledge error", "review invoice", "reference contract", "timeline for correction"],
        },
    },

    # ── MEDIUM: Some ambiguity, requires reasoning ──
    {
        "id": "med_001",
        "difficulty": "medium",
        "sender": "dev.team@agency.co",
        "subject": "API rate limits too restrictive for our use case",
        "body": "Hi,\n\nWe're building an integration that needs to make ~500 API calls per minute, but the current rate limit is 100/min. Is this a bug in our plan settings or do we need to upgrade?\n\nWe're on the Pro plan and the docs say 'generous rate limits' but don't specify exact numbers.\n\nThis is blocking our client delivery.\n\nThanks",
        "timestamp": "2025-03-15T10:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 8, "account_age_days": 200, "plan": "pro"},
        "ground_truth": {
            "priority": "medium",
            "category": "general_inquiry",
            "department": "support",
            "sentiment": "frustrated",
            "key_response_points": ["clarify rate limits for plan", "suggest enterprise upgrade", "provide documentation link"],
        },
    },
    {
        "id": "med_002",
        "difficulty": "medium",
        "sender": "lisa.wong@healthcare.com",
        "subject": "Data export missing records from February",
        "body": "Hello,\n\nI exported all records for February 2025 but the CSV only contains 1,847 rows. Our system shows we processed 2,103 records that month. The 256 missing records are critical for our compliance audit.\n\nCould this be a bug in the export function or is there a row limit I'm not aware of?\n\nWe need this resolved before our audit on March 20th.\n\nBest,\nLisa",
        "timestamp": "2025-03-13T14:30:00Z",
        "has_attachment": True,
        "thread_length": 2,
        "sender_history": {"previous_tickets": 4, "account_age_days": 500, "plan": "enterprise"},
        "ground_truth": {
            "priority": "critical",
            "category": "bug_report",
            "department": "engineering",
            "sentiment": "frustrated",
            "key_response_points": ["acknowledge data loss severity", "escalate to engineering", "mention compliance deadline", "provide interim workaround"],
        },
    },
    {
        "id": "med_003",
        "difficulty": "medium",
        "sender": "alex@freelancer.me",
        "subject": "Downgrade plan but keep data",
        "body": "Hey,\n\nI need to downgrade from Pro to Free plan starting next billing cycle. My current cycle ends April 1st.\n\nQuestions:\n1. Will I lose my existing data?\n2. Can I export everything first?\n3. Will my API keys still work?\n4. If I upgrade again later, does my data come back?\n\nThanks,\nAlex",
        "timestamp": "2025-03-15T12:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 1, "account_age_days": 90, "plan": "pro"},
        "ground_truth": {
            "priority": "medium",
            "category": "billing",
            "department": "billing",
            "sentiment": "neutral",
            "key_response_points": ["answer each question", "data retention policy", "export instructions", "API key implications"],
        },
    },
    {
        "id": "med_004",
        "difficulty": "medium",
        "sender": "security@partner-audit.com",
        "subject": "SOC2 compliance questionnaire - due March 25",
        "body": "Dear Security Team,\n\nAs part of our annual vendor assessment, please complete the attached SOC2 compliance questionnaire by March 25, 2025.\n\nKey areas:\n- Data encryption at rest and in transit\n- Access control and authentication\n- Incident response procedures\n- Data retention and deletion policies\n\nPlease also provide your latest SOC2 Type II report if available.\n\nRegards,\nPartner Audit Team",
        "timestamp": "2025-03-12T09:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 0, "account_age_days": 0, "plan": "enterprise"},
        "ground_truth": {
            "priority": "high",
            "category": "general_inquiry",
            "department": "security",
            "sentiment": "neutral",
            "key_response_points": ["acknowledge deadline", "confirm compliance status", "attach SOC2 report", "assign to security team"],
        },
    },

    # ── HARD: Ambiguous, multi-faceted, requires nuance ──
    {
        "id": "hard_001",
        "difficulty": "hard",
        "sender": "vp.engineering@whale-customer.com",
        "subject": "Considering cancellation - multiple unresolved issues",
        "body": "To whom it may concern,\n\nWe've been a customer for 3 years paying $15,000/month. Over the past quarter, we've experienced:\n\n1. Three major outages (tickets #1089, #1156, #1203)\n2. API response times degraded 40% since January\n3. Our dedicated support contact left and wasn't replaced\n4. The promised analytics feature from the roadmap hasn't materialized\n\nWe're evaluating alternatives and will make a decision by end of March. I'd like to schedule a call with your leadership team to discuss a path forward.\n\nRegards,\nVP Engineering",
        "timestamp": "2025-03-14T08:00:00Z",
        "has_attachment": False,
        "thread_length": 4,
        "sender_history": {"previous_tickets": 12, "account_age_days": 1095, "plan": "enterprise"},
        "ground_truth": {
            "priority": "critical",
            "category": "general_inquiry",
            "department": "support",
            "sentiment": "angry",
            "key_response_points": ["acknowledge each issue", "express urgency", "executive escalation", "schedule call", "retention offer", "dedicated contact"],
        },
    },
    {
        "id": "hard_002",
        "difficulty": "hard",
        "sender": "reporter@tech-news.com",
        "subject": "Press inquiry: Data breach rumors",
        "body": "Hello,\n\nI'm a reporter at TechNews covering cybersecurity. We've received tips from multiple sources claiming your platform experienced a data breach affecting user credentials in late February.\n\nWe're publishing our article by Friday March 17. We'd like to give you the opportunity to comment.\n\nSpecifically:\n1. Can you confirm or deny a security incident in February 2025?\n2. How many users were potentially affected?\n3. What remediation steps have been taken?\n\nPlease respond by end of day Thursday.\n\nBest,\nReporter",
        "timestamp": "2025-03-15T07:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
        "sender_history": {"previous_tickets": 0, "account_age_days": 0, "plan": "none"},
        "ground_truth": {
            "priority": "critical",
            "category": "general_inquiry",
            "department": "security",
            "sentiment": "neutral",
            "key_response_points": ["do not confirm or deny", "escalate to legal and security", "provide PR-approved statement only", "tight deadline awareness"],
        },
    },
    {
        "id": "hard_003",
        "difficulty": "hard",
        "sender": "ops@ecommerce-platform.com",
        "subject": "Webhook failures causing order sync issues - losing revenue",
        "body": "URGENT\n\nSince 2am this morning, approximately 30% of our webhooks are failing with 502 errors. This means orders placed on our site aren't syncing to your system, causing:\n\n- Duplicate charges to customers\n- Inventory mismatches\n- Failed fulfillment for ~200 orders so far\n\nWe process $50K in orders daily through your platform. Every hour this continues costs us ~$2K in manual reconciliation plus customer refunds.\n\nWe need:\n1. Immediate investigation of webhook infrastructure\n2. List of all failed webhook deliveries since 2am\n3. ETA for resolution\n4. A postmortem after resolution\n\nThis is our second webhook-related outage this quarter.\n\nJamie, Operations Lead",
        "timestamp": "2025-03-15T06:30:00Z",
        "has_attachment": True,
        "thread_length": 3,
        "sender_history": {"previous_tickets": 7, "account_age_days": 600, "plan": "enterprise"},
        "ground_truth": {
            "priority": "critical",
            "category": "bug_report",
            "department": "engineering",
            "sentiment": "angry",
            "key_response_points": ["acknowledge revenue impact", "immediate escalation", "provide status page link", "commit to ETA", "list failed webhooks", "postmortem promise", "compensation discussion"],
        },
    },
]


def get_emails_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Filter emails by difficulty level."""
    return [e for e in EMAILS if e["difficulty"] == difficulty]


def get_email_by_id(email_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific email by ID. Returns None if not found."""
    for e in EMAILS:
        if e["id"] == email_id:
            return e
    return None


def get_all_email_ids() -> List[str]:
    """Return all email IDs."""
    return [e["id"] for e in EMAILS]
