"""
Graders for Email Triage tasks.

Each grader scores agent performance 0.0 to 1.0 with partial credit.
Graders are deterministic — same input always produces same score.
"""

from typing import Dict, Any, Optional


def _normalize(value: str) -> str:
    """Normalize string for comparison."""
    return value.lower().strip().replace(" ", "_").replace("-", "_")


def grade_classification(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    EASY TASK: Grade priority + category classification.

    Scoring:
      - Priority correct: 0.5 points
      - Category correct: 0.5 points
      - Priority partially correct (adjacent level): 0.15 points
    Total: 0.0 to 1.0
    """
    score = 0.0
    details = {}

    # Priority scoring (0.5 max)
    pred_priority = _normalize(action.get("priority", ""))
    true_priority = _normalize(ground_truth["priority"])

    priority_levels = ["low", "medium", "high", "critical"]
    if pred_priority == true_priority:
        score += 0.5
        details["priority"] = "correct"
    elif pred_priority in priority_levels and true_priority in priority_levels:
        pred_idx = priority_levels.index(pred_priority)
        true_idx = priority_levels.index(true_priority)
        if abs(pred_idx - true_idx) == 1:
            score += 0.15
            details["priority"] = "adjacent"
        else:
            details["priority"] = "wrong"
    else:
        details["priority"] = "invalid"

    # Category scoring (0.5 max)
    pred_category = _normalize(action.get("category", ""))
    true_category = _normalize(ground_truth["category"])

    if pred_category == true_category:
        score += 0.5
        details["category"] = "correct"
    else:
        details["category"] = "wrong"

    return {"score": round(score, 3), "details": details}


def grade_routing(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    MEDIUM TASK: Grade classification + department routing.

    Scoring:
      - Priority correct: 0.3 points
      - Category correct: 0.3 points
      - Department correct: 0.4 points
      - Partial credit for adjacent priority: 0.1 points
    Total: 0.0 to 1.0
    """
    score = 0.0
    details = {}

    # Priority (0.3 max)
    pred_priority = _normalize(action.get("priority", ""))
    true_priority = _normalize(ground_truth["priority"])
    priority_levels = ["low", "medium", "high", "critical"]

    if pred_priority == true_priority:
        score += 0.3
        details["priority"] = "correct"
    elif pred_priority in priority_levels and true_priority in priority_levels:
        pred_idx = priority_levels.index(pred_priority)
        true_idx = priority_levels.index(true_priority)
        if abs(pred_idx - true_idx) == 1:
            score += 0.1
            details["priority"] = "adjacent"
        else:
            details["priority"] = "wrong"
    else:
        details["priority"] = "invalid"

    # Category (0.3 max)
    pred_category = _normalize(action.get("category", ""))
    true_category = _normalize(ground_truth["category"])
    if pred_category == true_category:
        score += 0.3
        details["category"] = "correct"
    else:
        details["category"] = "wrong"

    # Department routing (0.4 max)
    pred_dept = _normalize(action.get("department", ""))
    true_dept = _normalize(ground_truth["department"])
    if pred_dept == true_dept:
        score += 0.4
        details["department"] = "correct"
    else:
        details["department"] = "wrong"

    return {"score": round(score, 3), "details": details}


def grade_full_triage(action: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    HARD TASK: Grade classification + routing + response quality.

    Scoring:
      - Priority correct: 0.15
      - Category correct: 0.15
      - Department correct: 0.2
      - Response quality: 0.5 (based on key points coverage)
    Total: 0.0 to 1.0
    """
    score = 0.0
    details = {}

    # Priority (0.15 max)
    pred_priority = _normalize(action.get("priority", ""))
    true_priority = _normalize(ground_truth["priority"])
    priority_levels = ["low", "medium", "high", "critical"]

    if pred_priority == true_priority:
        score += 0.15
        details["priority"] = "correct"
    elif pred_priority in priority_levels and true_priority in priority_levels:
        pred_idx = priority_levels.index(pred_priority)
        true_idx = priority_levels.index(true_priority)
        if abs(pred_idx - true_idx) == 1:
            score += 0.05
            details["priority"] = "adjacent"
        else:
            details["priority"] = "wrong"
    else:
        details["priority"] = "invalid"

    # Category (0.15 max)
    pred_category = _normalize(action.get("category", ""))
    true_category = _normalize(ground_truth["category"])
    if pred_category == true_category:
        score += 0.15
        details["category"] = "correct"
    else:
        details["category"] = "wrong"

    # Department (0.2 max)
    pred_dept = _normalize(action.get("department", ""))
    true_dept = _normalize(ground_truth["department"])
    if pred_dept == true_dept:
        score += 0.2
        details["department"] = "correct"
    else:
        details["department"] = "wrong"

    # Response quality (0.5 max)
    response = action.get("response", "").lower()
    key_points = ground_truth.get("key_response_points", [])
    if not response or not key_points:
        details["response"] = "missing"
    else:
        hits = 0
        for point in key_points:
            keywords = point.lower().split()
            if any(kw in response for kw in keywords if len(kw) > 3):
                hits += 1
        coverage = hits / len(key_points) if key_points else 0
        response_score = coverage * 0.5

        # Bonus for response length (reasonable length = better)
        word_count = len(response.split())
        if word_count < 10:
            response_score *= 0.5  # Too short penalty
        elif word_count > 500:
            response_score *= 0.8  # Too long penalty

        score += response_score
        details["response"] = f"{hits}/{len(key_points)} key points, {word_count} words"

    return {"score": round(min(score, 1.0), 3), "details": details}


# ─── Task Registry ────────────────────────────────────────────

GRADERS = {
    "classification": grade_classification,
    "routing": grade_routing,
    "full_triage": grade_full_triage,
}

TASK_DEFINITIONS = [
    {
        "task_id": "classification",
        "name": "Email Classification",
        "difficulty": "easy",
        "description": "Classify email priority (low/medium/high/critical) and category (bug_report/feature_request/billing/account_access/general_inquiry/spam)",
        "required_fields": ["priority", "category"],
        "max_steps": 1,
        "grader": "classification",
    },
    {
        "task_id": "routing",
        "name": "Email Routing",
        "difficulty": "medium",
        "description": "Classify email AND route to the correct department (engineering/product/billing/security/support/spam_filter)",
        "required_fields": ["priority", "category", "department"],
        "max_steps": 1,
        "grader": "routing",
    },
    {
        "task_id": "full_triage",
        "name": "Full Email Triage",
        "difficulty": "hard",
        "description": "Classify, route, AND draft an appropriate response addressing key concerns",
        "required_fields": ["priority", "category", "department", "response"],
        "max_steps": 1,
        "grader": "full_triage",
    },
]
