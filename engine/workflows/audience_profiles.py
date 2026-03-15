"""
Audience profile definitions for course content adaptation.
"""

from typing import Optional, Literal

# Type alias for audience options
AudienceType = Optional[Literal["kids", "general", "advanced"]]

# ============================================================================
# AUDIENCE PROFILES - Concise guidelines for each audience
# ============================================================================

AUDIENCE_PROFILES = {
    "kids": {
        "name": "KIDS (Ages 8-12)",
        "summary": "Simple, fun, story-like content for children",
        "rules": [
            "Use ONLY simple, everyday words a 10-year-old knows",
            "ONE idea per paragraph, MAX 3 short sentences",
            "Explain like telling a story: 'Imagine you are...'",
            "Use comparisons to games, school, pets, food, sports",
            "Be fun and exciting - 'Here's the cool part!'",
            "Ask questions: 'Have you ever wondered...?'",
            "Keep titles under 6 words, make them sound like adventures"
        ],
        "activities_rules": [
            "Make it feel like a game, not a test",
            "Only 2-3 answer choices",
            "Very short questions (one sentence)"
        ]
    },
    
    "general": {
        "name": "GENERAL AUDIENCE (No Prerequisites)",
        "summary": "Clear, accessible content for anyone with zero background",
        "rules": [
            "Assume ZERO prior knowledge - explain everything",
            "Avoid ALL jargon - use plain language",
            "Define technical terms immediately in parentheses",
            "Short paragraphs (3-4 sentences max)",
            "Connect concepts to everyday life situations",
            "Warm and helpful tone, like a patient friend",
            "Never condescending - respect reader intelligence"
        ],
        "activities_rules": [
            "Clear, straightforward questions",
            "3-4 answer options",
            "Practical, real-world scenarios"
        ]
    },
    
    "advanced": {
        "name": "ADVANCED / TECHNICAL AUDIENCE",
        "summary": "Professional-level content for experts and academics",
        "rules": [
            "Use precise technical terminology confidently",
            "Assume familiarity with foundational concepts",
            "Include nuances, trade-offs, and edge cases",
            "Reference research and authoritative sources",
            "Analytical tone - encourage critical thinking",
            "Dense, information-rich content is OK",
            "Include industry practices and current debates"
        ],
        "activities_rules": [
            "Complex, multi-step problems",
            "4-5 options with nuanced distractors",
            "Analysis and synthesis questions"
        ]
    }
}


def build_audience_guidelines(audience: AudienceType, context: str = "general") -> str:
    """
    Build the audience guidelines block to inject into prompts.
    
    Args:
        audience: The target audience (None, "kids", "general", "advanced")
        context: The context - "index", "theory", or "activities"
    
    Returns:
        Formatted guidelines string, or empty string if audience is None
    """
    if audience is None:
        return ""
    
    profile = AUDIENCE_PROFILES.get(audience)
    if profile is None:
        return ""
    
    # Build concise guidelines with prominent audience name
    lines = [
        "=====================================",
        f"TARGET AUDIENCE: {profile['name']}",
        "=====================================",
        f"({profile['summary']})",
        "",
        "ADAPT ALL CONTENT TO THIS AUDIENCE:",
    ]
    
    for rule in profile["rules"]:
        lines.append(f"- {rule}")
    
    # Add activity-specific rules for activities context
    if context == "activities" and "activities_rules" in profile:
        lines.append("")
        lines.append("ACTIVITY RULES:")
        for rule in profile["activities_rules"]:
            lines.append(f"- {rule}")
    
    return "\n".join(lines)
