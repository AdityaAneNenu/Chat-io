# backend/agents/struct_agent.py

import re

def structure_data(content):
    if not content:
        return {"main_points": []}

    # Try to extract bullet points or numbered items
    points = re.findall(r"(?:[-*•]\s+|\d+\.\s+)(.+)", content)

    # If no bullet-style found, fallback to splitting by sentences
    if not points:
        points = re.split(r"[.?!]\s+", content)
    
    # Clean, trim, and remove empties
    cleaned_points = [p.strip() for p in points if p.strip()]
    
    # Optional: limit to top 5
    return {
        "main_points": cleaned_points[:5]
    }
