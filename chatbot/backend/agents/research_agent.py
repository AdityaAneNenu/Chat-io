import wikipedia
import requests
import os
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def fetch_web_search(query: str) -> Dict[str, str]:
    """Fetch information using Groq's web search capabilities"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Provide comprehensive information about: {query}
        Include key facts, important details, and relevant context.
        Structure your response with clear paragraphs."""
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return {
                "content": response.json()["choices"][0]["message"]["content"],
                "source": [f"Groq AI generated content for: {query}"]
            }
        else:
            logger.error(f"Groq API error: {response.text}")
            return {"content": "", "source": []}
            
    except Exception as e:
        logger.error(f"Error with Groq API: {str(e)}")
        return {"content": "", "source": []}

# Keep existing wikipedia and news functions but update fetch_all_sources
def fetch_all_sources(query: str, news_api_key: Optional[str] = None) -> Dict[str, str]:
    """Combine all available sources"""
    groq_data = fetch_web_search(query)
    wiki_data = fetch_wikipedia(query)
    news_data = fetch_news(query, news_api_key) if news_api_key else {"content": "", "source": []}

    combined_content = f"{groq_data['content']}\n\n{wiki_data['content']}\n\n{news_data['content']}"
    combined_sources = groq_data['source'] + wiki_data['source'] + news_data['source']

    return {
        "content": combined_content.strip(),
        "sources": [s for s in combined_sources if s]  # Filter out empty sources
    }