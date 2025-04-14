# backend/agents/research_agent.py

import wikipedia
import requests
import os

def fetch_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
        page = wikipedia.page(query)
        return {"content": summary, "source": [page.url]}
    except Exception:
        return {"content": "", "source": []}

def fetch_news(query, api_key):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        res = requests.get(url).json()
        articles = res.get("articles", [])
        content = " ".join([a.get("description", "") or "" for a in articles])
        sources = [a.get("url") for a in articles if a.get("url")]
        return {"content": content, "source": sources}
    except Exception:
        return {"content": "", "source": []}

def fetch_all_sources(query, api_key):
    wiki_data = fetch_wikipedia(query)
    news_data = fetch_news(query, api_key)

    combined_content = f"{wiki_data['content']}\n\n{news_data['content']}"
    combined_sources = wiki_data['source'] + news_data['source']

    return {
        "content": combined_content.strip(),
        "sources": combined_sources
    }
