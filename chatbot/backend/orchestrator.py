from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import json
import re
import wikipedia
import concurrent.futures
from typing import List, Dict, Any
import logging

app = Flask(__name__)
CORS(app)

# Configuration
GROQ_API_KEY = ""
GEMINI_API_KEY = ""
NEWS_API_KEY = ""

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    mode = data.get("mode", "summarize")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Get knowledge from all sources in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_wiki = executor.submit(fetch_wikipedia, query)
            future_news = executor.submit(fetch_news, query)
            future_groq = executor.submit(fetch_groq_knowledge, query)
            
            wiki_data = future_wiki.result()
            news_data = future_news.result()
            groq_data = future_groq.result()

        # Combine sources
        combined_content = f"{wiki_data['content']}\n\n{news_data['content']}\n\n{groq_data['content']}"
        combined_sources = wiki_data['sources'] + news_data['sources'] + groq_data['sources']

        # Get responses from both LLMs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if mode == "compare":
                future_gemini = executor.submit(get_gemini_comparison, query, combined_content)
                future_groq = executor.submit(get_groq_response, query, combined_content, "compare")
            else:
                future_gemini = executor.submit(get_gemini_summary, query, combined_content)
                future_groq = executor.submit(get_groq_response, query, combined_content, "summarize")
            
            gemini_res = future_gemini.result()
            groq_res = future_groq.result()

        # Select best response
        best_response = select_best_response(
            groq_res, 
            gemini_res,
            query,
            mode,
            combined_sources
        )

        # Format the response before returning
        formatted_answer = best_response.get("summary", "No summary available")
        case_studies = best_response.get("structured_data", {}).get("case_studies", [])
        
        if case_studies:
            formatted_answer += "\n\nCase Studies:\n" + "\n".join([
                f"- {cs.get('language', 'N/A')}: {cs.get('title', 'No title')} – {cs.get('description', 'No description')}"
                for cs in case_studies
            ])

        return jsonify({
            "query": query,
            "mode": mode,
            "source": best_response.get("source", "Unknown"),
            "answer": formatted_answer,
            "sources": best_response.get("sources", []),
            "metrics": best_response.get("metrics", {})
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "query": query,
            "source": "Error",
            "answer": f"Could not process request: {str(e)}",
            "sources": []
        }), 500

def select_best_response(groq_res: Dict, gemini_res: Dict, query: str, mode: str, sources: List[str]) -> Dict:
    """Select the best response between Groq and Gemini outputs"""
    # Score responses based on quality metrics
    groq_score = score_response(groq_res, mode)
    gemini_score = score_response(gemini_res, mode)
    
    # Prefer Gemini for comparisons, Groq for summaries
    if mode == "compare":
        gemini_score += 0.2
    else:
        groq_score += 0.2
    
    # Select winner
    winner = gemini_res if gemini_score > groq_score else groq_res
    
    # Enhance with sources
    return {
        "query": query,
        "mode": mode,
        "source": "Gemini" if gemini_score > groq_score else "Groq",
        "summary": winner.get("summary", "No summary available"),
        "structured_data": winner.get("structured_data", {}),
        "sources": sources[:5],  # Limit to top 5 sources
        "metrics": {
            "groq_score": groq_score,
            "gemini_score": gemini_score
        }
    }

def score_response(response: Dict, mode: str) -> float:
    """Score response quality (0-1 scale)"""
    score = 0.0
    
    # Content length
    content = response.get("summary", "")
    score += min(1.0, len(content) / 500) * 0.3
    
    # Structured data quality
    struct_data = response.get("structured_data", {})
    if mode == "compare":
        points = struct_data.get("comparison_points", [])
        score += min(1.0, len(points) / 5) * 0.4
    else:
        points = struct_data.get("key_points", [])
        score += min(1.0, len(points) / 3) * 0.4
    
    # JSON validity
    try:
        json.dumps(struct_data)
        score += 0.3
    except:
        pass
    
    return score

# Data Fetching Functions
def fetch_wikipedia(query: str) -> Dict:
    try:
        summary = wikipedia.summary(query, sentences=5)
        page = wikipedia.page(query)
        return {
            "content": summary,
            "sources": [page.url]
        }
    except Exception as e:
        logger.warning(f"Wikipedia error: {str(e)}")
        return {"content": "", "sources": []}

def fetch_news(query: str) -> Dict:
    if not NEWS_API_KEY:
        return {"content": "", "sources": []}
    
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        res = requests.get(url).json()
        articles = res.get("articles", [])[:3]  # Top 3 articles
        content = " ".join(a.get("description", "") for a in articles if a.get("description"))
        sources = [a["url"] for a in articles if a.get("url")]
        return {
            "content": content,
            "sources": sources
        }
    except Exception as e:
        logger.warning(f"NewsAPI error: {str(e)}")
        return {"content": "", "sources": []}

def fetch_groq_knowledge(query: str) -> Dict:
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "Provide comprehensive information"},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return {
            "content": response.json()["choices"][0]["message"]["content"],
            "sources": [f"Groq AI: {query}"]
        }
    except Exception as e:
        logger.error(f"Groq error: {str(e)}")
        return {"content": "", "sources": []}

# LLM Processing Functions
def get_groq_response(query: str, context: str, mode: str) -> Dict:
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if mode == "compare":
            prompt = f"""Compare entities in this context:
            {context}
            
            Provide JSON with:
            - summary: overall comparison
            - comparison_points: array of attributes with values"""
        else:
            prompt = f"""Summarize this content about {query}:
            {context}
            
            Provide JSON with:
            - summary: brief overview
            - key_points: array of main points"""
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.5
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"Groq processing error: {str(e)}")
        return {"summary": "Error", "structured_data": {}}

def get_gemini_summary(query: str, context: str) -> Dict:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""Summarize this content about {query}:
                    {context}
                    
                    Provide JSON with:
                    - summary: overview
                    - key_points: array
                    - entities: array"""
                }]
            }],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        response = requests.post(url, json=payload)
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return extract_json(text)
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {"summary": "Error", "structured_data": {}}

def get_gemini_comparison(query: str, context: str) -> Dict:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"""Compare entities in this context:
                    {context}
                    
                    Provide JSON with:
                    - summary: overall comparison
                    - comparison_points: array of attributes
                    - recommendations: array"""
                }]
            }],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        response = requests.post(url, json=payload)
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return extract_json(text)
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return {"summary": "Error", "structured_data": {}}

def extract_json(text: str) -> Dict:
    """Extract JSON from LLM response text"""
    try:
        json_match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        return json.loads(text)
    except:
        return {"summary": text, "structured_data": {}}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)