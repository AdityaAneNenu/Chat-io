from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import requests
import json
import re
import logging
import wikipedia
import redis
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Redis connection
class RedisCache:
    def __init__(self):
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        try:
            self.redis.ping()
            logger.info("Connected to Redis successfully")
        except redis.ConnectionError:
            logger.error("Failed to connect to Redis")
            
    def _generate_key(self, query: str, mode: str) -> str:
        """Generate consistent Redis key from query and mode"""
        return md5(f"{mode}:{query}".encode()).hexdigest()
    
    def cache_response(self, query: str, mode: str, response: dict, ttl: int = 3600) -> bool:
        """Cache a response with time-to-live in seconds"""
        key = self._generate_key(query, mode)
        try:
            self.redis.setex(
                name=key,
                time=timedelta(seconds=ttl),
                value=json.dumps(response)
            )
            return True
        except Exception as e:
            logger.error(f"Redis cache error: {str(e)}")
            return False
    
    def get_cached_response(self, query: str, mode: str) -> dict:
        """Get cached response if exists"""
        key = self._generate_key(query, mode)
        try:
            cached = self.redis.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None

# Initialize Redis cache instance
cache = RedisCache()

# API Keys (replace with your actual keys)
GROQ_API_KEY = ""
GEMINI_API_KEY = ""
NEWS_API_KEY = ""

class ResponseFormatter:
    @staticmethod
    def safe_string(value: Union[str, Dict]) -> str:
        """Safely convert value to stripped string"""
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value).strip()

    @staticmethod
    def format_comparison(response: Dict) -> Dict:
        """Format comparison response into polished markdown"""
        structured_data = response.get("structured_data", {})
        output = [f"# {ResponseFormatter.safe_string(response.get('query', 'Comparison'))}\n"]

        # Overview Section
        if summary := response.get("summary"):
            output.append(f"## Overview\n{ResponseFormatter.safe_string(summary)}\n\n")

        # Similarities Section
        if similarities := structured_data.get("similarities"):
            output.append("## Key Similarities\n")
            output.extend(f"- {ResponseFormatter.safe_string(s)}\n" for s in similarities)

        # Differences Section
        if differences := structured_data.get("differences"):
            output.append("\n## Key Differences\n")
            for diff in differences:
                feature = ResponseFormatter.safe_string(diff.get('feature', '')).replace('_', ' ').title()
                output.append(f"### {feature}\n")
                output.append(f"- **{diff.get('first', 'Option 1')}**: {ResponseFormatter.safe_string(diff.get('first_value', ''))}\n")
                output.append(f"- **{diff.get('second', 'Option 2')}**: {ResponseFormatter.safe_string(diff.get('second_value', ''))}\n\n")

        # Metrics Section
        if metrics := structured_data.get("metrics"):
            output.append("## Comparative Metrics\n")
            for metric, values in metrics.items():
                metric_name = ResponseFormatter.safe_string(metric).replace('_', ' ').title()
                output.append(f"- **{metric_name}**: ")
                if isinstance(values, dict):
                    output.append(", ".join([f"{k}: {v}" for k, v in values.items()]))
                output.append("\n")

        # Case Studies Section
        if case_studies := structured_data.get("case_studies"):
            output.append("\n## Case Studies\n")
            for case in case_studies:
                title = ResponseFormatter.safe_string(case.get('title', 'Case Study')).title()
                output.append(f"### {title}\n")
                output.append(f"{ResponseFormatter.safe_string(case.get('description', ''))}\n")
                if case_metrics := case.get("metrics"):
                    output.append("**Metrics**:\n")
                    output.extend(f"- {k.title()}: {v}\n" for k, v in case_metrics.items())

        # Future Outlook Section
        if future := structured_data.get("future"):
            output.append("\n## Future Outlook\n")
            output.extend(f"- {ResponseFormatter.safe_string(item.get('description', ''))}\n" for item in future)

        # Sources Section
        if sources := response.get("sources"):
            output.append("\n## References\n")
            output.extend(
                f"{i}. [{s.get('title', f'Source {i}')}]({s.get('url', '#')})\n"
                for i, s in enumerate(sources, 1)
            )

        return {
            "query": response.get("query", ""),
            "mode": "compare",
            "source": response.get("source", "Multiple Sources"),
            "answer": "\n".join(output).strip(),
            "structured_data": structured_data,
            "sources": sources
        }

    @staticmethod
    def format_summary(response: Dict) -> Dict:
        """Format summary response into clean markdown"""
        structured_data = response.get("structured_data", {})
        output = [f"# {ResponseFormatter.safe_string(response.get('query', 'Summary'))}\n"]

        # Overview Section
        if summary := response.get("summary"):
            output.append(f"## Overview\n{ResponseFormatter.safe_string(summary)}\n\n")

        # Key Points Section
        if key_points := structured_data.get("key_points"):
            output.append("## Key Points\n")
            output.extend(f"- {ResponseFormatter.safe_string(kp)}\n" for kp in key_points)

        # Current Status Section
        if status := structured_data.get("current_status"):
            output.append("\n## Current Status\n")
            output.append(f"{ResponseFormatter.safe_string(status)}\n")

        # Sources Section
        if sources := response.get("sources"):
            output.append("\n## References\n")
            output.extend(
                f"{i}. [{s.get('title', f'Source {i}')}]({s.get('url', '#')})\n"
                for i, s in enumerate(sources, 1)
            )

        return {
            "query": response.get("query", ""),
            "mode": "summarize",
            "source": response.get("source", "Groq"),
            "answer": "\n".join(output).strip(),
            "structured_data": structured_data,
            "sources": sources
        }

class KnowledgeSource:
    @staticmethod
    def fetch_wikipedia(entities):
        """Fetch Wikipedia content for entities"""
        content = ""
        sources = []
        
        for entity in entities:
            try:
                clean_entity = re.sub(r'[^a-zA-Z0-9 ]', '', entity).strip()
                if not clean_entity:
                    continue
                    
                page = wikipedia.page(clean_entity, auto_suggest=True)
                content += f"=== {clean_entity} ===\n{page.summary}\n\n"
                sources.append({
                    "title": f"Wikipedia: {page.title}",
                    "url": page.url,
                    "type": "encyclopedia"
                })
            except wikipedia.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0])
                    content += f"=== {e.options[0]} ===\n{page.summary}\n\n"
                    sources.append({
                        "title": f"Wikipedia: {page.title}",
                        "url": page.url,
                        "type": "encyclopedia"
                    })
                except:
                    logger.warning(f"Wikipedia disambiguation error for {entity}")
            except Exception as e:
                logger.warning(f"Wikipedia error for {entity}: {str(e)}")
        
        return {"content": content, "sources": sources}

    @staticmethod
    def fetch_news(entities):
        """Fetch recent news articles"""
        content = ""
        sources = []
        
        if not NEWS_API_KEY or NEWS_API_KEY == "your_news_api_key":
            return {"content": "", "sources": []}
        
        for entity in entities:
            try:
                url = (f"https://newsapi.org/v2/everything?"
                      f"q={entity}"
                      f"&pageSize=3&sortBy=relevancy&apiKey={NEWS_API_KEY}")
                
                res = requests.get(url, timeout=10).json()
                for article in res.get("articles", [])[:3]:
                    if article.get("url"):
                        content += f"**{article.get('title','')}**\n{article.get('description','')}\n\n"
                        sources.append({
                            "url": article["url"],
                            "title": article.get("title", "News article"),
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "date": article.get("publishedAt", "")[:10],
                            "type": "news"
                        })
            except Exception as e:
                logger.warning(f"NewsAPI error: {str(e)}")
        
        return {"content": content, "sources": sources}

    @staticmethod
    def fetch_scholar_sources(query):
        """Fetch academic sources using Google Scholar"""
        sources = []
        try:
            # Mock response for now
            sources.append({
                "title": f"Academic Paper on {query}",
                "url": f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
                "authors": "Various Authors",
                "year": "2024",
                "type": "academic"
            })
        except Exception as e:
            logger.warning(f"Google Scholar error: {str(e)}")
        return sources

    @staticmethod
    def fetch_academic_sources(entities):
        """Fetch scholarly papers using Google Scholar as primary source"""
        query = " AND ".join(entities)
        sources = KnowledgeSource.fetch_scholar_sources(query)
        
        return {
            "content": "\n".join([f"Title: {s['title']}\nAuthors: {s['authors']}\nYear: {s['year']}\n" 
                      for s in sources[:3]]),
            "sources": sources[:3]
        }

    @staticmethod
    def get_knowledge_content(query):
        """Get comprehensive information using Groq"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        prompt = f"""Provide a thorough, balanced analysis about: {query}
    Cover these aspects:
    1. Historical context and background
    2. Key facts and current status
    3. Different perspectives on the topic
    4. Relevant statistics and data
    5. Future implications

    Present information in a neutral, factual manner."""
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a knowledgeable research assistant"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 3000
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise Exception(f"Knowledge gathering failed: {str(e)}")

class AIModels:
    @staticmethod
    def get_gemini_comparison(entities, context):
        """Get comparison from Gemini"""
        prompt = f"""Compare and contrast {entities[0]} and {entities[1]} using this context:
    {context}

    Provide:
    1. Key similarities and differences
    2. Historical context
    3. Current status
    4. Future outlook

    Format response as JSON with:
    - overview (summary paragraph)
    - similarities (array)
    - differences (array of objects with feature, first_value, second_value)
    - key_dates (array of important events)
    - sources (array of recommended readings)"""

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.4}
                },
                timeout=40
            )
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return Utils.extract_structured_response(text)
        except Exception as e:
            logger.error(f"Gemini comparison failed: {str(e)}")
            return {}

    @staticmethod
    def get_groq_comparison(entities, context):
        """Get comparison from Groq"""
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        prompt = f"""Analyze and compare {entities[0]} and {entities[1]} using:
    {context}

    Provide:
    1. Comparative analysis
    2. Key metrics (as key-value pairs)
    3. Case studies (with title, description, and metrics)
    4. Future projections

    Format response as JSON with:
    - comparison_overview
    - key_metrics (object)
    - case_studies (array of objects)
    - future_scenarios (array of objects)"""

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a comparative analysis expert"},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.4,
                    "max_tokens": 2000
                },
                timeout=35
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"Groq comparison failed: {str(e)}")
            return {}

class Utils:
    @staticmethod
    def extract_structured_response(text):
        """Extract JSON from response text"""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                return json.loads(text[start:end])
            return {}
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            return {}

    @staticmethod
    def extract_comparison_entities(query, content):
        """Extract entities to compare from query"""
        patterns = [
            r"(?i)(?:compare|contrast|difference between)\s+(.+?)\s+and\s+(.+)",
            r"(?i)(.+?)\s+(?:vs|versus|compared to)\s+(.+)",
            r"(?i)(?:which|what).*?(?:better|worse|superior)\s+(.+?)\s+or\s+(.+)"
        ]
        
        clean_query = re.sub(r"[?,.!]", "", query)
        for pattern in patterns:
            if matches := re.search(pattern, clean_query):
                return [re.sub(r'[^a-zA-Z0-9 ]', '', m).strip().lower() for m in matches.groups()]
        
        noun_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
        return list(dict.fromkeys([np.lower() for np in noun_phrases if len(np) > 3]))[:2]

    @staticmethod
    def refine_responses(gemini_res, groq_res):
        """Combine best parts from both responses"""
        combined = {
            "query": f"{gemini_res.get('title', 'Comparison')} vs {groq_res.get('title', 'Comparison')}",
            "mode": "compare",
            "source": "Gemini & Groq Synthesis",
            "summary": gemini_res.get("overview", "") or groq_res.get("comparison_overview", ""),
            "structured_data": {
                "similarities": gemini_res.get("similarities", []),
                "differences": gemini_res.get("differences", []),
                "metrics": groq_res.get("key_metrics", {}),
                "case_studies": groq_res.get("case_studies", []),
                "future": groq_res.get("future_scenarios", [])
            },
            "sources": gemini_res.get("sources", []) + groq_res.get("sources", [])
        }
        return combined

class QueryProcessor:
    @staticmethod
    def process_query(query, mode):
        """Main processing function that handles both modes"""
        knowledge = KnowledgeSource.get_knowledge_content(query)
        
        if mode == "summarize":
            return QueryProcessor.handle_summarization(query, knowledge)
        elif mode == "compare":
            return QueryProcessor.handle_comparison(query, knowledge)
        else:
            raise ValueError("Unsupported mode")

    @staticmethod
    def handle_summarization(query, knowledge):
        """Generate a well-structured summary for any topic"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        prompt = f"""Create a comprehensive summary about: {query}
    Using this context:
    {knowledge}

    Structure your response with:
    1. Overview (1 paragraph)
    2. Key points (3-5 bullet points)
    3. Important context (1 paragraph)
    4. Current status (if applicable)
    5. Further reading suggestions

    Format as JSON with these fields:
    - summary (the overview)
    - key_points (array of strings)
    - current_status (text)
    - sources (array of suggested sources)"""
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a professional summarizer"},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.4,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise Exception("Invalid API response format")
                
            content = result["choices"][0]["message"]["content"]
            
            try:
                json_data = json.loads(content) if isinstance(content, str) else content
                return {
                    "query": query,
                    "mode": "summarize",
                    "source": "Groq",
                    "summary": json_data.get("summary", ""),
                    "structured_data": {
                        "key_points": json_data.get("key_points", []),
                        "current_status": json_data.get("current_status", ""),
                        "suggested_sources": json_data.get("sources", [])
                    },
                    "sources": QueryProcessor.get_sources(query)
                }
            except json.JSONDecodeError:
                return {
                    "query": query,
                    "mode": "summarize",
                    "source": "Groq",
                    "summary": content,
                    "structured_data": {},
                    "sources": QueryProcessor.get_sources(query)
                }
                
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")

    @staticmethod
    def handle_comparison(query, knowledge):
        """Handle comparisons between entities"""
        entities = Utils.extract_comparison_entities(query, knowledge)
        
        if len(entities) < 2:
            entities = ["democracy", "monarchy"]  # Default comparison
        
        # Gather comprehensive context
        wiki_data = KnowledgeSource.fetch_wikipedia(entities)
        news_data = KnowledgeSource.fetch_news(entities)
        academic_data = KnowledgeSource.fetch_academic_sources(entities)
        
        combined_context = f"""
        GROQ KNOWLEDGE BASE:
        {knowledge}
        
        WIKIPEDIA CONTENT:
        {wiki_data['content']}
        
        NEWS CONTEXT:
        {news_data['content']}
        
        ACADEMIC RESEARCH:
        {academic_data['content']}
        """
        
        # Get responses from both LLMs in parallel
        with ThreadPoolExecutor() as executor:
            gemini_future = executor.submit(AIModels.get_gemini_comparison, entities, combined_context)
            groq_future = executor.submit(AIModels.get_groq_comparison, entities, combined_context)
            
            gemini_res = gemini_future.result()
            groq_res = groq_future.result()
        
        # Combine and select best response
        final_response = Utils.refine_responses(gemini_res, groq_res)
        final_response["sources"] = (
            wiki_data["sources"] + 
            news_data["sources"] + 
            academic_data["sources"]
        )
        
        return final_response

    @staticmethod
    def get_sources(query):
        """Gather relevant sources for any query"""
        sources = []
        
        try:
            search_results = wikipedia.search(query, results=2)
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=True)
                    sources.append({
                        "title": f"Wikipedia: {page.title}",
                        "url": page.url,
                        "type": "encyclopedia"
                    })
                except:
                    continue
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")

        if NEWS_API_KEY and NEWS_API_KEY != "your_news_api_key":
            try:
                url = (f"https://newsapi.org/v2/everything?"
                      f"q={query.replace(' ', '+')}"
                      f"&pageSize=2&sortBy=relevancy&apiKey={NEWS_API_KEY}")
                news_data = requests.get(url, timeout=10).json()
                for article in news_data.get("articles", [])[:2]:
                    if article.get("url"):
                        sources.append({
                            "title": article.get("title", "News article"),
                            "url": article["url"],
                            "type": "news",
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "date": article.get("publishedAt", "")[:10]
                        })
            except Exception as e:
                logger.warning(f"NewsAPI failed: {str(e)}")
        
        # Add academic sources from Google Scholar
        sources.extend(KnowledgeSource.fetch_scholar_sources(query)[:2])
        
        return sources

# Flask routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html")

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    mode = data.get("mode", "summarize")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Check cache first
    if cached := cache.get_cached_response(query, mode):
        logger.info(f"Returning cached response for query: '{query}'")
        return jsonify(cached)
        
    try:
        logger.info(f"Processing query: '{query}' in mode: {mode}")
        response = QueryProcessor.process_query(query, mode)
        
        # Format the response
        if mode == "compare":
            formatted_response = ResponseFormatter.format_comparison(response)
        else:
            formatted_response = ResponseFormatter.format_summary(response)
        
        # Cache the response for 1 hour (3600 seconds)
        cache.cache_response(query, mode, formatted_response)
            
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            "query": query,
            "mode": mode,
            "source": "Error",
            "answer": f"Could not process your request: {str(e)}",
            "structured_data": {},
            "sources": []
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)