import os
import logging
import markdown
import json
import re
from typing import Optional, Dict, Any, List, Union
from .langchain_utils import init_groq_chain, get_clean_chain, get_chain

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LangChain instances
groq_chain = init_groq_chain()  # Singleton instance for direct Groq access
clean_chain = get_clean_chain()  # Singleton instance for clean processing
default_chain = get_chain()      # Default chain from your example

class MarkdownEnforcer:
    """Enhanced markdown cleaning and validation"""
    @staticmethod
    def parse(text: str) -> str:
        """Nuclear option for JSON removal"""
        # Remove all JSON-like structures (including nested)
        text = re.sub(r'\{[^{}]*\}', '', text, flags=re.DOTALL)
        # Also handle multi-level JSON structures
        text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        # Remove residual markdown code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove empty headers
        text = re.sub(r'^#+\s*$', '', text, flags=re.MULTILINE)
        # Clean trailing JSON at the end of the document
        text = re.sub(r'\s*\{.*$', '', text, flags=re.DOTALL)
        return text.strip()

class ResponseFormatter:
    @staticmethod
    def clean_json_response(response_text: str) -> Dict[str, Any]:
        """Extract and merge JSON data more effectively"""
        # Clean the text first by removing any obvious markdown artifacts
        clean_text = re.sub(r'```json|```', '', response_text)
        
        try:
            # Try direct JSON parse first
            data = json.loads(clean_text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        
        # Handle JSON at end of text
        json_match = re.search(r'(\{.*\})$', response_text, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                text_content = response_text[:json_match.start()].strip()
                if text_content:
                    json_data['summary'] = text_content
                return json_data
            except json.JSONDecodeError:
                pass
        
        # Original code's JSON extraction logic
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                text_before = response_text[:json_match.start()].strip()
                if text_before:
                    json_data['summary'] = text_before + "\n\n" + json_data.get('summary', '')
                return json_data
            except json.JSONDecodeError:
                pass
        
        return {'summary': response_text.strip()}

    @staticmethod
    def format_response(response: Dict[str, Any]) -> str:
        output = []
        
        # Unified title handling
        title = response.get('query') or response.get('title') or "Analysis Results"
        output.append(f"# {title}\n")
        
        # Priority content ordering
        if 'summary' in response:
            output.append(f"## Overview\n{response['summary'].strip()}\n")
        
        # Structured section handling
        sections = [
            ('current_status', 'Current Status'),
            ('key_points', 'Key Points'),
            ('metrics', 'Comparative Metrics'),
            ('case_studies', 'Case Studies'),
            ('timeline', 'Historical Timeline'),
            ('future', 'Future Outlook')
        ]
        
        for key, heading in sections:
            if key in response and response[key]:  # Check if section has content
                output.append(f"\n## {heading}")
                # Add specialized formatting for each section type
                if key == 'metrics' and isinstance(response[key], dict):
                    for k, v in response[key].items():
                        output.append(f"- **{k}**: {v}")
                elif key == 'case_studies' and isinstance(response[key], list):
                    for study in response[key]:
                        if not isinstance(study, dict):
                            continue
                        output.append(f"\n### {study.get('title', 'Case Study')}")
                        output.append(study.get('description', ''))
                        if 'metrics' in study and isinstance(study['metrics'], dict):
                            output.append("\n**Metrics**:")
                            for mk, mv in study['metrics'].items():
                                output.append(f"- {mk}: {mv}")
                elif key == 'future' and isinstance(response[key], list):
                    for projection in response[key]:
                        if not isinstance(projection, dict):
                            continue
                        output.append(f"- **{projection.get('title', 'Projection')}**: "
                                    f"{projection.get('description', '')}")
                elif key == 'key_points' and isinstance(response[key], list):
                    for point in response[key]:
                        if isinstance(point, str) and point.strip():
                            output.append(f"- {point.strip()}")
                else:
                    # Handle generic list items
                    if isinstance(response[key], list):
                        for item in response[key]:
                            if isinstance(item, str) and item.strip():
                                output.append(f"- {item}")
                    elif isinstance(response[key], str):
                        output.append(response[key])
                
                output.append("")  # Add spacing between sections
        
        # Source validation - only include relevant and valid sources
        sources = []
        for source_list in [response.get('sources', []), response.get('suggested_sources', [])]:
            if isinstance(source_list, list):
                for s in source_list:
                    if isinstance(s, str) and s.strip():
                        # Filter out unrelated or generic sources
                        if not any(keyword in s.lower() for keyword in ['musician', 'monster', 'scholar.google']):
                            sources.append(s)
        
        if sources:
            output.append("\n## References")
            # Format sources nicely
            for i, source in enumerate(set(sources), 1):
                # Check if already formatted as markdown link
                if re.match(r'^\[.*\]\(http.*\)$', source):
                    output.append(f"{i}. {source}")
                # Format as link if it starts with http
                elif source.startswith('http'):
                    # Extract domain name for display text
                    domain = re.search(r'https?://(?:www\.)?([^/]+)', source)
                    display = domain.group(1) if domain else source
                    output.append(f"{i}. [{display}]({source})")
                else:
                    output.append(f"{i}. {source}")
        
        return "\n".join(line for line in output if line.strip())

    @staticmethod
    def format_for_web(markdown_text: str) -> str:
        """Converts Markdown to HTML with proper styling"""
        # Convert markdown to HTML with extra extensions for tables and definition lists
        html = markdown.markdown(
            markdown_text,
            extensions=['tables', 'def_list', 'sane_lists', 'attr_list']
        )
        
        # Additional formatting for comparison-style content
        if "## Key Similarities" in markdown_text or "## Key Differences" in markdown_text:
            # Add comparison-specific classes
            html = html.replace('<h2>Key Similarities</h2>', '<h2 class="comparison-section similarities">Key Similarities</h2>')
            html = html.replace('<h2>Key Differences</h2>', '<h2 class="comparison-section differences">Key Differences</h2>')
            html = html.replace('<h3>Option 1', '<h3 class="comparison-option option-one">Option 1')
            html = html.replace('<h3>Option 2', '<h3 class="comparison-option option-two">Option 2')
        
        return f'<div class="formatted-response">{html}</div>'

def validate_output(text: str) -> str:
    """Military-grade output validation"""
    # Check for residual JSON/formatting
    if re.search(r'\{.*?\}|\}|\{', text, flags=re.DOTALL):
        # Instead of raising an error, attempt to clean it
        text = re.sub(r'\{.*?\}|\}|\{', '', text, flags=re.DOTALL)
        logger.warning("Cleaned residual JSON artifacts in output")
    
    # Content validation
    lines = [line for line in text.split('\n') if line.strip()]
    if len(lines) < 3:
        # Add a minimal structure if content is insufficient
        if len(lines) > 0:
            text = f"# {lines[0]}\n\n## Overview\n\nInsufficient data available."
        else:
            text = "# Analysis Results\n\n## Overview\n\nNo data available."
        logger.warning("Added minimal structure to insufficient content")
        
    if not any(line.startswith('#') for line in lines):
        # Add a header if missing
        text = f"# Analysis Results\n\n{text}"
        logger.warning("Added missing section header")
        
    return text

async def get_response(query: str, mode: str) -> str:
    """Primary async response handler with triple protection"""
    try:
        raw = await default_chain.ainvoke({"query": query, "mode": mode})
        
        # Triple protection:
        cleaned = MarkdownEnforcer().parse(raw)
        validated = validate_output(cleaned)
        
        return markdown.markdown(validated)  # Final HTML conversion
        
    except Exception as e:
        logger.error(f"Error in get_response: {str(e)}")
        return markdown.markdown(f"Error: {str(e)}")

def enforce_markdown(text: str) -> str:
    """Legacy markdown cleaner (maintained for compatibility)"""
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\{|\}', '', text)  # Clean any remaining braces
    text = re.sub(r'##+', '##', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def fallback_summarize(text: str) -> str:
    """Fallback summarization when API fails"""
    sentences = text.split('. ')
    summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text
    return f"## Overview\n\n{summary}"

async def summarize_with_groq(text: str, instruction: Optional[str] = None, mode: str = "summarize") -> str:
    """Summarize text using Groq API through LangChain with async support"""
    if not text:
        return "No content provided to summarize."
    
    try:
        max_length = 28000
        if len(text) > max_length:
            logger.info(f"Truncating text from {len(text)} to {max_length} chars")
            text = text[:max_length] + "... [content truncated]"
        
        if mode == "clean":
            response = await clean_chain.ainvoke({"query": text, "mode": mode})
            return enforce_markdown(response)
        else:
            enhanced_prompt = f"""
            {instruction or "Compare these subjects"}:
            {text}
            
            Format response STRICTLY as:
            - Start with a clear overview (2-3 paragraphs)
            - Include 3-5 key points (most important facts)
            - Add relevant metrics if available
            - Provide current status
            - Include only verified and relevant sources
            
            DO NOT include irrelevant information or generic links.
            DO NOT format using JSON - use plain text with clear section titles.
            DO NOT include sources that aren't directly related to the topic.
            """
            response = await groq_chain.ainvoke({"query": enhanced_prompt})
            
            # Extract structured data
            key_points = extract_key_points(response)
            timeline = extract_timeline(response)
            
            # Clean any JSON from the response
            response = re.sub(r'\{.*?\}', '', response, flags=re.DOTALL)
            
            formatted = ResponseFormatter().format_response({
                "summary": response,
                "query": text[:50] + "..." if len(text) > 50 else text,
                "key_points": key_points,
                "timeline": timeline
            })
            
            # Additional cleanup to ensure no JSON artifacts
            return enforce_markdown(formatted)
            
    except Exception as e:
        logger.error(f"Error with Groq API: {str(e)}")
        return fallback_summarize(text)

def extract_key_points(text: str) -> List[str]:
    """Extract key points from response text"""
    points = []
    
    # Look for bullet points or numbered lists
    matches = re.findall(r'(?:^|\n)[-*•]?\s*(.+?)(?=\n|$)', text)
    if matches:
        for match in matches:
            clean_point = match.strip()
            # Filter out very short or empty points
            if clean_point and len(clean_point) > 10:
                points.append(clean_point)
    
    # Look for key points section
    key_section = re.search(r'(?:key points|main points|key findings)(?:\s*:)?(.*?)(?=\n\s*\n|\n\s*#|$)', 
                           text, re.IGNORECASE | re.DOTALL)
    if key_section:
        section_points = re.findall(r'[-*•]?\s*(.+?)(?=\n[-*•]|\n\n|$)', key_section.group(1))
        for point in section_points:
            clean_point = point.strip()
            if clean_point and len(clean_point) > 10 and clean_point not in points:
                points.append(clean_point)
    
    return points[:5]  # Return up to 5 key points

def extract_timeline(text: str) -> List[Dict]:
    """Extract timeline events from response text"""
    events = []
    # Improved regex to catch more date formats
    matches = re.finditer(r'(\b\d{4}\b|[A-Z][a-z]+ \d{1,2},? \d{4}|[A-Z][a-z]+ \d{4})\s*[:-]\s*(.+?)(?=\n|$)', text)
    for match in matches:
        events.append({
            "date": match.group(1).strip(),
            "description": match.group(2).strip()
        })
    return events

def process_with_groq(text: str, mode: str = "summarize") -> str:
    """Process text using Groq with different modes (sync wrapper)"""
    import asyncio
    return asyncio.run(summarize_with_groq(text, None, mode))

def process_response(raw_response: dict) -> dict:
    """
    Processes raw API response into clean formatted output.
    Guarantees no JSON artifacts in final output.
    """
    formatter = ResponseFormatter()
    
    try:
        if isinstance(raw_response.get('answer'), str):
            # First clean any obvious JSON artifacts
            raw_answer = MarkdownEnforcer.parse(raw_response['answer'])
            cleaned_data = formatter.clean_json_response(raw_answer)
        else:
            cleaned_data = raw_response
        
        # Extract and consolidate sources
        all_sources = []
        if 'sources' in raw_response and isinstance(raw_response['sources'], list):
            all_sources.extend(s for s in raw_response['sources'] if isinstance(s, str) and s.strip())
            
        if 'suggested_sources' in raw_response and isinstance(raw_response['suggested_sources'], list):
            all_sources.extend(s for s in raw_response['suggested_sources'] if isinstance(s, str) and s.strip())
            
        # Filter irrelevant sources
        filtered_sources = [s for s in all_sources if not any(
            keyword in s.lower() for keyword in ['musician', 'monster', 'google', 'scholar'])]
        
        # Get key points from structured data or extract them
        key_points = raw_response.get('structured_data', {}).get('key_points', [])
        if not key_points and 'answer' in raw_response:
            key_points = extract_key_points(raw_response['answer'])
        
        formatted_text = formatter.format_response({
            **cleaned_data,
            'query': raw_response.get('query', ''),
            'key_points': key_points,
            'sources': filtered_sources
        })
        
        # Final cleaning to remove any JSON artifacts
        formatted_text = enforce_markdown(formatted_text)
        
        return {
            "query": raw_response.get("query", ""),
            "answer": formatted_text,
            "sources": filtered_sources,
            "structured_data": raw_response.get("structured_data", {}),
            "format_type": "markdown"
        }
    except Exception as e:
        logger.error(f"Error in process_response: {str(e)}")
        # Provide a safe fallback
        return {
            "query": raw_response.get("query", "Error"),
            "answer": f"# {raw_response.get('query', 'Error')}\n\n## Overview\n\nThere was an error processing this response. Please try again.",
            "sources": [],
            "structured_data": {},
            "format_type": "markdown"
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_async():
        test_text = "The Russia-Ukraine conflict is an ongoing international conflict..."
        
        print("=== Testing get_response ===")
        print(await get_response(test_text, "summarize"))
        
        print("\n=== Testing summarize_with_groq ===")
        print(await summarize_with_groq(test_text))
    
    asyncio.run(test_async())
    
    test_response = {
        "query": "Russia Ukraine conflict",
        "answer": """{
            "summary": "The conflict began in 2014...",
            "key_points": ["Point 1", "Point 2"],
            "timeline": [
                {"date": "2014", "description": "Crimea annexation"},
                {"date": "2022", "description": "Full-scale invasion"}
            ]
        }""",
        "sources": ["source1.com", "source2.org"],
        "suggested_sources": ["source3.edu"],
        "structured_data": {
            "key_points": ["Point 1", "Point 2"],
            "entities": ["Russia", "Ukraine"]
        }
    }
    print("\n=== Testing process_response ===")
    print(process_response(test_response))