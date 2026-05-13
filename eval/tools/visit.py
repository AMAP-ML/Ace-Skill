"""
Visit tool - visit webpage and extract content
Priority: crawl4ai (headless browser) > Serper Scrape API > trafilatura > Jina API
"""

import os
import json
import asyncio
import requests
from tools.base import BaseTool
from tools.tool_registry import register_tool

# --- crawl4ai (preferred: headless browser, high-quality markdown) ---
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

try:
    import trafilatura
except ImportError:
    trafilatura = None

# Serper Scrape API (uses the same key as web_search)
SERPER_API_KEY = os.environ.get("SERPAPI_KEY")
SERPER_SCRAPE_AVAILABLE = SERPER_API_KEY is not None

# Legacy Jina support (kept for backward compat)
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_AVAILABLE = JINA_API_KEY is not None


def _strip_after_think_tag(text: str):
    """Strip qwen think prefix and keep content after </think> marker."""
    marker = "\n</think>\n\n"
    think_prefix = "Thinking Process:"
    if not isinstance(text, str):
        return text
    if not text.startswith(think_prefix):
        return text
    if marker in text:
        return text.split(marker, 1)[1]
    return ""


def _is_qwen35_model(model_name: str) -> bool:
    """Return True if model name indicates qwen3.5 series."""
    return "qwen3.5" in (model_name or "").strip().lower()


@register_tool("visit")
class Visit(BaseTool):
    name = "visit"
    description = "Visit a webpage and extract its main content. Use when you have a specific URL to visit (often after getting a URL from web search or image search). Extracts and returns the main textual content of the webpage."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL of the webpage to visit (must start with http:// or https://)"
            },
            "goal": {
                "type": "string",
                "description": "What information you want to find on this page (helps focus the extraction)"
            }
        },
        "required": ["url", "goal"]
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        if not CRAWL4AI_AVAILABLE and trafilatura is None and not SERPER_SCRAPE_AVAILABLE and not JINA_AVAILABLE:
            raise ImportError(
                "At least one of crawl4ai, trafilatura, SERPAPI_KEY, or JINA_API_KEY "
                "is required for Visit tool"
            )
        
        # Configuration
        self.max_content_length = config.get('max_content_length', 5000) if config else 5000
        self.use_llm_summary = config.get('use_llm_summary', True) if config else True  # Default enabled
        self.timeout = config.get('timeout', 15) if config else 15
        
        # API configuration (reuse EXPERIENCE_* env vars from ace_skill)
        self.api_key = os.environ.get("EXPERIENCE_API_KEY")
        self.api_endpoint = os.environ.get("EXPERIENCE_END_POINT")
        self.model_name = os.environ.get("EXPERIENCE_MODEL_NAME")
        if not self.api_key or not self.api_endpoint:
            print(f"[Visit] LLM summary disabled: api_key or api_endpoint not configured")

        # Decoding parameters from args (aligned with shell script EXPERIENCE_* settings)
        args = config.get('args') if config else None
        self.temperature = getattr(args, 'experience_temperature', 0.7) if args else 0.7
        self.top_p = getattr(args, 'experience_top_p', 0.8) if args else 0.8
        self.top_k = getattr(args, 'experience_top_k', None) if args else None
        self.presence_penalty = getattr(args, 'experience_presence_penalty', None) if args else None
        self.repetition_penalty = getattr(args, 'experience_repetition_penalty', None) if args else None
    
    def call(self, params, **kwargs):
        """
        Visit webpage and extract content
        
        Args:
            params: Dictionary containing url and goal
            **kwargs: Optional llm parameters for content summary
            
        Returns:
            Extracted webpage content or summary
        """
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "Error: Invalid parameters format"
        
        url = params.get("url", "")
        goal = params.get("goal", "")
        
        if not url:
            return "Error: No URL provided"
        
        try:
            print(f"[Visit] Fetching URL: {url}")
            print(f"[Visit] Goal: {goal}")
            
            content = None

            # ---- Strategy 0: crawl4ai (headless browser, best quality) ----
            if CRAWL4AI_AVAILABLE and not content:
                print("[Visit] Trying crawl4ai (headless browser)...")
                content = self._crawl4ai_fetch(url)

            # ---- Strategy 1: Serper Scrape API (fast, reliable) ----
            if SERPER_SCRAPE_AVAILABLE and not content:
                print("[Visit] Trying Serper Scrape API...")
                content = self._serper_scrape(url)

            # ---- Strategy 2: requests + trafilatura (local, free) ----
            if not content:
                downloaded = None
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }

                try:
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    downloaded = response.text
                except Exception as e:
                    print(f"[Visit] Requests failed: {e}, trying trafilatura.fetch_url as fallback")
                    if trafilatura is not None:
                        try:
                            downloaded = trafilatura.fetch_url(url)
                        except Exception as e2:
                            print(f"[Visit] Trafilatura.fetch_url also failed: {e2}")

                if downloaded and trafilatura is not None:
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        output_format='markdown',
                        include_links=True,
                        include_tables=True,
                        favor_recall=True,
                        include_formatting=True,
                        deduplicate=True
                    )
                    if not content:
                        print("[Visit] Markdown extraction failed, trying plain text with favor_recall")
                        content = trafilatura.extract(
                            downloaded,
                            include_comments=False,
                            output_format='txt',
                            favor_recall=True,
                            deduplicate=True
                        )
                elif downloaded and trafilatura is None:
                    print("[Visit] Trafilatura not available, skipped local extraction")

            # ---- Strategy 3: Jina API (legacy fallback) ----
            if not content and JINA_AVAILABLE:
                print("[Visit] Fallback to Jina API...")
                jina_result = self._jina_readpage(url)
                if jina_result and not jina_result.startswith("[visit] Failed to read page."):
                    content = jina_result
                    print(f"[Visit] Jina API extraction successful: {len(content)} characters")
            
            # All methods failed
            if not content:
                return f"Error: No content extracted from {url} (crawl4ai, Serper, trafilatura, and Jina all failed)"
            
            print(f"[Visit] Extracted {len(content)} characters")
            
            # Limit length (avoid too long)
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n\n[Content truncated due to length...]"
            
            # If summary function is enabled, use API to summarize content
            if self.use_llm_summary and goal and self.api_key and self.api_endpoint:
                try:
                    summary = self._summarize_with_api(content, goal, url)
                    return summary
                except Exception as e:
                    print(f"[Visit] API summarization failed: {e}, returning raw content")
                    # Return raw content when failed
            
            # Otherwise return extracted content
            return f"Content from {url}:\n\nGoal: {goal}\n\n{content}"
        
        except Exception as e:
            error_msg = f"Error visiting {url}: {str(e)}"
            print(f"[Visit] {error_msg}")
            return error_msg
    
    def _summarize_with_api(self, content, goal, url):
        """
        Use API to summarize webpage content (using WebWatcher's structured prompt)
        
        Args:
            content: Webpage content
            goal: Visiting goal
            url: Webpage URL
            
        Returns:
            Summarized content
        """
        import json
        
        prompt = f"""Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction**: Identify and extract the **most relevant information** from the content. Never miss any important information. Output the **full original context** as far as possible (can be more than three paragraphs)
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judging the contribution of the information to the goal

## **Output Format**
Please respond in JSON format with the following fields:
{{
  "evidence": "Key quotes or facts from the page that are directly relevant to the goal",
  "summary": "A concise summary of how the webpage content answers or relates to the user's goal"
}}"""
        
        try:
            # Build API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes webpage content based on user goals."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 8192
            }
            if self.top_k is not None:
                payload["top_k"] = self.top_k
            if self.presence_penalty is not None:
                payload["presence_penalty"] = self.presence_penalty
            if self.repetition_penalty is not None:
                payload["repetition_penalty"] = self.repetition_penalty
            
            print(f"[Visit] Calling API to summarize content (model: {self.model_name})...")
            
            endpoint = self.api_endpoint
            if not endpoint.endswith("/chat/completions"):
                endpoint = endpoint.rstrip("/") + "/chat/completions"
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=240
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            summary_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            if _is_qwen35_model(self.model_name):
                summary_text = _strip_after_think_tag(summary_text)
            
            if not summary_text:
                raise ValueError("Empty response from API")
            
            try:
                # Extract JSON part (possibly wrapped in markdown code block)
                if '```json' in summary_text:
                    # Extract content from ```json ... ```
                    start = summary_text.find('```json') + 7
                    end = summary_text.find('```', start)
                    json_str = summary_text[start:end].strip()
                elif '```' in summary_text:
                    # Extract content from ``` ... ```
                    start = summary_text.find('```') + 3
                    end = summary_text.find('```', start)
                    json_str = summary_text[start:end].strip()
                elif summary_text.strip().startswith('{'):
                    # Directly JSON
                    json_str = summary_text.strip()
                else:
                    # Try to find first { and last }
                    left = summary_text.find('{')
                    right = summary_text.rfind('}')
                    if left != -1 and right != -1 and left < right:
                        json_str = summary_text[left:right+1]
                    else:
                        # Cannot find JSON, use raw text
                        raise ValueError("No JSON found")
                
                # Parse JSON
                parsed = json.loads(json_str)
                evidence = parsed.get('evidence', '')
                summary = parsed.get('summary', '')
                
                formatted_output = f"The useful information in {url} for user goal '{goal}' as follows:\n\n"
                if evidence:
                    formatted_output += f"**Evidence in page:**\n{evidence}\n\n"
                if summary:
                    formatted_output += f"**Summary:**\n{summary}\n\n"
                
                print(f"[Visit] API summarization successful (evidence: {len(evidence)} chars, summary: {len(summary)} chars)")
                return formatted_output
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # JSON parsing failed, use raw response
                print(f"[Visit] Failed to parse JSON response ({e}), using raw text")
                print(f"[Visit] API summarization successful ({len(summary_text)} chars)")
                return f"Summary from {url}:\n\n{summary_text}"
            
        except Exception as e:
            print(f"[Visit] API call failed: {e}")
            raise  # Re-throw exception, let outer layer catch and return raw content
    
    def _crawl4ai_fetch(self, url: str):
        """Extract webpage content using crawl4ai (headless browser).

        Returns markdown text on success, None on failure.
        Handles the async-to-sync bridge required by the synchronous call() method.
        """
        if not CRAWL4AI_AVAILABLE:
            return None

        async def _async_fetch():
            browser_cfg = BrowserConfig(headless=True, verbose=False)
            run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)
            if result and result.markdown:
                md = result.markdown
                return md.raw_markdown if hasattr(md, "raw_markdown") else str(md)
            return None

        async def _async_fetch_with_timeout():
            return await asyncio.wait_for(_async_fetch(), timeout=60)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    text = pool.submit(asyncio.run, _async_fetch_with_timeout()).result(timeout=90)
            else:
                text = asyncio.run(_async_fetch_with_timeout())

            if text:
                print(f"[Visit] crawl4ai extraction successful: {len(text)} characters")
                return text
            print("[Visit] crawl4ai returned empty content")
            return None
        except Exception as e:
            print(f"[Visit] crawl4ai request failed: {e}")
            return None

    def _serper_scrape(self, url: str):
        """Scrape webpage content via Serper Scrape API (https://scrape.serper.dev).

        Uses the same SERPAPI_KEY as the web_search tool.
        Returns extracted text on success, None on failure.
        """
        if not SERPER_SCRAPE_AVAILABLE:
            return None

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"url": url, "includeMarkdown": True}
        max_retries = 2
        timeout = 20

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://scrape.serper.dev",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    print(f"[Visit] Serper Scrape API error {resp.status_code}: {resp.text[:200]}")
                    if attempt == max_retries - 1:
                        return None
                    continue

                data = resp.json()

                # Prefer markdown, fall back to text
                text = data.get("markdown") or data.get("text") or ""
                if text:
                    print(f"[Visit] Serper Scrape successful: {len(text)} characters")
                    return text

                print("[Visit] Serper Scrape returned empty content")
                return None

            except Exception as e:
                print(f"[Visit] Serper Scrape request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
        return None

    def _jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina Reader API as fallback.
        
        Args:
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        if not JINA_AVAILABLE:
            return "[visit] Jina API not available (JINA_API_KEY not set)"
        
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        max_retries = 3
        timeout = 20
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(f"[Visit] Jina API error {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        return "[visit] Failed to read page."
            except Exception as e:
                print(f"[Visit] Jina API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
        
        return "[visit] Failed to read page."

