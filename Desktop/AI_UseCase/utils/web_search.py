import requests
from config import config

BING_API_KEY = config.BING_API_KEY if hasattr(config, "BING_API_KEY") else None
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

def web_search(query):
    """Perform simple web search via Bing API"""
    if not BING_API_KEY:
        return ""
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": 3}
    try:
        response = requests.get(BING_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        snippets = []
        for v in results.get("webPages", {}).get("value", []):
            snippets.append(v.get("snippet", ""))
        return "\n".join(snippets)
    except Exception as e:
        return f"Web search error: {str(e)}"
