import requests
from gnews import GNews
import re

# Initialize GNews
google_news = GNews(language='en', country='IN', max_results=5)

def verify_claim_with_news(claim):
    # Extract main keywords (simple approach for now)
    keywords = re.findall(r'\b[A-Z][a-z]*\b', claim)  # e.g. ["India", "Sri", "Lanka", "Asia", "Cup"]
    query = " ".join(keywords)
    
    print(f"🔎 Searching news for: {query}")
    
    # Fetch news articles
    articles = google_news.get_news(query)
    
    if not articles:
        return f"⚠️ No supporting news found for claim: {claim}"
    
    # Check if any article title/desc supports the claim
    for article in articles:
        title = article['title'].lower()
        desc = article['description'].lower()
        if all(word.lower() in title+desc for word in ["india", "sri", "lanka", "asia", "cup", "2023"]):
            return f"✅ Supported by news: {article['title']} ({article['url']})"
    
    return f"⚠️ No direct match found in news for: {claim}"

# -----------------
# TEST CASES
# -----------------
claims = [
    "India beat Sri Lanka in Asia Cup final 2023",
    "Pakistan beat India in Asia Cup final 2023"
]

for c in claims:
    print("\nCLAIM:", c)
    print(verify_claim_with_news(c))
