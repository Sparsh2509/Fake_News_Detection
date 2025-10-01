import requests
import re
import spacy

# ----------- CONFIG ------------
API_KEY = "6a1a544b-563d-441c-9850-7eba37e4b6d9"  # Get it from https://cricketdata.org
BASE_URL = "https://api.cricapi.com/v1"
# -------------------------------

# Load spaCy English model (for entity extraction)
nlp = spacy.load("en_core_web_sm")

def extract_entities(claim):
    """Extract teams, year, and tournament info from news sentence"""
    doc = nlp(claim)
    teams = []
    year = None

    for ent in doc.ents:
        if ent.label_ == "DATE" and ent.text.isdigit():
            year = ent.text
        if ent.label_ == "ORG" or ent.label_ == "GPE":
            teams.append(ent.text)

    # fallback: regex year
    if not year:
        match = re.search(r"(20\d{2})", claim)
        if match:
            year = match.group(1)

    return list(set(teams)), year

def get_match_results(team1, team2, year):
    """Fetch cricket matches from API and filter by teams + year"""
    url = f"{BASE_URL}/matches"
    params = {"apikey": API_KEY, "offset": 0}
    resp = requests.get(url, params=params)

    if resp.status_code != 200:
        return None, f"API error: {resp.text}"

    data = resp.json()
    for match in data.get("data", []):
        if team1 in match["teams"] and team2 in match["teams"]:
            if year and year in match.get("date", ""):
                return match, None
    return None, "No match found for given year & teams."

def verify_claim(claim):
    """Check claim against actual match result"""
    teams, year = extract_entities(claim)

    if len(teams) < 2 or not year:
        return "Could not extract enough info from claim."

    team1, team2 = teams[0], teams[1]
    match, err = get_match_results(team1, team2, year)

    if err:
        return f"⚠️ {err}"

    if not match:
        return "⚠️ Match not found in records."

    actual_winner = match.get("winner")
    if not actual_winner:
        return "⚠️ Winner info not available yet."

    # check if claim mentions correct winner
    if actual_winner.lower() in claim.lower():
        return f"✅ Verified: {claim} (Source: CricketData.org)"
    else:
        return f"❌ Fake News!\n✅ Correct: {actual_winner} won. (Source: CricketData.org)"

# ------------------ TEST ------------------

if __name__ == "__main__":
    claim1 = "Pakistan beat India in Asia Cup final 2025"
    claim2 = "India beat Pakistan in Asia Cup final 2025"

    print("CLAIM:", claim1)
    print(verify_claim(claim1))
    print("\nCLAIM:", claim2)
    print(verify_claim(claim2))


