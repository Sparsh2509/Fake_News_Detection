import requests
import re

API_KEY = "211a6306b770418e9f7bea6cb695360a47a3dcd9b328f39402a74348879e670b"  # ⬅️ replace with your CricAPI key
BASE_URL = "https://api.cricapi.com/v1"

# --- Step 1: Extract claim details ---
def extract_claim_details(claim):
    # Example claim: "India beat Pakistan in Asia Cup final 2025"
    match = re.search(r"(\w+)\s+beat\s+(\w+).+(\d{4})", claim, re.IGNORECASE)
    if match:
        team1, team2, year = match.groups()
        return team1.capitalize(), team2.capitalize(), int(year)
    return None, None, None

# --- Step 2: Check API for data ---
def check_claim_against_data(team1, team2, year):
    url = f"{BASE_URL}/matches?apikey={API_KEY}"
    response = requests.get(url).json()

    if not response.get("status") == "success":
        return None, "⚠️ API error: " + str(response.get("message"))

    for match in response.get("data", []):
        if team1 in match.get("teams", []) and team2 in match.get("teams", []):
            match_year = int(match["dateTimeGMT"][:4])
            if match_year == year:
                return match, None

    return None, "⚠️ No match found for given year & teams."

# --- Step 3: Validate claim ---
def validate_claim(claim):
    team1, team2, year = extract_claim_details(claim)

    if not team1 or not team2 or not year:
        return "❌ Could not extract claim details."

    match, error = check_claim_against_data(team1, team2, year)

    if error:
        return error

    winner = match.get("winner", "Unknown")
    if team1 == winner:
        return f"✅ TRUE: {team1} did beat {team2} in {year}."
    elif team2 == winner:
        return f"❌ FALSE: {team2} actually won in {year}."
    else:
        return f"⚠️ Match found but unclear winner. Data: {winner}"

# --- Test ---
if __name__ == "__main__":
    claim1 = "Sri Lanka beat India in Asia Cup final 2023"
    print("CLAIM:", claim1)
    print(validate_claim(claim1))

    claim2 = "India beat Pakistan in Asia Cup final 2023"
    print("\nCLAIM:", claim2)
    print(validate_claim(claim2))
