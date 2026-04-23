import httpx
from typing import Optional

REST_COUNTRIES_BASE = "https://restcountries.com/v3.1"

# Only fetch the fields we actually need (faster, lighter)
FIELDS_FILTER = "name,capital,population,currencies,languages,area,flags,region,subregion,timezones"

async def fetch_country_data(country_name: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Fetch country data from REST Countries API.
    Returns (data, error). One of them will always be None.
    """
    url = f"{REST_COUNTRIES_BASE}/name/{country_name.strip()}"
    params = {"fields": FIELDS_FILTER}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)

        if response.status_code == 404:
            return None, f"Country '{country_name}' was not found. Please check the spelling."

        if response.status_code != 200:
            return None, f"API returned status {response.status_code}. Please try again."

        data = response.json()

        # API returns a list — take the best 
        best_match = _pick_best_match(data, country_name)
        return best_match, None

    except httpx.TimeoutException:
        return None, "The country data service timed out. Please try again."
    except httpx.RequestError as e:
        return None, f"Network error while fetching country data: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def _pick_best_match(results: list, query: str) -> dict:
    """Prefer exact common name match, otherwise return first result."""
    query_lower = query.lower()
    for country in results:
        common_name = country.get("name", {}).get("common", "").lower()
        if common_name == query_lower:
            return country
    return results[0]  # fallback to first result