import json
import os
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.tools import fetch_country_data

#  Lazy loader 
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")                 
        if not api_key:
            raise ValueError("GROQ_API_KEY not found! Check your .env file.")
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",                   
            temperature=0,
            api_key=api_key
        )
    return _llm


# NODE 1: Intent & Field Identification

async def identify_intent(state: AgentState) -> AgentState:
    system_prompt = """You are an intent parser for a country information service.

Given a user question, extract:
1. "country": the country name mentioned (string or null)
2. "fields": list of requested data fields from this allowed set:
   ["population", "capital", "currency", "languages", "area", "region", "flag", "timezone"]

Rules:
- If no country is mentioned, set country to null
- If the question is not about a country at all, set both to null
- If "all" or "everything" is requested, return all fields
- Be liberal: "money" → currency, "people" → population, "size" → area

Respond ONLY with valid JSON, no markdown, no explanation.
Example: {"country": "Germany", "fields": ["population", "capital"]}"""

    user_msg = f'User question: "{state["user_question"]}"'

    try:
        response = await get_llm().ainvoke([          
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])

        parsed = json.loads(response.content.strip())
        country = parsed.get("country")
        fields = parsed.get("fields", [])

        if not country:
            return {
                **state,
                "country_name": None,
                "requested_fields": [],
                "intent_error": "I couldn't identify a country in your question. Please mention a specific country."
            }

        return {
            **state,
            "country_name": country,
            "requested_fields": fields if fields else ["capital", "population"],
            "intent_error": None
        }

    except json.JSONDecodeError:
        return {
            **state,
            "country_name": None,
            "requested_fields": [],
            "intent_error": "Failed to parse your question. Please rephrase it."
        }



# NODE 2: Tool Invocation

async def invoke_tool(state: AgentState) -> AgentState:
    if state.get("intent_error"):
        return state

    country_name = state["country_name"]
    data, error = await fetch_country_data(country_name)

    if error:
        return {**state, "raw_country_data": None, "api_error": error}

    return {**state, "raw_country_data": data, "api_error": None}



# NODE 3: Answer Synthesis

async def synthesize_answer(state: AgentState) -> AgentState:
    if state.get("intent_error"):
        return {**state, "final_answer": state["intent_error"]}

    if state.get("api_error"):
        return {**state, "final_answer": state["api_error"]}

    country_data = state["raw_country_data"]
    fields = state["requested_fields"]
    question = state["user_question"]

    relevant_data = _extract_relevant_data(country_data, fields)

    system_prompt = """You are a helpful country information assistant.
Answer the user's question using ONLY the provided data.
- Be concise and direct
- Format numbers with commas (e.g., 83,000,000)
- If a specific field is missing from the data, say "that information is not available"
- Do not make up any information
- Keep answers to 1-3 sentences unless multiple fields are requested"""

    user_msg = f"""User question: "{question}"

Available country data:
{json.dumps(relevant_data, indent=2)}

Answer the question based only on this data."""

    try:
        response = await get_llm().ainvoke([          
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])
        return {**state, "final_answer": response.content.strip()}

    except Exception as e:
        return {**state, "final_answer": f"I found the data but couldn't format the answer: {str(e)}"}


def _extract_relevant_data(data: dict, fields: list) -> dict:
    result = {"country": data.get("name", {}).get("common", "Unknown")}

    field_map = {
        "capital":    lambda d: d.get("capital", []),
        "population": lambda d: d.get("population"),
        "currency":   lambda d: {
            code: info.get("name")
            for code, info in (d.get("currencies") or {}).items()
        },
        "languages":  lambda d: list((d.get("languages") or {}).values()),
        "area":       lambda d: d.get("area"),
        "region":     lambda d: f"{d.get('region', '')} / {d.get('subregion', '')}",
        "flag":       lambda d: d.get("flags", {}).get("png"),
        "timezone":   lambda d: d.get("timezones", []),
    }

    for field in fields:
        if field in field_map:
            try:
                result[field] = field_map[field](data)
            except Exception:
                result[field] = None

    return result