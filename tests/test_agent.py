import pytest
import asyncio
from agent.graph import country_agent
from agent.state import AgentState


def make_state(question: str) -> AgentState:
    return {
        "user_question": question,
        "country_name": None,
        "requested_fields": None,
        "intent_error": None,
        "raw_country_data": None,
        "api_error": None,
        "final_answer": None,
    }


@pytest.mark.asyncio
async def test_population_question():
    result = await country_agent.ainvoke(make_state("What is the population of Germany?"))
    assert result["country_name"] == "Germany"
    assert "population" in result["requested_fields"]
    assert result["final_answer"] is not None
    await asyncio.sleep(15)   # ← wait between tests to avoid rate limit


@pytest.mark.asyncio
async def test_currency_question():
    result = await country_agent.ainvoke(make_state("What currency does Japan use?"))
    assert result["country_name"] == "Japan"
    assert "currency" in result["requested_fields"]
    assert result["final_answer"] is not None
    await asyncio.sleep(15)


@pytest.mark.asyncio
async def test_invalid_country():
    result = await country_agent.ainvoke(make_state("What is the capital of Narnia?"))
    assert result["final_answer"] is not None
    await asyncio.sleep(15)


@pytest.mark.asyncio
async def test_no_country_mentioned():
    result = await country_agent.ainvoke(make_state("What is 2 + 2?"))
    assert result["intent_error"] is not None