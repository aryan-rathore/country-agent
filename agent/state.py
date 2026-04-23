from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    # Input from the user
    user_question: str

    # Output from Node 1: Intent identification
    country_name: Optional[str]
    requested_fields: Optional[List[str]]   
    intent_error: Optional[str]           

    # Output from Node 2: Tool/API call
    raw_country_data: Optional[dict]
    api_error: Optional[str]

    # Output from Node 3: Final answer
    final_answer: Optional[str]