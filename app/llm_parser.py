import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def load_schema_reference() -> str:
    reference_path = Path(__file__).parent / "rag_context" / "schema_reference.txt"
    return reference_path.read_text(encoding="utf-8")

def parse_property_details(user_prompt: str):
    schema_reference = load_schema_reference()
    
    system_message = f"""
    Extract and normalize property features from the user prompt into JSON, using the schema reference below.
    
    SCHEMA REFERENCE:
    {schema_reference}

    Return only these keys:
    - neighborhood
    - building_class_category
    - gross_sqft
    - year_built
    - distance_to_station
    - within_half_mi
    
    Rules:
    - neighborhood must be one of the allowed neighborhood values from the schema reference.
    - building_class_category must be one of the allowed building_class_category values from the schema reference.
    - Use the building class synonym mappings to convert user-friendly terms into model-ready values.
    - For example, if the user says "home", "house", or "single-family home", map it to "one_family_dwellings" when the context indicates a single-family property.
    - Do not return "home", "house", "dwellings", or any free-text building type.
    - gross_sqft must be a number.
    - year_built must be a number.
    - distance_to_station must be a number.
    - within_half_mi must be 0 or 1.
    - if a value is missing or cannot be confidently mapped, use null.
    - do not include extra keys.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)