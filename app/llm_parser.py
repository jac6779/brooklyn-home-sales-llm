import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def parse_property_details(user_prompt: str):
    system_message = """
    Extract property features from the user prompt into JSON.

    Return only these keys:
    - neighborhood
    - building_class_category
    - gross_sqft
    - year_built
    - distance_to_station
    - within_half_mi

    Rules:
    - neighborhood must be lowercase with underscores
    - building_class_category must be lowercase with underscores
    - gross_sqft must be a number
    - year_built must be a number
    - distance_to_station must be a number
    - within_half_mi must be 0 or 1
    - if a value is missing, use null
    - do not include extra keys
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