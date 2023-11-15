"""
Just mockups of an actual databases / APIs for demo purposes
"""
from typing import Any


def query_tank_specs(model_id: str) -> dict[str, Any]:
    """ mock up for demo purposes """
    # ES-48-1080-H 1080 | Gallon Carbon Steel
    print(f'query_tank_specs: {model_id}')
    sample_specs = {
        "volume_gallons": 30,
        "size": "48\" OD x 148\" OA",
        "design_pressure": "160 psig",
        "orientation": "Horizontal",
        "contents": "Hot Water",
        "service": "Storage",
        "material": "Carbon Steel",
        "empty_weight": "1600 lbs",
        "lining": "Epoxy",
        "exterior": "Prime",
        "insulation": None,
        "code": "ASME Section IV, HLW",
        "model": model_id
    }
    return sample_specs


# creating OpenAPI specification for the query_tank_specs function
def query_tank_specs_tool() -> dict[str, Any]:
    """ :returns function spec wrapped to include as is in tools """
    open_api_specs = {
        "name": "query_tank_specs",
        "description": "Get the Specification for the Storage Tank",
        "parameters": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string",
                             "description": "ID of the Storage Tank, e.g. ES-48-1080-H"},
            },
            "required": ["model_id"]
        }
    }
    tool_obj = {
        "type": "function",
        "function": open_api_specs
    }
    return tool_obj
