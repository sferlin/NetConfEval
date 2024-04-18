import json


def step_1_input_formatter(requirement: str) -> dict:
    return {"input": "Here are my requirements: " + requirement}


def step_1_output_formatter(output: any) -> dict:
    output = output if type(output) == dict else json.loads(output)
    return {"extend": False, "args": [output], "metadata": {}}


def step_1_conflict_formatter(output: any) -> dict:
    output = output if type(output) == dict or type(output) == str else json.loads(output)
    return {"extend": False, "args": [output], "metadata": {}}


def step_2_input_formatter(data: dict) -> dict:
    return {"input": data["input"] if data["extend"] else ""}


def step_2_output_formatter(data: any) -> dict:
    return {"extend": False}
