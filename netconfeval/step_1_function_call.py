import argparse
import json
import os
import re
import sys
import time
from json import JSONDecodeError

from deepdiff import DeepDiff
from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.common.model_configs import model_configurations, get_model_instance
from netconfeval.common.utils import *


def add_reachability(formal_specification: dict[str, dict], source: str, prefix: str) -> None:
    if "reachability" not in formal_specification:
        formal_specification["reachability"] = {}
    if source not in formal_specification["reachability"]:
        formal_specification["reachability"][source] = []
    if prefix not in formal_specification["reachability"][source]:
        formal_specification["reachability"][source].append(prefix)


def add_waypoint(formal_specification: dict[str, dict], source: str, prefix: str, waypoints: list[str]) -> None:
    if "waypoint" not in formal_specification:
        formal_specification["waypoint"] = {}
    src_pre = f"({source},{prefix})"
    if src_pre not in formal_specification["waypoint"]:
        formal_specification["waypoint"][src_pre] = []
    formal_specification["waypoint"][src_pre].extend(
        waypoint for waypoint in waypoints if
        waypoint not in formal_specification["waypoint"][src_pre])


def add_load_balance(formal_specification: dict[str, dict], source: str, prefix: str, num: int) -> None:
    if "loadbalancing" not in formal_specification:
        formal_specification["loadbalancing"] = {}
    src_pre = f"({source},{prefix})"
    if src_pre not in formal_specification["loadbalancing"]:
        formal_specification["loadbalancing"][src_pre] = num


def split_parameters(param_string: str) -> list:
    # Regex pattern to split parameters, avoiding splitting inside quotes or brackets
    pattern_split_params = r"(?:[^,\"'\[\]]+|\"[^\"]*\"|'[^']*'|\[[^\]]*\])"
    return re.findall(pattern_split_params, param_string)


def clean_parameters(params: list) -> list:
    # Trim whitespace and filter out empty strings
    return [param.strip() for param in params if param.strip()]


def parse_functions(result, funs):
    pattern_function_call = r"(\w+)\(([^)]+)\)"
    function_calls = re.findall(pattern_function_call, funs)

    current_module = globals()
    for function_name, param_string in function_calls:
        try:
            parameters = []
            if function_name == "add_reachability":
                pattern = r'"(.*?)"'
                parameters = re.findall(pattern, param_string)
            elif function_name == "add_waypoint":
                pattern = r'"(.*?)"'
                parameters = re.findall(pattern, param_string)
                parameters[2] = parameters[2].strip('][').split(', ')
            elif function_name == "add_load_balance":
                pattern = r'"(.*?)",\s*"(.*?)",\s*(\d+)'
                param = re.findall(pattern, param_string)
                parameters.append(param[0][0])
                parameters.append(param[0][1])
                parameters.append(int(param[0][2]))
            else:
                continue

            # Check if the function exists
            if function_name in current_module:
                # Get the function by name
                function = current_module[function_name]
                # Call the function with unpacked parameters
                function(result, *parameters)
            else:
                print(f"Function {function_name} not found.")
        except Exception:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)
    parser.add_argument('--n_runs', type=int, required=False, default=5)
    parser.add_argument("--policy_file", type=str, required=False,
                        default=os.path.join("..", "assets", "step_1_policies.csv"))
    parser.add_argument('--batch_size', type=int, nargs="+", required=False,
                        default=[1, 2, 5, 10, 20, 25, 50, 100])
    parser.add_argument('--policy_types', choices=["reachability", "waypoint", "loadbalancing"],
                        required=False, nargs='+', default=["reachability"])
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "results_function_call")
    )
    parser.add_argument('--adhoc', action='store_true', required=False)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    policy_types = SortedSet(args.policy_types)
    if "reachability" not in policy_types:
        logging.error("`reachability` is not in policy_types! Aborting...")
        exit(1)
    elif "loadbalancing" in policy_types and "waypoint" not in policy_types:
        logging.error("You cannot require for `loadbalancing` without `waypoint`! Aborting...")
        exit(1)

    if args.adhoc:
        from netconfeval.prompts.step_1_adhoc_function import SETUP_PROMPT, REACHABILITY_FUNCTION, \
            WAYPOINT_FUNCTION, LOADBALANCING_FUNCTION, ASK_FOR_RESULT_PROMPT

    os.makedirs(args.results_path, exist_ok=True)

    results_time = time.strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(
        os.path.abspath(
            os.path.join(
                args.results_path,
                f"log-{args.model}-{'adhoc' if args.adhoc else 'native'}-{'_'.join(policy_types)}-function-{results_time}.log"
            )
        )
    )
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.WARNING)
    logging.root.addHandler(file_handler)

    dataset = load_csv(args.policy_file, policy_types)

    model_type = model_configurations[args.model]['type']
    if args.adhoc:
        llm_step_1 = get_model_instance(args.model)
    elif model_type == 'HF':
        raise Exception(f"Native function calling not supported on type `{model_type}`!")

    n_policy_types = len(policy_types)
    max_n_requirements = max(args.batch_size) * n_policy_types
    w = None

    filename = f"result-{args.model}-{'adhoc' if args.adhoc else 'native'}-{'_'.join(policy_types)}-function-{results_time}.csv"

    with open(os.path.join(args.results_path, filename), 'w') as f:
        for it in range(0, args.n_runs):
            logging.info(f"Performing iteration n. {it + 1}...")
            samples = pick_sample(max_n_requirements, dataset, it, policy_types)

            for batch_size in args.batch_size:
                logging.info(f"Performing experiment "
                             f"with {batch_size * n_policy_types} batch size (iteration n. {it + 1})...")
                chunk_samples = list(chunk_list(samples, batch_size * n_policy_types))

                for i, sample in enumerate(chunk_samples):
                    logging.info(f"Performing experiment with {batch_size * n_policy_types} "
                                 f"batch size on chunk {i} (iteration n. {it + 1})...")

                    result_row = {
                        'model_error': '',
                        'format_error': '',
                        'batch_size': batch_size,
                        'n_policy_types': n_policy_types,
                        'max_n_requirements': max_n_requirements,
                        'iteration': it,
                        'chunk': i,
                        'time': 0,
                        'total': 0,
                        'success': 0,
                        'fail': 0,
                        'wrong': 0,
                        'accuracy': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_cost': 0,
                        'diff': '',
                    }

                    if w is None:
                        w = csv.DictWriter(f, result_row.keys())
                        w.writeheader()

                    expected_spec = transform_sample_to_expected(sample)
                    human_language = convert_to_human_language(sample)

                    logging.warning(f"==== RUN #{it + 1} (CHUNK #{i + 1}) - BATCH: {batch_size}*{n_policy_types} ====")
                    logging.warning("Expected Result: " + json.dumps(expected_spec, indent=4))
                    logging.warning("Human Translation: " + " ".join(human_language))

                    skip_compare = False
                    start_time = time.time()

                    result = {}

                    if not args.adhoc:
                        messages = [
                            {
                                "role": "system",
                                "content": "Behave as a network operator."
                                           "User will input network requirements in natural language, "
                                           "Your task is to translate the network requirements into multiple function calls."

                            },
                            {
                                "role": "user",
                                "content": "The requirements are as below:\n" + ' '.join(human_language)
                            }
                        ]

                        tools = []
                        available_functions = {}
                        if "reachability" in policy_types:
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": "add_reachability",
                                    "description": "Given location l1 and prefix p1, reachability means that l1 can send packets to p1 directly or through other locations."
                                                   "The function adds a reachability requirement to the formal specification.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "source": {
                                                "type": "string",
                                                "description": "The source of location reachability.",
                                            },
                                            "prefix": {
                                                "type": "string",
                                                "description": "The destination prefix.",
                                            },
                                        },
                                        "required": ["source", "prefix"],
                                    },
                                },
                            })
                            available_functions["add_reachability"] = add_reachability

                        if "waypoint" in policy_types:
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": "add_waypoint",
                                    "description": "Given location l1 and destination prefix p1, waypoint means that if l1 wants to reach p1, the traffic path should include a list of locations."
                                                   "The function adds a waypoint requirement to the formal specification.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "source": {
                                                "type": "string",
                                                "description": "The source location of waypoint.",
                                            },
                                            "prefix": {
                                                "type": "string",
                                                "description": "The destination prefix.",
                                            },
                                            "waypoints": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "A list of waypoint locations to include.",
                                            },
                                        },
                                        "required": ["source", "prefix", "waypoints"],
                                    },
                                },
                            })
                            available_functions["add_waypoint"] = add_waypoint

                        if "loadbalancing" in policy_types:
                            tools.append({
                                "type": "function",
                                "function": {
                                    "name": "add_load_balance",
                                    "description": "Given location l1 and destination prefix p1, load balancing means if l1 wants to reach p1, the traffic path should be load-balanced across a certain number of paths."
                                                   "The function adds a load-balancing requirement to the formal specification.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "source": {
                                                "type": "string",
                                                "description": "The source location of load balancing.",
                                            },
                                            "prefix": {
                                                "type": "string",
                                                "description": "The destination prefix.",
                                            },
                                            "num": {
                                                "type": "integer",
                                                "items": {"type": "integer"},
                                                "description": "The number of paths for load balance.",
                                            },
                                        },
                                        "required": ["source", "prefix", "num"],
                                    },
                                }
                            })
                            available_functions["add_load_balance"] = add_load_balance

                        try:
                            client = OpenAI()
                            response = client.chat.completions.create(
                                model=model_configurations[args.model]['model_name'],
                                messages=messages,
                                tools=tools,
                                tool_choice="auto",
                            )
                            response_message = response.choices[0].message
                            result_row['prompt_tokens'] = response.usage.prompt_tokens
                            result_row['completion_tokens'] = response.usage.completion_tokens

                            completion_cost = get_openai_token_cost_for_model(
                                model_configurations[args.model]['model_name'],
                                result_row['completion_tokens'], is_completion=True
                            )
                            prompt_cost = get_openai_token_cost_for_model(
                                model_configurations[args.model]['model_name'],
                                result_row['prompt_tokens']
                            )
                            result_row['total_cost'] = prompt_cost + completion_cost

                            tool_calls = response_message.tool_calls
                            if tool_calls:
                                for tool_call in tool_calls:
                                    function_name = tool_call.function.name
                                    function_to_call = available_functions[function_name]
                                    function_args = json.loads(tool_call.function.arguments)
                                    function_to_call(
                                        result,
                                        **function_args
                                    )

                            logging.warning("LLM Result: " + str(result))
                        except JSONDecodeError:
                            result_row['format_error'] = str(e)
                            skip_compare = True
                            result = response_message
                        except Exception as e:
                            result_row['model_error'] = str(e)
                            skip_compare = True
                            result = response_message
                    else:
                        combined_system_prompt = [SETUP_PROMPT]
                        if "reachability" in policy_types:
                            combined_system_prompt.append(REACHABILITY_FUNCTION)
                        if "waypoint" in policy_types:
                            combined_system_prompt.append(WAYPOINT_FUNCTION)
                        if "loadbalancing" in policy_types:
                            combined_system_prompt.append(LOADBALANCING_FUNCTION)
                        combined_system_prompt.append(ASK_FOR_RESULT_PROMPT)

                        messages = [
                            ("system", "\n".join(combined_system_prompt)),
                            ("user", "The requirements are as below\n" + ' '.join(human_language))
                        ]

                        try:
                            if model_type == 'openai':
                                with get_openai_callback() as cb:
                                    llm_result = llm_step_1.invoke(messages)
                                    result_row['prompt_tokens'] = cb.prompt_tokens
                                    result_row['completion_tokens'] = cb.completion_tokens
                                    result_row['total_cost'] = cb.total_cost
                            else:
                                llm_result = llm_step_1.invoke(messages)

                            fns = llm_result.content
                            fns = fns.replace("```json\n", "").replace("```", "").replace("plaintext\n", "")
                            parse_functions(result, fns)
                        except Exception as e:
                            result_row['model_error'] = str(e)
                            skip_compare = True
                            result = None

                        logging.warning("LLM Result: " + str(result))

                    logging.warning("==================================================================")

                    if not skip_compare:
                        result_row['time'] = time.time() - start_time

                        new_result = copy.copy(result)
                        if "waypoint" in result:
                            new_result["waypoint"] = {}
                            for k, v in result["waypoint"].items():
                                new_result["waypoint"][k.replace(" ", "")] = v
                        if "loadbalancing" in result:
                            new_result["loadbalancing"] = {}
                            for k, v in result["loadbalancing"].items():
                                new_result["loadbalancing"][k.replace(" ", "")] = v

                        compare_result(expected_spec, new_result, result_row)
                        diff = DeepDiff(expected_spec, new_result, ignore_order=True)
                        result_row['diff'] = str(diff)

                    w.writerow(result_row)
                    f.flush()


if __name__ == "__main__":
    main(parse_args())
