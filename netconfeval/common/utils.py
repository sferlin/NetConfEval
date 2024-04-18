import copy
import csv
import logging
import math
import random
from itertools import islice
from typing import Generator, Sized

from sortedcontainers import SortedSet


def load_csv(csv_file: str, policy_types: SortedSet[str]) -> list:
    filtered_data = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key_type = row["type"].replace("PolicyType.", "").replace("Simple", "").lower()
            if key_type in policy_types:
                if key_type == "loadbalancing":
                    filtered_row = {
                        "type": key_type,
                        "source": row['Sources'],
                        "subnet": row['subnet'],
                        "specifics": int(row['specifics'])
                    }
                    filtered_data.append(filtered_row)
                elif row['Status'] == 'PolicyStatus.HOLDS' and row['Sources'] != row['specifics']:
                    filtered_row = {
                        "type": key_type,
                        "source": row['Sources'],
                        "subnet": row['subnet'],
                        "specifics": row['specifics']
                    }
                    filtered_data.append(filtered_row)

    return filtered_data


def pick_sample(n: int, requirements: list, it: int, policy_types: SortedSet[str]) -> list:
    random.seed(5000 + it)
    shuffled_list = sorted(requirements, key=lambda k: random.random())
    if len(policy_types) == 1 and "reachability" in policy_types:
        return [x for x in shuffled_list if x["type"] == "reachability"][:n]

    n_per_requirement = math.ceil(n / len(policy_types))
    waypoint_samples = [x for x in shuffled_list if x["type"] == "waypoint"][:n_per_requirement]

    final_list = []

    for sample in waypoint_samples:
        reachability_samples = [
            x for x in requirements if x["type"] == "reachability" and x["source"] == sample["source"]
                                       and x["subnet"] == sample["subnet"]
        ]
        filtered_sample = None if not reachability_samples else reachability_samples.pop()
        final_list.append(filtered_sample)

        final_list.append(sample)

        if "loadbalancing" in policy_types:
            loadbalancing_samples = [
                x for x in requirements if x["type"] == "loadbalancing" and x["source"] == sample["source"]
            ]
            filtered_sample = None if not loadbalancing_samples else loadbalancing_samples.pop()
            filtered_sample["subnet"] = sample["subnet"]
            final_list.append(filtered_sample)

    return final_list[:n]


def list_chunks(iterable: any, size: int) -> Generator:
    it = iter(iterable)
    item = list(islice(it, size))
    while item:
        yield item
        item = list(islice(it, size))


def chunk_list(iterable: Sized, size: int) -> list | Generator:
    return [iterable] if len(iterable) < size else list_chunks(iterable, size)


def transform_sample_to_expected(data: list[dict]) -> dict:
    transformed_data = {
        "reachability": {},
        "waypoint": {},
        "loadbalancing": {}
    }

    for item in data:
        if item["type"] == 'reachability':
            if item["source"] not in transformed_data["reachability"]:
                transformed_data["reachability"][item["source"]] = []
            transformed_data["reachability"][item["source"]].append(item["subnet"])
        elif item["type"] == 'waypoint':
            key = f'({item["source"]},{item["subnet"]})'
            if key not in transformed_data["waypoint"]:
                transformed_data["waypoint"][key] = []
            transformed_data["waypoint"][key].append(item["specifics"])
        elif item["type"] == 'loadbalancing':
            key = f'({item["source"]},{item["subnet"]})'
            transformed_data["loadbalancing"][key] = int(item["specifics"])

    return transformed_data


reachability_statements = [
    "{source} can reach {subnet}.",
    "The subnet {subnet} is reachable from {source}.",
    "Connectivity from {source} to {subnet} is established.",
    "Traffic originating from {source} can reach the subnet {subnet}.",
    "{subnet} is accessible from {source}."
]

waypoint_statements = [
    "Traffic from {source} to {subnet} passes through {specifics}.",
    "Routing traffic between {source} and {subnet} goes via {specifics}.",
    "The route from {source} to {subnet} includes {specifics}.",
    "To reach {subnet} from {source}, traffic is directed through {specifics}.",
    "The path between {source} and {subnet} involves {specifics}."
]

loadbalancing_statements = [
    "To reach {subnet} from {source}, traffic is balanced on {specifics} paths.",
    "Traffic from {source} to {subnet} is evenly distributed across {specifics} paths.",
    "To reach {subnet} from {source}, traffic should be split across {specifics} paths.",
    "The number of paths between {source} and {subnet} should be {specifics}.",
    "Traffic from {source} to {subnet} should be load-balanced across {specifics} paths."
]

no_reachability_statements = [
    "{source} cannot reach {subnet}.",
]

no_waypoint_statements = [
    "Traffic from {source} to {subnet} should not passes through {specifics}.",
]


def convert_to_human_language(data: list[dict]) -> list[str]:
    language_statements = []

    for item in data:
        if item["type"] == 'reachability':
            statement = random.choice(reachability_statements)
        elif item["type"] == 'waypoint':
            statement = random.choice(waypoint_statements)
        elif item["type"] == 'loadbalancing':
            statement = random.choice(loadbalancing_statements)
        elif item["type"] == 'no_reachability':
            statement = random.choice(no_reachability_statements)
        elif item["type"] == 'no_waypoint':
            statement = random.choice(no_waypoint_statements)
        else:
            return []

        statement = statement.replace("{source}", item["source"])
        statement = statement.replace("{subnet}", item["subnet"])
        statement = statement.replace("{specifics}", str(item["specifics"]))

        language_statements.append(statement)

    return language_statements


def insert_conflict(data: list[dict], start: int | None = None, end: int | None = None) -> None:
    if start is not None and end is None:
        raise Exception("You need to pass `end` if you pass `start`.")
    if end is not None and start is None:
        raise Exception("You need to pass `start` if you pass `end`.")

    index = random.randint(0, len(data) - 1) if end is None else end
    if start is not None:
        conflict_statement_1 = data[start]
    else:
        conflict_statement_1 = data[index]

    conflict_statement_2 = copy.copy(conflict_statement_1)
    if conflict_statement_1["type"] == 'reachability':
        conflict_statement_2["type"] = "no_reachability"
    elif conflict_statement_1["type"] == 'waypoint':
        conflict_statement_2["type"] = "no_waypoint"
    elif conflict_statement_1["type"] == 'loadbalancing':
        conflict_statement_2["specifics"] += 1

    data.insert(index, conflict_statement_2)


def check_subnet_exists_in_reachability(router: str, subnet: str, result: dict) -> bool:
    if "reachability" not in result:
        return False

    result = result["reachability"]
    if router in result.keys():
        if subnet in result[router]:
            return True

    return False


def check_value_exists_in_waypoint(waypoint: str, switches: list, result: dict) -> bool:
    if "waypoint" not in result:
        return False

    waypoints = result["waypoint"]
    if waypoint in waypoints and (switch in waypoints[waypoint] for switch in switches):
        return True

    return False


def check_value_exists_in_loadbalacing(loadbalancing: str, num: int, result: dict) -> bool:
    if "loadbalancing" not in result:
        return False

    loadbalancings = result["loadbalancing"]
    if loadbalancing in loadbalancings and int(loadbalancings[loadbalancing]) == num:
        return True

    return False


def compare_result(expected: dict, result: dict, result_row: dict) -> None:
    count_expected = 0
    count_res = 0
    count_fail = 0
    count_wrong = 0
    if "reachability" in expected:
        for router, subnets in expected["reachability"].items():
            for subnet in subnets:
                count_expected += 1
                if not check_subnet_exists_in_reachability(router, subnet, result):
                    count_fail += 1
                    logging.warning(f"Fail to translate expected reachability from `{router}` to `{subnet}`.")

    if "waypoint" in expected:
        for waypoint, switches in expected["waypoint"].items():
            count_expected += 1
            if not check_value_exists_in_waypoint(waypoint, switches, result):
                count_fail += 1
                logging.warning(f"Fail to translate expected waypoint `{waypoint}`.")

    if "loadbalancing" in expected:
        for loadbalancing, num in expected["loadbalancing"].items():
            count_expected += 1
            if not check_value_exists_in_loadbalacing(loadbalancing, num, result):
                count_fail += 1
                logging.warning(f"Fail to translate expected load balancing `{loadbalancing}`.")

    if "reachability" in result:
        for router, subnets in result["reachability"].items():
            for subnet in subnets:
                count_res += 1
                if not check_subnet_exists_in_reachability(router, subnet, expected):
                    if "waypoint" not in expected or f"({router},{subnet})" not in expected["waypoint"]:
                        count_wrong += 1
                        logging.warning(f"Translate wrongly reachability from `{router}` to `{subnet}`.")

    if "waypoint" in result:
        for waypoint, switches in result["waypoint"].items():
            count_res += 1
            if not check_value_exists_in_waypoint(waypoint, switches, expected):
                count_wrong += 1
                logging.warning(f"Translate wrongly waypoint `{waypoint}`.")

    if "loadbalancing" in result:
        for loadbalancing, num in result["loadbalancing"].items():
            count_res += 1
            if not check_value_exists_in_loadbalacing(loadbalancing, num, expected):
                count_wrong += 1
                logging.warning(f"Translate wrongly load balancing `{loadbalancing}`.")

    result_row['total'] = count_res
    result_row['fail'] = count_fail
    result_row['wrong'] = count_wrong
    result_row['success'] = count_expected - count_fail
    result_row['accuracy'] = result_row['success'] / count_expected
