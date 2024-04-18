topology_test = {
    'h1': {0: 'A'},
    'h2': {0: 'A'},
}

expected_test = {
    'h1': {'h2': ['h1', 'h2']},
    'h2': {'h1': ['h2', 'h1']}
}

feedback_wrong = """
Invalid output format of the function: The output should be Dict[str, Dict[str, List[str]]], a dictionary where:
The keys are strings representing the host names (e.g., "h1", "h2", etc.).
The values are dictionaries that represent the paths from the key host to each of other reachable hosts. These dictionaries have:
Keys as strings representing the other host names.
Values as lists of strings representing the sequence of switches (containing source and destination) to be traversed to get from one host to another host. The path should only consists of hosts and routers, and should exclude LAN identifiers.
For example: "h1": {"h2": ["h1", "s1", "h2"], "h3": ["h1", "s2", "h3"]} means h1 should send packets to h2 along the path made of s1, and send packet to h3 along the path made of s2
"""


# ~function_code~

def test_gpt_output():
    output = compute_routing_paths(topology_test, None)

    assert output == expected_test, \
        (f"{feedback_wrong}. "
         f"I have topology: {topology_test} as input. "
         f"Expected output of the function should be {expected_test}, but I got {output}.")
