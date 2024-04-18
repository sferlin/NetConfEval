topology_test = {
    'h1': {0: 'A', 1: 'D', 2: 'H'},
    'h2': {0: 'C', 1: 'G', 2: 'K'},
    's1': {0: 'A', 1: 'B'},
    's2': {0: 'B', 1: 'C'},
    's3': {0: 'D', 1: '3'},
    's4': {0: 'E', 1: 'F'},
    's5': {0: 'F', 1: 'G'},
    's6': {0: 'H', 1: 'I'},
    's7': {0: 'I', 1: 'J'},
    's8': {0: 'J', 1: 'K'}
}

specification_test = {
    "reachability": {
    },
    "loadbalancing": {
        "(h1, h2)": 2
    }
}

expected_test_1 = {
    'h1': {'h2': [['h1', 's1', 's2', 'h2'], ['h1', 's3', 's4', 's5', 'h2']]},
    'h2': {'h1': [['h2', 's2', 's1', 'h1']]}
}

expected_test_2 = {
    'h1': {'h2': [['h1', 's1', 's2', 'h2'], ['h1', 's6', 's7', 's8', 'h2']]},
    'h2': {'h1': [['h2', 's2', 's1', 'h1']]}
}


feedback_wrong = """If the string "(source host, destination host)" is one of the keys in the loadbalancing map, pick the corresponding integer value number of shortest paths, and put them in the list.
Otherwise pick out only the shortest path satisfying all the requirements."""


# ~function_code~

def test_gpt_output():
    output = compute_routing_paths(topology_test, specification_test)

    assert output == expected_test_1 or output == expected_test_2, \
        (f"{feedback_wrong}. "
         f"I have topology: {topology_test} as input. "
         f"Expected output of the function should be {expected_test_1} or {expected_test_2}, but I got {output}.")