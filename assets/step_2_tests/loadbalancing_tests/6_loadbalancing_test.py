topology_test = {
    'h1': {0: 'A'},
    'h2': {0: 'E', 1: 'F'},
    's1': {0: 'A', 1: 'B', 2: 'C'},
    's2': {0: 'B', 1: 'D'},
    's3': {0: 'C', 1: 'F'},
    's4': {0: 'D', 1: 'E'}
}

specification_test = {
    "reachability": {
    },
    "loadbalancing": {
        "(h1, h2)": 2
    }
}

expected_test = {
    'h1': {'h2': [['h1', 's1', 's3', 'h2'], ['h1', 's1', 's2', 's4', 'h2']]},
    'h2': {'h1': [['h2', 's3', 's1', 'h1']]}
}

feedback_wrong = """If the string "(source host, destination host)" is one of the keys in the loadbalancing map, pick the corresponding integer value number of shortest paths, and put them in the list.
Otherwise pick out only the shortest path satisfying all the requirements."""


# ~function_code~

def test_gpt_output():
    output = compute_routing_paths(topology_test, specification_test)

    assert output == expected_test, \
        (f"{feedback_wrong}. "
         f"I have topology: {topology_test} as input. "
         f"Expected output of the function should be {expected_test}, but I got {output}.")