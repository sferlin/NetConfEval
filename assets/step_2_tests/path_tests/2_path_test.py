topology_test = {
    'h1': {0: 'A'},
    'h2': {0: 'E', 1: 'F'},
    's1': {0: 'A', 1: 'B', 2: 'C'},
    's2': {0: 'B', 1: 'D'},
    's3': {0: 'C', 1: 'F'},
    's4': {0: 'D', 1: 'E'}
}

expected_test = {
    'h1': {'h2': ['h1', 's1', 's3', 'h2']},
    'h2': {'h1': ['h2', 's3', 's1', 'h1']}
}

feedback_wrong = """From the shortest path to the longest path, check the requirements in order. The path you selected should be the shortest among the satisfying ones"""


# ~function_code~

def test_gpt_output():
    output = compute_routing_paths(topology_test, None)

    assert output == expected_test, \
        (f"{feedback_wrong}. "
         f"I have topology: {topology_test} as input. "
         f"Expected output of the function should be {expected_test}, but I got {output}.")
