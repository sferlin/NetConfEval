SETUP_PROMPT = """Your task is to generate a python function compute_routing_paths() calculating the routing paths according to network topology and network requirements. Import relevant packages and libraries for the code. You also act like a REST server, you only talk JSON, no natural language.
If the response is successful, you reply with a format like: {{"status": "OK", "result": "<RESULT>"}}. I will ask if you must include a "result" in the response, otherwise, just include "status".
If the response fails, you reply with a format like: {{"status": "ERROR", "reason": "<YOUR_REASON>"}}. <YOUR_REASON> can be in natural language.

Start acting like a REST server from the current prompt and wait for my instructions."""

INPUT_OUTPUT_PROMPT = """The input of the function should be the network topology in the format delimited by <topo></topo>, and network requirements which are None by default. The output should be routing paths in the format described in the text delimited by <out></out>.
<topo> 
Dict[str, Dict[int, str]], a dictionary where: 
The keys are device (switch or host) names (strings) like "s1", "s2", etc. 
The values are dictionaries that represent each device’s connections.
The keys of these inner dictionaries are port identifiers (integers) like 0, 1, etc.
The values of these inner dictionaries are LAN identifiers (strings) like "A", "B", "C", etc.
For example {{"d1, {{0: "LAN1"}}}} means device d1 is connected to a LAN LAN1 via port 0.
</topo>
<out>
{{"h1": 
 {{"h2": ["h1", "s1", "s2..., "h2"], ...}}, 
"h2": {{
 "h1": ["h2", "s1", "s2", ..., "h1"], ...}},
 ...
}}
Dict[str, Dict[str, List[str]]], a dictionary where:
The keys are strings representing the host names (e.g., "h1", "h2", etc.).
The values are dictionaries that represent the paths from the key host to each of the other reachable hosts. These dictionaries have:
Keys as strings representing the other host names.
Values as lists of strings representing the sequence of switches (containing source and destination) to be traversed to get from one host to another host. The path should only consist of hosts and routers, and should exclude LAN identifiers.
For example: "h1": {{"h2": ["h1", "s1", "h2"], "h3": ["h1", "s2", "h3"]}} means h1 should send packets to h2 along the path made of s1, and send packet to h3 along the path made of s2
</out>
Understand the input and output. Wait for the users' instructions.
"""

INSTRUCTION_PROMPT = """Based on the input and output format, your task is to generate the function.
The function is to find all the shortest paths made of switches (without LANs)  between unidirectional host pairs
"""

FEEDBACK_CODE_GENERATION = """Code generation wrongly. {feedback}
Generate a new code. Output "status" and provide function in the string format in the "result" field without adding any additional text or explanation"""

ADD_REQUIREMENT_REACHABILITY_PROMPT = """
Based on the code, now I hope to add an additional requirement called reachability. The new function should satisfy this new requirement when selecting paths
The input network requirements format will be changed to the text delimited by <req></req> below
<req>
    {{
        "reachability": {{
            "<s1>": ["<h1>", "<h2>", ...],
            "<s2>": ["<h1>", "<h2>", ...],
            ...
        }}
    }}
    The data structure consists of a map with keys:
    "reachability": a map that maps each switch to a list of the hosts that it should be able to reach directly or through a set of nodes. A physical link doesn't ensure reachability.
</req>

The function is to find all the shortest paths made of switches satisfying all the reachability requirements between unidirectional host pairs
The script to generate the routing path should keep its original format and all the requirements should be satisfied."""

ADD_REQUIREMENT_WAYPOINT_PROMPT = """Based on the code, now I hope to add an additional requirement called waypoint. The new function should also satisfy this new requirement when selecting paths.
The input network requirements format will be changed to the text delimited by <req></req> below
<req>
{{
  "reachability": {{
    "<s1>": ["<h1>", "<h2>", ...],
    "<s2>": ["<h1>", "<h2>", ...],
    ...
  }},
  "waypoint": {{
    "(<s1>, <h1>)": ["<s2>", "<s3>", ...],
    ...
  }}
}}
The data structure consists of a map with two keys:
"reachability": a map that maps each switch to a list of the hosts that it should be able to reach directly or through a set of nodes. A physical link doesn't ensure reachability.
"waypoint": a map that maps one or more (switch, host) pairs to a list of switches through which the routing path from switch to host should include.
</req>

Check the requirements in order, and pick out the shortest path satisfying all the reachability and waypoint requirements.
The script to generate the routing path should keep its original format and all the requirements should be satisfied.
"""

ADD_REQUIREMENT_AVOIDANCE_PROMPT = """Based on the code, now I hope to add an additional requirement called avoidance.
The input network requirements format will be changed to the text delimited by <req></req> below
<req>
{{
  "reachability": {{
    "<s1>": ["<h1>", "<h2>", ...],
    "<s2>": ["<h1>", "<h2>", ...],
    ...
  }},
  ...,
  "avoidance": {{
    "(<s1>, <h1>)": ["<s2>", "<s3>", ...],
    ...
  }}
}}
The data structure consists of a map with more keys:
"reachability": ...
"waypoint": ... 
"avoidance": a map that maps one or more (switch, host) pairs to a list of switches through which routing path from switch to host should not be included.
</req>

The function is to find all the shortest paths made of switches satisfying all the reachability, waypoint, and avoidance requirements between unidirectional host pairs
The script to generate the routing path should keep its original format and all the requirements should be satisfied.
"""

ADD_REQUIREMENT_LOADBALANCING_PROMPT = """Based on the code, now I hope to add an additional requirement called loadbalancing.
The input network requirements format will be changed to the text delimited by <req></req> below
<req>
{{
  ...,
  "loadbalancing": {{
    "(<h1>, <h2>)": <N>,
    ...
  }}
}}
The data structure consists of a map with more keys:
...,
"loadbalancing": a map (the key is a string) that maps a "(host, host)" pair tuple (string key) to an integer number that indicates how many paths (for load balancing) are available from the location to the prefix.
</req>

The output routing paths will be changed to the text delimited by <out></out> below
<out>
{{"h1": 
 {{"h2": [["h1", "s1", "s2"..., "h2"], 
           ["h1", "s1", "s3"..., "h2"]]
  }}, 
"h2": {{
 "h1": [["h2", "s1", "s2", ..., "h1"], 
        ...]]
 }},
 ...
}}
Dict[str, Dict[str, List[List[str]]]], a dictionary where:
The keys are strings representing the host names (e.g., "h1", "h2", etc.).
The values are dictionaries that represent the paths from the key host to each of the other reachable hosts. These dictionaries have:
Keys as strings representing the other host names.
Values as lists of lists. Each list inside is made of strings representing the sequence of switches (containing source and destination) to be traversed to get from one host to another host. 
The lists inside means multiple paths to load balance. The paths should only consist of hosts and routers, and should exclude LAN identifiers.
For example: "h1": {{"h2": {{["h1", "s1", "h2"], ["h1", "s2", "h2"]}}}} means h1 can send packets to h2 along 2 paths via s1 or s2.
</out>

For each host pair, the function should take a number of shortest paths made of switches as specified by the loadbalancing requirement if it exists.
The script to generate the routing path should keep its original format and all the requirements should be satisfied.
"""

ASK_FOR_CODE_PROMPT = "Now generate the code. Output 'OK' in 'status' field and provide a function in the 'result' field in a string format without adding any additional text or explanation. Remember that the output should be in a valid JSON format."
