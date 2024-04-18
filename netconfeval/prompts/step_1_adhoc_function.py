SETUP_PROMPT = """You need to translate human-language requirements into function calls.
The available functions are listed below:"""

REACHABILITY_FUNCTION = """ - add_reachability(l1: str, p1: str):
Given location l1 and prefix p1, reachability means that l1 can send packets to p1 directly or through other locations.
The function adds a reachability requirement to the formal specification.
For example: add_reachability("a", "b")"""

WAYPOINT_FUNCTION = """ - add_waypoint(l1: str, p1: str, w: list[str]):
Given location l1 and destination prefix p1, waypoint means that if l1 wants to reach p1, the traffic path should include a list of locations w.
The function adds a waypoint requirement to the formal specification.
For example: add_waypoint("a", "b", ["c"])"""

LOADBALANCING_FUNCTION = """ - add_load_balance(l1: str, p1: str, num: int): 
Given location l1 and destination prefix p1, load balancing means if l1 wants to reach p1, the traffic path should be load-balanced across num number of paths.
The function adds a load-balancing requirement to the formal specification.
For example: add_load_balance("a", "b", 1)"""

ASK_FOR_RESULT_PROMPT = "Please translate all these requirements. Write only the calls to the functions (no specific programming syntax). No additional text."
