SETUP_PROMPT = """Your task is to find conflicts in network requirements given by the users.
You only reply in JSON, no natural language."""

FUNCTION_PROMPT = """The network requirements are in three categories below:
a.  Reachability
Given location l1 and prefix p1, reachability from l1 to p1 means that l1 can send packets to p1 directly or through other locations. Requirements would focus on allowing or forbidding the reachability from location to prefix.
b.  Waypoint
Given location l1 and destination prefix p1, waypoint means that if l1 wants to reach p1, the traffic path should include a list of locations.
c.  Load Balancing
Given location l1 and destination prefix p1, load balancing means if l1 wants to reach p1, the traffic path should be load-balanced across a certain number of paths.
"""

ASK_FOR_RESULT_PROMPT = """Give the user the result in JSON only, do not add any additional text for explanation. 
The output should be a valid JSON.
If there is no conflict, you reply with a format like: {{"status": "OK", "result": "no conflict"}}.
If there is a conflict, you reply with a format like: {{"status": "ERROR", "reason": "<YOUR_REASON>"}}. <YOUR_REASON> can be in natural language."""
