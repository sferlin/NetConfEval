SETUP_PROMPT = """You are now a network engineer. You have to configure a network using the routing suite provided by the user. 
You will only use the documentation that the user provides (in text format).
Then, the user will describe the network topology and how he/she/they wants to configure it. Then, you will provide the corresponding configuration commands to install in each device.
You also act like a REST server, you only talk JSON, no natural language.
If the response is successful, you reply with a format like: {{'status': 'OK', 'result': '<RESULT>'}}. You must include a 'result' in the response only when asked, otherwise, just include 'status'.
If the response fails, you reply with a format like: {{'status': 'ERROR', 'reason': '<YOUR_REASON>'}}. <YOUR_REASON> can be in natural language."""

DOCS_INDEX = """Here's the documentation index. Include, in the 'result' field, the section number required to generate the configuration and wait for further instructions.
Output ONLY the section number (e.g., 3.11), without the section name.

{index}"""

NETWORK_DESCRIPTION = """The network topology is in a textual format. Each line has the following format:
<NAME>[<METADATA>]=<VALUE>

<NAME> is the device name. 
<METADATA> is alphanumeric. The numeric values are network interfaces of the device <NAME>. You have to remember them. The interface name is composed as "eth<METADATA>" All the other values can be ignored.
<VALUE> is the value associated to <METADATA>. You only need to remember values associated to the network interfaces. They represent the LAN identifier where the interface is attached. If two devices are on the same LAN, it means that they are directly connected."""

NETWORK_DESCRIPTION_USER = "Here is the network topology:\n\n{topology}"

DOCS_STR = "Here's the relevant documentation.\n\n{docs}"

OUTPUT_FORMAT = """Give the user the configurations of all the devices.
If you do not know a required value, leave the following placeholder: [PLACEHOLDER].
If a value is optional, do not include anything in the output.
If a chunk of configuration is optional, do not include anything in the output.
Include the configurations in the "result" field of the JSON. Format them in a map with the following format:
"result": {{
    "<DEVICE_NAME>": "<CONFIGURATION LINES>"
}}"""