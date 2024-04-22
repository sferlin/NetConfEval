import argparse
import io
import ipaddress
import json
import logging
import os
import re
import sys

from Kathara.manager.Kathara import Kathara
from Kathara.model.Lab import Lab
from Kathara.model.Machine import Machine

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.prompts.step_3_low_level import *


def start_container() -> Machine:
    lab = Lab("netconfeval")
    machine = lab.new_machine("frr_test", image="kathara/frr")

    Kathara.get_instance().undeploy_machine(machine)
    Kathara.get_instance().deploy_machine(machine)

    return machine


def stop_container(machine: Machine) -> None:
    Kathara.get_instance().undeploy_machine(machine)


guess_for_placeholders: dict = {
    'riftd': {'system-id': '123', 'level': '2'}
}

lines_to_ignore: dict = {
    'bgpd': ['router-id'],
    'ospfd': ['ospf router-id'],
    'ripd': ['version 1', 'version 2'],
}


def format_config(config: str, daemon: str) -> str:
    split_conf = config.splitlines()
    for i, line in enumerate(split_conf):
        if '[PLACEHOLDER]' in line:
            split_line = line.split("[PLACEHOLDER]")
            before_ph = split_line[0].strip()
            # Do some guessing and pray
            if daemon in guess_for_placeholders:
                found = False
                for token, value in guess_for_placeholders[daemon].items():
                    if token in before_ph:
                        # Found!
                        split_conf[i] = line.replace('[PLACEHOLDER]', value)
                        found = True
                        break

                if not found:
                    logging.warning(f"Replacement for placeholder in line `{line}` not found!")

    if daemon not in lines_to_ignore:
        return "\n".join(split_conf)

    final_conf = []
    for i, line in enumerate(split_conf):
        # Remove some spourious lines (e.g., ospf router-id which is not required, or rip version)
        ignored = False

        for to_ignore in lines_to_ignore[daemon]:
            if to_ignore in line:
                ignored = True
                break

        if not ignored:
            final_conf.append(line)

    return "\n".join(final_conf)


def apply_and_dump(machine: Machine, config_path: str, daemon: str) -> str:
    # Load the configuration in FRR
    for command in [
        f"cp {config_path} /etc/frr/{daemon}.conf",
        "systemctl restart frr"
    ]:
        logging.warning(f"Executing command `{command}` in FRR container")
        exec_output = Kathara.get_instance().exec(machine.name, command, lab=machine.lab)
        try:
            while True:
                next(exec_output)
        except StopIteration:
            pass

    # Dump the normalized configuration
    command = f"vtysh -c 'write terminal no-header'"
    logging.warning(f"Executing command `{command}` in FRR container")
    exec_output = Kathara.get_instance().exec(machine.name, command, lab=machine.lab)
    frr_output = ""
    try:
        while True:
            (stdout, _) = next(exec_output)
            stdout = stdout.decode('utf-8') if stdout else ""

            if stdout:
                frr_output += stdout
    except StopIteration:
        pass

    return frr_output


def apply_rift(config: str) -> (list, list):
    """
    Made-up parser for the FRR RIFT protocol.
    """
    config_data = {
        'level': 0,
        'system_id': None,
        'lie_address': None,
        'interfaces': set(),
        'interfaces_keys': {},
        'prefixes': set(),
        'redistribute': {}
    }
    config_errors = []

    split_config = config.splitlines()
    router_rift_line = set([i for i, x in enumerate(split_config) if 'router rift' in x.strip()])
    if router_rift_line:
        begin = router_rift_line.pop()
    else:
        config_errors.append("`router rift` not found.")
        return [], config_errors

    before_router = split_config[:begin]
    if before_router:
        for cmd_before in before_router:
            if cmd_before.strip():
                config_errors.append(f"Command `{cmd_before.strip()}` not valid.")

    # We ignore whatever is before "router rift"
    split_config = split_config[(begin + 1):]
    for idx, line in enumerate(split_config):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('!') or stripped_line == "exit":
            # We ignore these lines
            continue

        if stripped_line.startswith('level') or stripped_line.startswith('no level'):
            matches = re.search(
                r"^(?P<del>no )?level (?P<lvl>.*)$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `level` command: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            if delete:
                config_data['level'] = 0
            else:
                lvl = matches.group('lvl')
                if lvl.isnumeric():
                    lvl = int(lvl)
                    if lvl < 1 or lvl > 20:
                        config_errors.append(f"`level` must be between 0 and 20, got {lvl}.")
                    else:
                        config_data["level"] = lvl
                else:
                    if lvl == "leaf":
                        config_data["level"] = 1
                    elif lvl == "ew":
                        config_data["level"] = 21
                    elif lvl == "tof":
                        config_data["level"] = 20
                    else:
                        config_errors.append(f"`level` must be leaf|ew|tof, got `{lvl}`.")
        elif stripped_line.startswith('system-id') or stripped_line.startswith('no system-id'):
            matches = re.search(
                r"^(?P<del>no )?system-id (?P<sid>.*)$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `system-id` command: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            if delete:
                config_data["system_id"] = None
            else:
                sid = matches.group('sid')
                if not sid.isnumeric():
                    config_errors.append(f"`system-id` must be numeric, got `{sid}`.")
                    continue

                sid = int(sid)
                if sid < 1 or sid > 4294967295:
                    config_errors.append(f"`system-id` must be between 1 and 4294967295, got {sid}.")
                else:
                    config_data["system_id"] = sid
        elif stripped_line.startswith('lie-address') or stripped_line.startswith('no lie-address'):
            matches = re.search(
                r"^(?P<del>no )?lie-address (?P<lie>.*)$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `lie-address` command: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            lie_address = matches.group('lie')
            try:
                lie_address = ipaddress.ip_address(lie_address)
            except ValueError:
                config_errors.append(f"LIE address `{lie_address}` is not a valid IP address.")
                continue

            if delete:
                if config_data['lie_address']:
                    if config_data["lie_address"] == lie_address:
                        config_data["lie_address"] = None
                    else:
                        config_errors.append(
                            f"Cannot remove LIE address `{lie_address}`, current is `{config_data['lie_address']}`."
                        )
            else:
                config_data["lie_address"] = lie_address
        elif stripped_line.startswith('interface') or stripped_line.startswith('no interface'):
            matches = re.search(
                r"^(?P<del>no )?interface (?P<iface>\w+)( active-key (?P<key>\w+))?$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `interface` command: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            ifname = matches.group('iface')
            key = matches.group('key')
            if delete:
                try:
                    config_data["interfaces"].remove(ifname)
                    del config_data["interfaces_keys"][ifname]
                except KeyError:
                    continue
            else:
                if key:
                    if not key.isnumeric():
                        config_errors.append(f"Interface `active-key` must be numeric, got {key}.")
                        continue
                    else:
                        key = int(key)
                        if key < 0 or key > 255:
                            config_errors.append(f"Interface `active-key` must be between 0 and 255, got {key}.")
                            continue

                if ifname not in config_data['interfaces']:
                    config_data['interfaces'].add(ifname)
                config_data['interfaces_keys'][ifname] = key
        elif stripped_line.startswith('prefix') or stripped_line.startswith('no prefix'):
            matches = re.search(
                r"^(?P<del>no )?prefix (?P<prefix>.*)$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `prefix` command: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            prefix = matches.group('prefix')
            try:
                prefix = ipaddress.ip_network(prefix)
            except ValueError:
                config_errors.append(f"Prefix `{prefix}` is not a valid IP network.")
                continue

            if delete:
                try:
                    config_data["interfaces"].remove(prefix)
                except KeyError:
                    continue
            else:
                config_data['prefixes'].add(prefix)
        elif stripped_line.startswith('redistribute') or stripped_line.startswith('no redistribute'):
            matches = re.search(
                r"^(?P<del>no )?redistribute (?P<red>\w+)( metric (?P<m>\d+))?"
                r"( route-map (?P<rm>\w+)( direction (?P<dir>\w+))?)?$",
                stripped_line
            )

            if not matches:
                config_errors.append(f"Invalid `redistribute command`: `{stripped_line}`.")
                continue

            delete = matches.group('del') is not None
            redistribute = matches.group("red").strip()
            metric = matches.group("m")
            route_map = matches.group("rm")
            direction = matches.group("dir")

            if redistribute not in ['babel', 'bgp', 'connected', 'eigrp', 'isis' 'kernel' 'openfabric', 'ospf', 'sharp',
                                    'static', 'table']:
                config_errors.append(f"Invalid redistribute value: `{redistribute}`.")
                continue

            if metric:
                metric = metric.strip()
                if not metric.isnumeric():
                    config_errors.append(f"Invalid redistribute metric value: `{metric}`.")
                    continue
                else:
                    metric = int(metric)
                    if metric < 0 or metric > 16:
                        config_errors.append(f"Redistribute `metric` must be between 1 and 16, got {metric}.")
                        continue

            if route_map:
                route_map = route_map.strip()
                if direction:
                    direction = direction.strip()
                    if direction not in ['northbound', 'southbound']:
                        config_errors.append(f"Invalid redistribute direction: `{direction}`.")
                        continue

            if delete:
                if redistribute in config_data['redistribute']:
                    del config_data['redistribute'][redistribute]
            else:
                config_data['redistribute'][redistribute] = (metric, route_map, direction)
        else:
            config_errors.append(f"Undefined command `{stripped_line}` at line {idx + 1}.")

    # Reserialize the configuration in a proper format
    serialized_config = ["router rift"]
    if config_data['level'] > 0:
        lvl = config_data['level']
        lvl = "ew" if lvl == 21 else lvl
        serialized_config.append(f"  level {lvl}")
    if config_data['system_id']:
        serialized_config.append(f"  system-id {config_data['system_id']}")
    if config_data['lie_address']:
        serialized_config.append(f"  lie-address {config_data['lie_address']}")
    for iface in sorted(config_data['interfaces'], key=lambda x: re.sub(r'[^0-9]', '', x)):
        iface_string = f"  interface {iface}"
        if config_data['interfaces_keys'][iface]:
            iface_string += f" active-key {config_data['interfaces_keys'][iface]}"
        serialized_config.append(iface_string)
    for prefix in config_data['prefixes']:
        serialized_config.append(f"  prefix {prefix}")
    for red, (m, rm, direction) in config_data['redistribute'].items():
        red_string = f"  redistribute {red}"
        if m:
            red_string += f" metric {m}"
        if rm:
            red_string += f" route-map {rm}"
        if direction:
            red_string += f" direction {direction}"
        serialized_config.append(red_string)

    return serialized_config, config_errors


def run_rift(results: dict) -> dict[str, str]:
    name_to_formatted = {}

    for name, expected in results.items():
        expected_for_rift = format_config(expected, "riftd")
        rift_expected, _ = apply_rift(expected_for_rift)
        name_to_formatted[name] = "\n".join(rift_expected)

    return name_to_formatted


scenario_to_daemon: dict = {
    "ospf_simple": "ospfd",
    "ospf_multiarea": "ospfd",
    "rip": "ripd",
    "rift_dc": run_rift,
    "bgp_simple": "bgpd",
}


def run_results(scenario: str, results: dict, machine: Machine) -> dict[str, str] | None:
    daemon = scenario_to_daemon[scenario]
    if daemon is None:
        logging.info(f"Cannot evaluate scenario {scenario} since it is unsupported.")
        return None

    if not isinstance(daemon, str):
        logging.info(f"Running custom daemon parser for scenario {scenario}...")
        return daemon(results)

    guest_to_host = {}
    for name, expected in results.items():
        expected_for_frr = format_config(expected, daemon)
        guest_to_host[f"/expected/{name}"] = io.StringIO(expected_for_frr)
    guest_to_host["/etc/frr/daemons"] = io.StringIO(f"{daemon}=yes")

    Kathara.get_instance().copy_files(machine, guest_to_host)

    # At this point we can run the check
    name_to_formatted = {}
    for name, result in results.items():
        # Load the expected configuration in FRR
        logging.warning(f"Loading EXPECTED configuration in FRR container")
        name_to_formatted[name] = apply_and_dump(machine, f"/expected/{name}", daemon)

    Kathara.get_instance().exec(machine.name, 'rm -Rf /expected', lab=machine.lab)

    return name_to_formatted


dataset: dict = {
    'ospf_simple': {
        'goal': """
            I want to configure the network using OSPF protocol.

            Routers are all located in area 0.0.0.0.

            Here are how routers speak OSPF:
            - bb0 speaks OSPF on all interfaces falling in 10.0.0.0/16
            - bb1 speaks OSPF on all interfaces falling in 10.0.0.0/16
            - bb2 speaks OSPF on all interfaces falling in 10.0.0.0/16
            - bb3 speaks OSPF on all interfaces falling in 10.0.0.0/16
            - bb4 speaks OSPF on all interfaces falling in 10.0.0.0/16

            I want to set the following costs (if not specified, assume cost=10):
            - bb0[0] cost 21
            - bb0[1] cost 36
            - bb1[1] cost 45
            - bb3[0] cost 7

            Here are the announced networks:
            - bb0 redistributes directly connected
            - bb1 redistributes directly connected
            - bb2 redistributes directly connected
            - bb3 redistributes directly connected
            - bb4 redistributes directly connected""",
        'docs': ['ospf']
    },
    'ospf_multiarea': {
        'goal': """
                I want to configure the network using OSPF.

                Routers are all located in area 0.0.0.0.

                Routers are located in the following areas (if not specified, assume area=0.0.0.0):
                - r1 and r2 are in area 1.1.1.1 (stub)
                - r3 is in area 2.2.2.2 (stub)
                - r4, r5, and r6 are in area 3.3.3.3 (stub)

                Here are how routers speak OSPF:
                - bb0 speaks OSPF on all interfaces falling in 10.0.0.0/16
                - bb1 speaks OSPF as follows:
                    - area 0.0.0.0 on all interfaces falling in 10.0.0.0/16
                    - area 1.1.1.1 on all interfaces falling in 100.0.0.0/30
                    - area 2.2.2.2 on all interfaces falling in 110.0.0.0/30
                - bb2 speaks OSPF as follows:
                    - area 0.0.0.0 on all interfaces falling in 10.0.0.0/16
                    - area 2.2.2.2 on all interfaces falling in 120.0.0.0/30
                    - area 3.3.3.3 on all interfaces falling in 130.0.0.0/30
                - bb3 speaks OSPF on all interfaces falling in 10.0.0.0/16
                - bb4 speaks OSPF on all interfaces falling in 10.0.0.0/16
                - r1 speaks OSPF on all interfaces falling in 100.0.0.0/30 and 200.0.0.0/16
                - r2 speaks OSPF on all interfaces falling in 200.0.0.0/16
                - r3 speaks OSPF on all interfaces falling in 210.0.0.0/16, 110.0.0.0/30, and 120.0.0.0/30
                - r4 speaks OSPF on all interfaces falling in 220.0.0.0/16
                - r5 speaks OSPF on all interfaces falling in 220.0.0.0/16 and 130.0.0.0/30
                - r6 speaks OSPF on all interfaces falling in 220.0.0.0/16

                I want to set the following costs (if not specified, assume cost=10):
                - bb1[0] cost 90
                - bb2[0] cost 100

                Here are the announced networks:
                - bb0 redistributes directly connected
                - bb1 redistributes directly connected
                - bb2 redistributes directly connected
                - bb3 redistributes directly connected
                - bb4 redistributes directly connected
                - r1 redistributes directly connected
                - r2 redistributes directly connected
                - r3 redistributes directly connected
                - r4 redistributes directly connected
                - r5 redistributes directly connected
                - r6 redistributes directly connected""",
        'docs': ['ospf']
    },
    'rip': {
        'goal': """
                I want to configure the network using RIP.

                All routers speak RIP with routers falling in 100.1.0.0/24.

                All routers redistribute directly connected.""",
        'docs': ['rip']
    },
    'bgp_simple': {
        'goal': """
                I want to configure the network using BGP.

                Routers are located in the following ASes:
                - router1 is in AS 1
                - router2 is in AS 2

                Here are how routers speak BGP:
                - router1 speaks BGP with router2 on interface 0 using the following IPv4 address: 193.10.11.2
                - router2 speaks BGP with router1 on interface 0 using the following IPv4 address: 193.10.11.1

                Here are the announced networks:
                - router1 announces the IPv4 prefix 195.11.14.0/24 to router2
                - router2 announces the IPv4 prefix 200.1.1.0/24 to router1""",
        'docs': ['bgp']
    },
    'rift_dc': {
        'goal': """
                    I want to configure the network using RIFT.

                    All nodes starting with "leaf_" are Leaf nodes, and should be assigned at level 1.
                    All nodes starting with "spine_" are Spine nodes, and level should not be assigned.
                    All nodes starting with "tof_" are Top-of-Fabric nodes, and they should be assigned to the dedicated level.

                    Here are how routers speak RIFT:
                    - All "leaf_" nodes speak RIFT on the first three interfaces.
                    - All "spine_" nodes speak RIFT on all interfaces.
                    - All "tof_" nodes speak RIFT on all interfaces.

                    I want to set the following properties:
                    - spine_1_1_2 should have system ID=874638
                    - tof_1_2_1 should speak RIFT on LIE address 224.0.8.24

                    Here are the announced networks:
                    - leaf_1_0_1 announces the IPv4 prefix 200.0.1.0/24
                    - leaf_1_0_2 announces the IPv4 prefix 200.0.2.0/24
                    - leaf_2_0_1 announces the IPv4 prefix 200.0.3.0/24
                    - leaf_2_0_1 announces the IPv6 prefix fafb::/64
                    - leaf_2_0_2 announces the IPv4 prefix 200.0.4.0/24""",
        'docs': ['rift']
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "datasets")
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    os.makedirs(args.results_path, exist_ok=True)

    assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_3_low_level'))

    # Start the Kathara machine with FRRouting
    frr_device = start_container()

    with open(os.path.join(args.results_path, "step_3_low_level.jsonl"), "w") as f:
        for name, data in dataset.items():
            logging.info(f"Generating data for `{name}`...")

            lab_path = os.path.join(assets_path, name)
            with open(os.path.join(lab_path, 'lab.conf')) as lab_file:
                topology = "".join(lab_file.readlines())

            combined_human_prompt = f"{NETWORK_DESCRIPTION}\n"
            combined_human_prompt += NETWORK_DESCRIPTION_USER.format(topology=topology) + "\n"
            combined_human_prompt += OUTPUT_FORMAT

            dev2configs = {}
            conf_path = os.path.join(lab_path, "configs")
            for dev_conf in filter(lambda x: not x.startswith('.'), os.listdir(conf_path)):
                dev_name, _ = os.path.splitext(dev_conf)
                with open(os.path.join(conf_path, dev_conf)) as dev_file:
                    expected = "".join(dev_file.readlines())

                clean_expected = expected.replace('!', '')
                clean_expected = [re.sub(r"^ +", "", x) for x in clean_expected.split('\n')]
                clean_expected = [x for x in clean_expected if x]

                dev2configs[dev_name] = "\n".join(clean_expected)

            formatted_expected = run_results(name, dev2configs, frr_device)

            result_row = {
                'scenario_name': name,
                'input': combined_human_prompt,
                'result': json.dumps(formatted_expected),
            }

            f.write(json.dumps(result_row) + "\n")
            f.flush()

    stop_container(frr_device)


if __name__ == "__main__":
    main(parse_args())
