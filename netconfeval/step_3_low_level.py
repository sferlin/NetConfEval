import argparse
import csv
import difflib
import io
import ipaddress
import json
import logging
import os
import re
import sys
import time
from string import whitespace

from Kathara.manager.Kathara import Kathara
from Kathara.model.Lab import Lab
from Kathara.model.Machine import Machine
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.common.model_configs import model_configurations, get_model_instance
from netconfeval.foundation.step.chain_step import ChainStep
from netconfeval.prompts.step_3_low_level import *


def text_from_pdf(pdf_path: str) -> str:
    loader = PDFMinerLoader(pdf_path)
    pdf_data = loader.load()
    text = pdf_data[0].page_content
    text = text.replace('\xa0', ' ').replace('{', '{{').replace('}', '}}')

    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)
    parser.add_argument('--n_runs', type=int, required=False, default=5)
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "results_low_level")
    )
    parser.add_argument('--mode', type=str, choices=['none', 'full', 'idx', 'rag'])
    parser.add_argument('--rag_chunk_size', type=int, required=False, default=9000)

    return parser.parse_args()


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


ignored_msgs: list = [
    'sendmsg_nexthop',  # Ignores the fact that nexthops are not reachable
]


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


def run_rift(results: dict):
    for name, result in results.items():
        expected_for_rift = format_config(result["expected"], "riftd")
        rift_expected, expected_errors = apply_rift(expected_for_rift)
        generated_for_rift = format_config(result["generated"], "riftd")
        logging.warning(f"{name} Formatted Generated:\n" + generated_for_rift)
        rift_generated, generated_errors = apply_rift(generated_for_rift)

        s = difflib.SequenceMatcher(
            lambda x: x in whitespace or x.strip() == '!',
            rift_expected,
            rift_generated,
            autojunk=False
        )
        result["diff_similarity_daemon"] = s.ratio()
        logging.warning(f"RIFT Diff Similarity for {name}: " + str(result["diff_similarity_daemon"]))

        d = difflib.Differ()
        logging.warning(f"DIFFS for {name}:\n" + "\n".join(list(d.compare(rift_expected, rift_generated))))

        if generated_errors:
            logging.info("There are some errors in the configuration!")
            result["daemon_errors"] = generated_errors
            result["n_daemon_errors"] = len(generated_errors)
            logging.warning(str(result["n_daemon_errors"]) + " RIFT Errors found:\n" + str(generated_errors))
        else:
            result["daemon_errors"] = []
            result["n_daemon_errors"] = 0


scenario_to_daemon: dict = {
    "ospf_simple": "ospfd",
    "ospf_multiarea": "ospfd",
    "rip": "ripd",
    "rift_dc": run_rift,
    "bgp_simple": "bgpd",
}


def run_results(scenario: str, results: dict, machine: Machine) -> None:
    daemon = scenario_to_daemon[scenario]
    if daemon is None:
        logging.info(f"Cannot evaluate scenario {scenario} since it is unsupported.")
        return

    if not isinstance(daemon, str):
        logging.info(f"Running custom daemon parser for scenario {scenario}...")
        daemon(results)
        return

    guest_to_host = {}
    for name, result in results.items():
        expected_for_frr = format_config(result["expected"], daemon)
        guest_to_host[f"/expected/{name}"] = io.StringIO(expected_for_frr)
        generated_for_frr = format_config(result["generated"], daemon)
        logging.warning(f"{name} Formatted Generated:\n" + generated_for_frr)
        guest_to_host[f"/gpt/{name}"] = io.StringIO(generated_for_frr)
    guest_to_host["/etc/frr/daemons"] = io.StringIO(f"{daemon}=yes")

    Kathara.get_instance().copy_files(machine, guest_to_host)

    # At this point we can run the check
    for name, result in results.items():
        # Load the expected configuration in FRR
        logging.warning(f"Loading EXPECTED configuration in FRR container")
        frr_expected = apply_and_dump(machine, f"/expected/{name}", daemon)

        # Load the generated configuration in FRR
        logging.warning(f"Loading GENERATED configuration in FRR container")
        frr_generated = apply_and_dump(machine, f"/gpt/{name}", daemon)

        split_frr_expected = frr_expected.splitlines()
        split_frr_generated = frr_generated.splitlines()
        s = difflib.SequenceMatcher(
            lambda x: x in whitespace or x.strip() == '!',
            split_frr_expected,
            split_frr_generated,
            autojunk=False
        )
        result["diff_similarity_daemon"] = s.ratio()
        logging.warning(f"FRR Diff Similarity for {name}: " + str(result["diff_similarity_daemon"]))

        d = difflib.Differ()
        logging.warning(f"DIFFS for {name}:\n" + "\n".join(list(d.compare(split_frr_expected, split_frr_generated))))

        command = f"/usr/lib/frr/{daemon} -f /gpt/{name} -C"
        exec_output = Kathara.get_instance().exec(machine.name, command, lab=machine.lab)
        frr_stderr = ""
        try:
            while True:
                (_, stderr) = next(exec_output)
                stderr = stderr.decode('utf-8') if stderr else ""

                if stderr:
                    frr_stderr += stderr
        except StopIteration:
            pass

        stderr_split = []
        for line in frr_stderr.split("\n"):
            if not line.startswith("%") and line:
                ignored = False
                for err in ignored_msgs:
                    if err in line:
                        ignored = True
                        break
                if not ignored:
                    stderr_split.append(line)

        if stderr_split:
            logging.info("There are some errors in the configuration!")
            result["daemon_errors"] = stderr_split
            result["n_daemon_errors"] = len(stderr_split)
            logging.warning(str(result["n_daemon_errors"]) + " FRR Errors found:\n" + str(stderr_split))
        else:
            result["daemon_errors"] = []
            result["n_daemon_errors"] = 0

    Kathara.get_instance().exec(machine.name, 'rm -Rf /gpt', lab=machine.lab)
    Kathara.get_instance().exec(machine.name, 'rm -Rf /expected', lab=machine.lab)


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


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    os.makedirs(args.results_path, exist_ok=True)

    rag_lbl = f'_{args.rag_chunk_size}'
    results_time = time.strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(
        os.path.abspath(
            os.path.join(args.results_path, f"log-{args.model}-{args.mode}{rag_lbl}-{results_time}.log")
        )
    )
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.WARNING)
    logging.root.addHandler(file_handler)

    assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_3_low_level'))
    index_text = None
    db = {}
    if args.mode == 'idx':
        index_text = text_from_pdf(os.path.join(assets_path, "index-patched.pdf"))
    elif args.mode == "rag":
        doc_names = ["3.3.pdf", "3.11.pdf", "3.17.pdf", "3.26.pdf"]
        documents = []
        for doc_name in doc_names:
            docs_path = os.path.join(assets_path, doc_name)

            loader = PDFMinerLoader(docs_path)
            pdf_data = loader.load()
            documents.extend(pdf_data)

        logging.info(f"Creating RAG DB with chunk size={args.rag_chunk_size}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.rag_chunk_size, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        db = Chroma.from_documents(chunks, OpenAIEmbeddings())

    llm = get_model_instance(args.model)

    # Start the Kathara machine with FRRouting
    frr_device = start_container()

    w = None

    filename = f"result-{args.model}-{args.mode}{rag_lbl}-{results_time}.csv"
    with (open(os.path.join(args.results_path, filename), 'w') as f):
        for it in range(0, args.n_runs):
            logging.info(f"Performing iteration n. {it + 1}...")

            for name, data in dataset.items():
                logging.info(f"Performing experiment `{name}` (iteration n. {it + 1})...")

                additional_rows = {}
                if args.mode == 'rag':
                    additional_rows = {'chunk_size': args.rag_chunk_size}

                result_row = {
                    'scenario_name': name,
                    'iteration': it,
                    'mode': args.mode,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_cost': 0,
                    'result': None,
                    'format_error': None,
                    'model_error': None,
                    'time': 0,
                    **additional_rows
                }

                if w is None:
                    w = csv.DictWriter(f, result_row.keys())
                    w.writeheader()

                lab_path = os.path.join(assets_path, name)
                with open(os.path.join(lab_path, 'lab.conf')) as lab_file:
                    topology = "".join(lab_file.readlines())

                logging.warning(f"==== RUN #{it + 1} (SCENARIO {name}) ====")

                messages = [
                    ("system", SETUP_PROMPT),
                    ("system", NETWORK_DESCRIPTION),
                    ("human", NETWORK_DESCRIPTION_USER),
                    ("human", "{goal}")
                ]

                goal_text = data['goal']

                start_time = time.time()
                if args.mode == 'idx':
                    messages.append(("system", DOCS_INDEX))

                    prompt_template = ChatPromptTemplate.from_messages(messages)
                    llm_chain = LLMChain(
                        llm=llm,
                        prompt=prompt_template,
                        verbose=True
                    )
                    chain_step = ChainStep(
                        llm_chain=llm_chain,
                        input_formatter=lambda x: x,
                        output_formatter=lambda x: x
                    )

                    try:
                        with get_openai_callback() as cb:
                            _, sec_num = chain_step.process(
                                {"topology": topology, "index": index_text, "goal": goal_text}
                            )
                            result_row['prompt_tokens'] = cb.prompt_tokens
                            result_row['completion_tokens'] = cb.completion_tokens
                            result_row['total_cost'] = cb.total_cost
                    except Exception as e:
                        logging.error(str(e))
                        result_row['model_error'] = str(e)
                        sec_num = "0"

                    logging.warning("LLM Result: " + sec_num)
                    if " " in sec_num:
                        # Sometimes the response is "3.11 OSPF"
                        sec_num_parts = sec_num.split(" ")
                        sec_num_parts = [x for x in sec_num_parts if re.match(r'\d+\.\d+', x)]
                        if sec_num_parts:
                            sec_num = sec_num_parts.pop()

                    doc_path = os.path.join(assets_path, f"{sec_num}.pdf")
                    if not os.path.exists(doc_path):
                        result_row['format_error'] = f"Section `{sec_num}` not correct for {name}."

                        logging.error(result_row['format_error'])

                        w.writerow(result_row)
                        f.flush()

                        continue

                llm_call_args = {}
                if args.mode == "idx":
                    docs_text = text_from_pdf(doc_path)

                    llm_call_args = {"topology": topology, "docs": docs_text, "goal": goal_text}
                    messages = [
                        ("system", SETUP_PROMPT),
                        ("system", NETWORK_DESCRIPTION),
                        ("human", NETWORK_DESCRIPTION_USER),
                        ("human", "{goal}"),
                        ("system", DOCS_STR),
                        ("system", OUTPUT_FORMAT),
                        ("system", ASK_FOR_OUTPUT),
                    ]
                elif args.mode == "none":
                    messages.extend([
                        ("system", OUTPUT_FORMAT),
                        ("system", ASK_FOR_OUTPUT),
                    ])

                    llm_call_args = {"topology": topology, "goal": goal_text}
                elif args.mode == "full":
                    messages.extend([
                        ("system", DOCS_STR),
                        ("system", OUTPUT_FORMAT),
                        ("system", ASK_FOR_OUTPUT),
                    ])

                    docs_text = text_from_pdf(os.path.join(assets_path, "full-docs-shrink.pdf"))

                    llm_call_args = {"topology": topology, "docs": docs_text, "goal": goal_text}
                elif args.mode == "rag":
                    relevant_docs_and_score = db.max_marginal_relevance_search(goal_text, k=8)

                    logging.warning(f"========= RAG Relevant chunks ({args.rag_chunk_size}=========\n" +
                                    "\n\n".join(["!!! CHUNK " + str(i) + " !!!\n" + x.page_content
                                                 for i, x in enumerate(relevant_docs_and_score)]) +
                                    f"\n===================================\n"
                                    )

                    relevant_docs_str = "\n".join([d.page_content for d in relevant_docs_and_score])

                    messages.extend([
                        ("system", DOCS_STR),
                        ("system", OUTPUT_FORMAT),
                        ("system", ASK_FOR_OUTPUT),
                    ])
                    llm_call_args = {"topology": topology, "docs": relevant_docs_str, "goal": goal_text}

                prompt_template = ChatPromptTemplate.from_messages(messages)
                llm_chain = LLMChain(
                    llm=llm,
                    prompt=prompt_template,
                    verbose=True
                )
                chain_step = ChainStep(
                    llm_chain=llm_chain,
                    input_formatter=lambda x: x,
                    output_formatter=lambda x: x
                )
                try:
                    with get_openai_callback() as cb:
                        status, output = chain_step.process(
                            llm_call_args
                        )
                        result_row['prompt_tokens'] += cb.prompt_tokens
                        result_row['completion_tokens'] += cb.completion_tokens
                        result_row['total_cost'] += cb.total_cost
                except Exception as e:
                    logging.error(str(e))
                    result_row['model_error'] = str(e)
                    output = {}
                result_row['time'] = time.time() - start_time

                logging.warning("LLM Result: " + str(output))

                dev2configs = {}
                conf_path = os.path.join(lab_path, "configs")
                for dev_conf in filter(lambda x: not x.startswith('.'), os.listdir(conf_path)):
                    dev_name, _ = os.path.splitext(dev_conf)
                    with open(os.path.join(conf_path, dev_conf)) as dev_file:
                        expected = "".join(dev_file.readlines())

                    generated = output[dev_name] if dev_name in output else ""
                    generated = generated if type(generated) is str else "\n".join(generated) \
                        if generated is not None else ""

                    clean_expected = expected.replace('!', '')
                    clean_expected = [re.sub(r"^ +", "", x) for x in clean_expected.split('\n')]
                    clean_expected = [x for x in clean_expected if x]
                    clean_generated = generated.replace('!', '')
                    clean_generated = [re.sub(r"^ +", "", x) for x in clean_generated.split('\n')]
                    clean_generated = [x for x in clean_generated if x]

                    s = difflib.SequenceMatcher(lambda x: "[PLACEHOLDER]" in x, clean_generated, clean_expected)
                    diff_similarity = s.ratio()

                    dev2configs[dev_name] = {
                        'expected': expected,
                        'generated': generated,
                        'diff_similarity': diff_similarity,
                    }

                run_results(name, dev2configs, frr_device)

                result_row['result'] = json.dumps(dev2configs)

                logging.warning("==================================================================")

                w.writerow(result_row)
                f.flush()

    stop_container(frr_device)


if __name__ == "__main__":
    main(parse_args())
