def compute_routing_paths(topo, requirements=None):
    import itertools
    import networkx as nx

    # Construct a network graph from the topology
    G = nx.Graph()
    for device, connections in topo.items():
        for port, lan in connections.items():
            G.add_edge(device, lan)

    # Identify the unidirectional host pairs (host1, host2) based on the network topology
    hosts = [device for device in topo.keys() if device.startswith('h')]
    host_pairs = list(itertools.permutations(hosts, 2))

    # For each host pair, find all possible paths
    paths = {}
    for host1, host2 in host_pairs:
        all_paths = list(nx.all_simple_paths(G, source=host1, target=host2))
        all_paths.sort(key=len)  # sort paths from shortest to longest

        # For each host pair, pick the shortest path among those satisfying the requirements
        for path in all_paths:
            switches_in_path = [node for node in path if node.startswith('s')]

            # Check reachability requirements
            if not all(host2 in requirements.get('reachability', {}).get(switch, []) for switch in switches_in_path):
                continue

            shortest_path = path
            break
        else:
            shortest_path = []  # no path found

        # Remove LAN identifiers in the path
        shortest_path = [node for node in shortest_path if node.startswith('h') or node.startswith('s')]

        if host1 not in paths:
            paths[host1] = {}
        paths[host1][host2] = shortest_path

    return paths