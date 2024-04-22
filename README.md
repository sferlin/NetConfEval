# NetConfEval: Can LLMs Facilitate Network Configuration?

## What is it?

We explore opportunities to utilize Large Language Models (LLMs) to make network configuration human-friendly, simplifying the configuration of network devices & development of routing algorithms and minimizing errors. We design a set of benchmarks (NetConfEval) to examine the effectiveness of different models in facilitating and automating network configuration described in our paper "[NetConfEval: Can LLMs Facilitate Network Configuration?](https://doi.org/10.1145/3656296)".


## Installation
Make sure to use Python 3.10 or later. Install this repository and install all the packages.
``` bash
git clone git@github.com:NetConfEval/NetConfEval.git
virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
```

To run experiments with OpenAI models, export your [OPENAI_API_KEY](https://platform.openai.com/api-keys) in the environment:
```bash
export OPENAI_API_KEY="YOUR OPENAI KEY"
```

To run experiments with Huggingface models, install the additional packages and login with your Huggingface account with your [Access Token](https://huggingface.co/settings/tokens).
```bash
pip install -r requirements-hf.txt
huggingface-cli login
```

## From High-Level Requirements to a Formal Specification
This test evaluates LLMs' ability to translate network operators' requirements into a formal specification. For instance, the input information can be converted into a simple data structure to specify the reachability, waypoints, and load-balancing policies in a network.

Here is an example of the experiment. We use `gpt-4-1106` to translate multiple requirements into a formal specification made of the three policies, with a batch size of 3:
```bash
python3 step_1_formal_spec_translation.py --n_run 1 --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 3
```

The experiment results will be stored in the directory named `results_spec_translation` by default.

## From High-Level Requirements to Functions/API Calls
This test evaluates the ability of LLMs' to translate natural language requirements into corresponding function/API calls, which is a common task in network configuration since many networks employ SDN, where a software controller can manage the underlying network via direct API calls.

To translate a few requirements into multiple function calls (```add_reachability(), add_waypoint(), add_load_balance()```) in parallel, run:

```bash
python3 step_1_function_call.py --n_runs 1 --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 3
```

As most models don't support parallel function calling natively, we customize an ad-hoc function calling methods. 

To run the experiment:

```bash
python3 step_1_function_call.py --n_runs 1 --model gpt-4 --policy_types reachability waypoint loadbalancing --batch_size 3 --adhoc
```

The experiment results will be stored in the directory named `results_function_call` by default.

## Developing Routing Algorithms
Traffic engineering is a critical yet complex problem in network management, particularly in large networks. Our experiment asks the models to create functions that compute routing paths based on specific network requirements (the shortest path, reachability, waypoint, load balancing). 

To run the experiment:

```bash
python3 step_2_code_gen.py --model gpt-4-1106 --n_runs 1 --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10; 
```

The experiment results will be stored in the directory named `results_code_gen` by default.

## Generating Low-level Configurations
This section explores the problem of transforming high-level requirements into detailed, low-level configurations suitable for installation on network devices. We handpicked four network scenarios publicly available in the [Kathará Network Emulator repository](https://github.com/KatharaFramework/Kathara-Labs). The selection encompasses the most widespread protocols and consists of two OSPF networks (one single-area network and one multi-area network), a RIP network, a BGP network featuring a basic peering between two routers, and a small fat-tree datacenter network running a made-up version of RIFT. All these scenarios leverage FRRouting as the routing suite. 

To run the experiment, you need to install [Docker](https://docs.docker.com/engine/install/) on your system. After that you can run:

```bash
python3 step_3_low_level.py --n_runs 1 --model gpt-4-turbo --mode rag --rag_chunk_size 9000
```

The experiment results will be stored in the directory named `results_low_level` by default.
