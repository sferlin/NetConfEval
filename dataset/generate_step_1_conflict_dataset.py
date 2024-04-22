import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.common.utils import *

policies_to_batch_sizes = {
    "reachability": [1, 2, 5, 10, 20, 25, 50, 100],
    "reachability,waypoint": [2, 4, 10, 20, 50, 100],
    "reachability,waypoint,loadbalancing": [3, 9, 33, 99],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, required=False, default=5)
    parser.add_argument("--policy_file", type=str, required=False,
                        default=os.path.join("..", "assets", "step_1_policies.csv"))
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

    with open(os.path.join(args.results_path, "step_1_spec_conflict.jsonl"), "w") as f:
        for policy_types, batch_sizes in policies_to_batch_sizes.items():
            policy_types = SortedSet(policy_types.split(','))
            dataset = load_csv(args.policy_file, policy_types)

            n_policy_types = len(policy_types)
            max_n_requirements = max(batch_sizes) * n_policy_types

            for it in range(0, args.n_runs):
                logging.info(f"Generating iteration n. {it + 1}...")
                samples = pick_sample(max_n_requirements, dataset, it, policy_types)

                for batch_size in batch_sizes:
                    flag_conflict = True
                    logging.info(f"Generating data "
                                 f"with {batch_size * n_policy_types} batch size (iteration n. {it + 1})...")
                    chunk_samples = list(chunk_list(samples, batch_size * n_policy_types))
                    if len(chunk_samples) == 1:
                        chunk_new = copy.deepcopy(chunk_samples[0])
                        chunk_samples.append(chunk_new)

                    for i, sample in enumerate(chunk_samples):
                        logging.info(f"Generating sample with {batch_size * n_policy_types} "
                                     f"batch size on chunk {i} (iteration n. {it + 1})...")

                        if flag_conflict:
                            insert_conflict(sample)

                        result_row = {
                            'iteration': it,
                            'chunk': i,
                            'batch_size': batch_size,
                            'n_policy_types': n_policy_types,
                            'max_n_requirements': max_n_requirements,
                            'conflict_exist': flag_conflict,
                            'human_language': convert_to_human_language(sample),
                            'expected': json.dumps(transform_sample_to_expected(sample))
                        }

                        flag_conflict = not flag_conflict

                        f.write(json.dumps(result_row) + "\n")
                        f.flush()


if __name__ == "__main__":
    main(parse_args())
