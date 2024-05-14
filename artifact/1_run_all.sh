#!/bin/bash

while getopts r: flag
do
  case "${flag}" in
    r) n_runs=${OPTARG};;
  esac
done

if [ -z "$n_runs" ]; then
  echo 'You need to specify the number of runs.' >&2
  exit 1
fi

# Translation
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-4-1106 --policy_types reachability --batch_size 1 2 5 10 20 50 100

python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-0613  --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-0613 --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-0613 --policy_types reachability --batch_size 1 2 5 10 20 50 100

python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-finetuned --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-finetuned --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model gpt-3.5-finetuned --policy_types reachability --batch_size 1 2 5 10 20 50 100

python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-7b-instruct --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-7b-instruct --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-7b-instruct --policy_types reachability --batch_size 1 2 5 10 20 50 100

python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-13b-instruct --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-13b-instruct --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_formal_spec_translation.py --results_path $(pwd)/results_spec_translation --n_run $n_runs --model codellama-13b-instruct --policy_types reachability --batch_size 1 2 5 10 20 50 100

# Conflict Detection
python3 ../netconfeval/step_1_formal_spec_conflict_detection.py --results_path $(pwd)/results_conflict_detection --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --n_run $n_runs --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_conflict_detection.py --results_path $(pwd)/results_conflict_detection --model gpt-4 --policy_types reachability waypoint loadbalancing --n_run $n_runs --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_conflict_detection.py --results_path $(pwd)/results_conflict_detection --model gpt-3.5-0613 --policy_types reachability waypoint loadbalancing --n_run $n_runs  --batch_size 1 3 11 33
python3 ../netconfeval/step_1_formal_spec_conflict_detection.py --results_path $(pwd)/results_conflict_detection --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --n_run $n_runs --combined --batch_size 1 3 11 33

# Conflict Distance
python3 ../netconfeval/step_1_formal_spec_conflict_distance.py --results_path $(pwd)/results_conflict_distance --n_run $n_runs --model gpt-3.5-0613 --policy_types reachability waypoint loadbalancing --batch_size 11

# Function Call
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability --batch_size 1 2 5 10 20 50 100

python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability waypoint --batch_size 1 2 5 10 25 50 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-4-1106 --policy_types reachability --batch_size 1 2 5 10 20 50 100 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-3.5-1106 --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-3.5-1106 --policy_types reachability waypoint --batch_size 1 2 5 10 25 50 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model gpt-3.5-1106 --policy_types reachability --batch_size 1 2 5 10 20 50 100 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model codellama-7b-instruct --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model codellama-7b-instruct --policy_types reachability waypoint --batch_size 1 2 5 10 25 50 --adhoc
python3 ../netconfeval/step_1_function_call.py --results_path $(pwd)/results_function_call --n_run $n_runs --model codellama-7b-instruct --policy_types reachability --batch_size 1 2 5 10 20 50 100 --adhoc

# Code Generation
python3 ../netconfeval/step_2_code_gen.py --results_path $(pwd)/results_code_gen --model gpt-4-1106 --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10
python3 ../netconfeval/step_2_code_gen.py --results_path $(pwd)/results_code_gen --model gpt-4-1106 --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --feedback
python3 ../netconfeval/step_2_code_gen.py --results_path $(pwd)/results_code_gen --model gpt-4-1106 --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --prompts no_detail
python3 ../netconfeval/step_2_code_gen.py --results_path $(pwd)/results_code_gen --model gpt-4-1106 --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --prompts no_detail --feedback

# Low-Level Configurations
python3 ../netconfeval/step_3_low_level.py --results_path $(pwd)/results_low_level --n_run $n_runs --model gpt-4-1106 --mode none
python3 ../netconfeval/step_3_low_level.py --results_path $(pwd)/results_low_level --n_run $n_runs --model gpt-4-1106 --mode full
python3 ../netconfeval/step_3_low_level.py --results_path $(pwd)/results_low_level --n_run $n_runs --model gpt-4-1106 --mode idx
python3 ../netconfeval/step_3_low_level.py --results_path $(pwd)/results_low_level --n_run $n_runs --model gpt-4-1106 --mode rag --rag_chunk_size 9000
