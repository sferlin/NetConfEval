#!/bin/bash

# Translation
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models gpt --policy_types reachability
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models gpt --policy_types reachability waypoint
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models gpt --policy_types reachability waypoint loadbalancing
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models codellama  --policy_types reachability
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models codellama  --policy_types reachability waypoint
python3 plot/step_1_plot_formal_spec_translation.py --results_path ../results_spec_translation --figures_path figures --models codellama  --policy_types reachability loadbalancing

# Conflict Detection
python3 plot/step_1_plot_formal_spec_conflict_detection.py --results_path ../results_conflict_detection --figures_path figures --metric f1_score --policy_types reachability waypoint loadbalancing
python3 plot/step_1_plot_formal_spec_conflict_detection.py --results_path ../results_conflict_detection --figures_path figures --metric recall --policy_types reachability waypoint loadbalancing
python3 plot/step_1_plot_formal_spec_conflict_detection.py --results_path ../results_conflict_detection --figures_path figures --combined --metric f1_score --policy_types reachability waypoint loadbalancing

# Conflict Distance
python3 plot/step_1_plot_formal_spec_conflict_heatmap.py --results_path ../results_conflict_distance --figures_path figures --policy_types reachability waypoint loadbalancing --model gpt-3.5-0613

# Function Call
python3 plot/step_1_plot_function_call.py --results_path ../results_function_call --figures_path figures --model gpt-4-1106 --type native
python3 plot/step_1_plot_function_call.py --results_path ../results_function_call --figures_path figures --model gpt-4-1106 --type adhoc

# Code Generation
python3 plot/step_2_plot_code_gen.py --results_path ../results_code_gen --figures_path figures --model gpt-4-1106

# Low-Level Configurations
python3 plot/step_3_plot_low_level.py --results_path ../results_low_level --figures_path figures --model gpt-4-1106