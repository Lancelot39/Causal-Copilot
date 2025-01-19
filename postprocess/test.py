import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('demo_data/20250114_000115/Abalone/output_graph/GES_global_state.pkl', 'rb') as f:
    global_state = pickle.load(f)
print(global_state.logging.global_state_logging)


import glob
global_state_files = glob.glob(f"{global_state.user_data.output_graph_dir}/*_global_state.pkl")
print(global_state_files)
global_state.logging.global_state_logging = []
for file in global_state_files:
    with open(file, 'rb') as f:
        temp_global_state = pickle.load(f)
        global_state.logging.global_state_logging.append(temp_global_state.algorithm.selected_algorithm)
print(global_state.logging.global_state_logging)
if len(global_state.logging.global_state_logging) > 1:
    algos = global_state.logging.global_state_logging
    print("Detailed analysis of which algorithm do you want to be included in the report?\n"
                            f"Please choose from the following: {', '.join(algos)}\n"
                            "Note that a comparision of all algorithms'results will be included in the report.")
