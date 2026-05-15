import os
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.llm]

if os.getenv("RUN_LLM_TESTS") != "1":
    pytest.skip("set RUN_LLM_TESTS=1 to run LLM selection integration checks", allow_module_level=True)

import re
import sys
import json
import argparse
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from causal_discovery.tests.user_simulation import BASE_MODEL as SIM_BASE_MODEL

BASELINE_MODEL = "gpt-4o-mini"
NUM_SIMULATIONS = 100
NUM_CHUNK_SIMULATIONS = 25  # Process 10 simulations at a time to avoid rate limits


def generate_dummy_args():
    # Create a dummy args object with necessary attributes for simulation.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default="data/dataset/sachs/sachs.csv", help='Path to dataset file')
    parser.add_argument('--initial_query', type=str, default="selected algorithm: FGES; filter: continuous", help='Initial algorithm query')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (skip heavy preprocessing)')
    # Modes: normal, skip, simulate
    parser.add_argument('--mode', type=str, choices=["normal", "skip", "simulate"], default="simulate", help='Mode of operation')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Turn on metrics evaluation if ground truth is available')
    # Additional arguments required by the global initialization
    parser.add_argument('--organization', type=str, default="org-Xa9VGT8plP28JBRKtqBRjG5l")
    parser.add_argument('--project', type=str, default="proj_t78xpJomilJJu5qtuOK4vWfR")
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--simulation_mode', type=str, default="offline")
    parser.add_argument('--data_mode', type=str, default="real")
    parser.add_argument('--demo_mode', type=bool, default=False)

    # Quick test of simulation output
    args = parser.parse_args()
    args.apikey = os.getenv("OPENAI_API_KEY")
    args.num_users = NUM_CHUNK_SIMULATIONS
    return args

async def simulate_task(task_id):
    """
    Generates one user simulation data by calling the simulate_user_query function.
    The fake_data (a pandas DataFrame) is converted to a JSON string.
    """
    try:
        from causal_discovery.tests.user_simulation import simulate_user_query
    except ImportError:
        raise ImportError("Could not import simulate_user_query from user_simulation.py.")
    args = generate_dummy_args()
    simulations = await asyncio.to_thread(simulate_user_query, args)
    valid_simulations = []
    # simulations is now a list of simulation dicts
    for i, sim in enumerate(simulations):
         sim.pop("fake_data", None)
         sim["simulation_id"] = f"{task_id+i}"
         valid_simulations.append(sim)
    return valid_simulations

async def select_algorithm(simulation):
    """
    For a given simulation, use the o1-mini baseline to select the appropriate algorithm
    from the available candidate list in algorithm.wrappers.__all__, and explain the choice.
    """
    import json
    import random
    try:
        import causal_discovery.wrappers as wrappers
    except ImportError:
        raise ImportError("Could not from causal_discovery.wrappers module.")

    args = generate_dummy_args()
    # Get the list of candidate algorithm names
    candidate_algorithms = wrappers.__all__

    user_query = simulation.get("initial_query", "")
    statistics = simulation.get("statistics", {})

    # Load algorithm tagging information
    tagging_path = os.path.join(os.path.dirname(__file__), "..", "context", "algos", "tagging.txt")
    with open(tagging_path, "r") as f:
        algorithm_tags = f.read()

    # Randomly sample cuda_visible
    cuda_visible = random.choice([True, False])
    cuda_warning = "" if cuda_visible else "\nCurrent machine doesn't support CUDA, do not choose any GPU-powered algorithms."

    # Build a prompt that includes the user query, statistics, candidate algorithms and algorithm tags.
    prompt = (
        f"Given the following user query:\n{user_query}\n\n"
        f"And the following statistics:\n{json.dumps(statistics)}\n\n"
        f"Candidate algorithms:\n{json.dumps(candidate_algorithms)}\n\n"
        f"Algorithm descriptions and tags:\n{algorithm_tags}\n\n"
        "Please choose the top 3 most suitable causal discovery algorithms for the data user provided and explain why these algorithms were chosen over the others.\n\n"
        f"{cuda_warning}\n\n"
        "Please return the selected algorithms and the explanation in a JSON object. Wrap the JSON object in <json>...</json> tags. For"
        "example:\n"
        "<json>\n"
        "{{"
        "  'selected_algorithms': ['...', '...', '...'], "
        "  'selection_response': '...'"
        "}}"
        "</json>"
    )

    # Use asyncopenai's asynchronous chat completion with the o1-mini model.
    try:
        args = generate_dummy_args()
        client = AsyncOpenAI()
        response = await client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for selection testing.",
            json_response=True
        )
        answer = response.choices[0].message.content.strip()
        answer = re.search(r'<json>(.*?)</json>', answer, re.DOTALL).group(1)
        answer = json.loads(answer)
    except Exception as e:
        answer = f"Error during selection: {str(e)}"
    
    selection = {
        "simulation_id": simulation.get("simulation_id"),
        "initial_query": user_query,
        "statistics": statistics,
        "selected_algorithms": answer.get("selected_algorithms"),
        "selection_response": answer.get("selection_response"),
        "cuda_visible": cuda_visible
    }
    return selection

async def process_simulation_chunk(start_idx):
    """Process a chunk of simulations"""
    results = await simulate_task(start_idx)
    return results

async def process_selection_chunk(simulations):
    """Process a chunk of selections"""
    selection_tasks = [select_algorithm(sim) for sim in simulations]
    return await asyncio.gather(*selection_tasks, return_exceptions=True)

async def baseline():
    # Step 1: Generate N user simulation data in chunks
    if os.path.exists(f"simulations_{SIM_BASE_MODEL}.json"):
        with open(f"simulations_{SIM_BASE_MODEL}.json", "r") as sim_file:
            valid_simulations = json.load(sim_file)
    else:
        valid_simulations = []
        for start_idx in tqdm(range(0, NUM_SIMULATIONS, NUM_CHUNK_SIMULATIONS), desc="Generating user simulations..."):
            chunk_results = await process_simulation_chunk(start_idx)
            
            # Process simulation results, handling any exceptions
            for idx, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    print(f"Simulation task {start_idx + idx} generated an exception: {result}")
                else:
                    valid_simulations.append(result)

        # Save all simulation data into a JSON file
        with open(f"simulations_{SIM_BASE_MODEL}.json", "w") as sim_file:
            json.dump(valid_simulations, sim_file, indent=4)

    # Step 2: Process selections in chunks
    final_selections = []
    for i in range(0, len(valid_simulations), NUM_CHUNK_SIMULATIONS):
        chunk = valid_simulations[i:i + NUM_CHUNK_SIMULATIONS]
        selection_results = await process_selection_chunk(chunk)
        
        # Process selection results, handling any exceptions
        for idx, result in enumerate(selection_results):
            if isinstance(result, Exception):
                print(f"Selection task for simulation {chunk[idx].get('simulation_id')} generated an exception: {result}")
            else:
                final_selections.append(result)

    # Save the selection responses into a JSON file
    with open(f"selection_results_{BASELINE_MODEL}.json", "w") as sel_file:
        json.dump(final_selections, sel_file, indent=4)

if __name__ == '__main__':
    asyncio.run(baseline())
