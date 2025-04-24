import asyncio
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from constants import GameNames
from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    a_create_scenario,
    build_scenario_creation_graph,
)


async def main():
    with open("data_creation/persona_jobs.jsonl", "r") as f:
        persona_jobs = [json.loads(line)["item"] for line in f][:10]

    game_name = GameNames.STAG_HUNT.value
    timestamp = datetime.now().strftime("%Y%m%d")
    scenario_path_base = f"data_creation/scenario_creation/langgraph_creation/scenarios/{game_name}_{timestamp}"
    history_path_base = f"data_creation/scenario_creation/langgraph_creation/histories/{game_name}_{timestamp}"

    # Create directories
    Path(scenario_path_base).mkdir(parents=True, exist_ok=True)
    Path(history_path_base).mkdir(parents=True, exist_ok=True)

    print("Building and compiling the scenario creation graph...")
    scenario_graph = build_scenario_creation_graph()
    print("Graph built and compiled.")

    batch_size = 10
    num_batches = (len(persona_jobs) + batch_size - 1) // batch_size

    print(
        f"Preparing to create {len(persona_jobs)} scenarios in {num_batches} batches..."
    )

    for batch_index in tqdm(range(num_batches), desc="Processing Batches"):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(persona_jobs))
        current_jobs = persona_jobs[start_index:end_index]

        batch_tasks = []
        batch_job_filenames = []
        batch_configs = []  # Store configs for history retrieval

        for persona_job in current_jobs:
            # Create unique thread_id for this job
            job_key = "-".join([persona_job, persona_job])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            thread_id = f"{game_name}_{job_key}_{timestamp}"

            # Include recursion_limit in config
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50,  # Increased from default to avoid recursion limit errors
            }
            batch_configs.append(config)

            task = asyncio.create_task(
                a_create_scenario(
                    graph=scenario_graph,
                    game_name=game_name,
                    participants=["You", "Bob"],
                    participant_jobs=[persona_job, persona_job],
                    config=config,  # Pass the config with recursion_limit
                )
            )
            batch_tasks.append(task)
            persona_job_camel_case = "".join(
                word.capitalize() for word in persona_job.split(" ")
            )
            batch_job_filenames.append(persona_job_camel_case)

        print(
            f"Starting batch {batch_index + 1}/{num_batches} with {len(batch_tasks)} tasks..."
        )
        batch_results = await tqdm_asyncio.gather(
            *batch_tasks, desc=f"Batch {batch_index + 1}"
        )
        print(f"Batch {batch_index + 1} finished.")

        # Process results and save files for the current batch
        print("Saving scenarios and histories for current batch...")
        for i, scenario in enumerate(batch_results):
            persona_job_filename = batch_job_filenames[i]
            config = batch_configs[i]

            # Get and save history
            try:
                history = []
                async for state in scenario_graph.aget_state_history(config):
                    history.append(state)

                if history:
                    history_path = (
                        f"{history_path_base}/{persona_job_filename}_history.json"
                    )
                    history_data = [
                        {
                            "step": i,
                            "values": state.values,
                            "metadata": state.metadata,
                            "created_at": state.created_at,
                        }
                        for i, state in enumerate(history)
                    ]
                    with open(history_path, "w") as f:
                        json.dump(history_data, f, indent=4)
            except Exception as e:
                print(f"Error saving history for {persona_job_filename}: {e}")

            if scenario:
                # Save scenario
                scenario_path = f"{scenario_path_base}/{persona_job_filename}.json"
                try:
                    with open(scenario_path, "w") as f:
                        json.dump(scenario, f, indent=4)
                except Exception as e:
                    print(f"Error saving scenario for {persona_job_filename}: {e}")

            else:
                # Determine the original job index for better error reporting if needed
                original_job_index = start_index + i
                failed_job_filename = batch_job_filenames[i]
                print(
                    f"Failed to create scenario for job index {original_job_index} (Filename: {failed_job_filename})"
                )
        print("Scenarios and histories for current batch saved.")


if __name__ == "__main__":
    asyncio.run(main())
