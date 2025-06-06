import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from constants import GameNames
from data_creation.scenario_creation.langgraph_creation.scenario_creation_graph import (
    a_create_scenario,
    build_scenario_creation_graph,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_creation/scenario_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration constants
TASK_TIMEOUT = 300  # 5 minutes per task
BATCH_TIMEOUT = 1800  # 30 minutes per batch
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries


class TaskTimeoutError(Exception):
    """Custom exception for task timeouts"""

    pass


class BatchTimeoutError(Exception):
    """Custom exception for batch timeouts"""

    pass


def get_existing_processed_personas(scenario_path_base: str) -> set:
    """
    Get a set of already processed personas based on existing scenario files.

    Args:
        scenario_path_base: Base path where scenario files are stored

    Returns:
        Set of processed persona job names (in original format, not camel case)
    """
    scenario_dir = Path(scenario_path_base)
    processed_personas = set()

    if scenario_dir.exists():
        for scenario_file in scenario_dir.glob("*.json"):
            # Extract filename without extension
            filename = scenario_file.stem

            # Convert from CamelCase back to original format
            # Insert spaces before capital letters (except the first character)
            original_persona = ""
            for i, char in enumerate(filename):
                if char.isupper() and i > 0:
                    # Only add space if the previous character was lowercase or
                    # this is not part of a sequence of capitals followed by lowercase
                    if filename[i - 1].islower() or (
                        i < len(filename) - 1
                        and filename[i + 1].islower()
                        and filename[i - 1].isupper()
                    ):
                        original_persona += " " + char.lower()
                    else:
                        original_persona += char.lower()
                else:
                    original_persona += char.lower()

            processed_personas.add(original_persona)

    return processed_personas


def filter_unprocessed_personas(persona_jobs: list, processed_personas: set) -> tuple:
    """
    Filter out already processed personas from the job list.

    Args:
        persona_jobs: List of all persona jobs
        processed_personas: Set of already processed persona jobs

    Returns:
        Tuple of (unprocessed_jobs, skipped_count)
    """
    unprocessed_jobs = []
    skipped_count = 0

    for persona_job in persona_jobs:
        if persona_job.lower() not in processed_personas:
            unprocessed_jobs.append(persona_job)
        else:
            skipped_count += 1

    return unprocessed_jobs, skipped_count


async def create_scenario_with_timeout(
    graph,
    game_name: str,
    participants: List[str],
    participant_jobs: List[str],
    config: dict,
    timeout: int = TASK_TIMEOUT,
) -> Optional[dict]:
    """
    Create scenario with timeout and retry logic.

    Args:
        graph: The scenario creation graph
        game_name: Name of the game
        participants: List of participant names
        participant_jobs: List of participant jobs
        config: Configuration for the task
        timeout: Timeout in seconds

    Returns:
        Scenario dict or None if failed
    """
    persona_job = participant_jobs[0]

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(
                f"Creating scenario for {persona_job}, attempt {attempt + 1}/{MAX_RETRIES + 1}"
            )

            # Create the task with timeout
            task = asyncio.create_task(
                a_create_scenario(
                    graph=graph,
                    game_name=game_name,
                    participants=participants,
                    participant_jobs=participant_jobs,
                    config=config,
                )
            )

            # Wait for task with timeout
            scenario = await asyncio.wait_for(task, timeout=timeout)

            if scenario:
                logger.info(f"Successfully created scenario for {persona_job}")
                return scenario
            else:
                logger.warning(f"Scenario creation returned None for {persona_job}")

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout creating scenario for {persona_job} (attempt {attempt + 1})"
            )
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying after {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"Max retries exceeded for {persona_job}")
                raise TaskTimeoutError(
                    f"Task timed out after {MAX_RETRIES + 1} attempts"
                )

        except Exception as e:
            logger.error(
                f"Error creating scenario for {persona_job} (attempt {attempt + 1}): {e}"
            )
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying after {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"Max retries exceeded for {persona_job}")
                raise

    return None


async def save_scenario_and_history(
    scenario: Optional[dict],
    scenario_graph,
    config: dict,
    persona_job_filename: str,
    scenario_path_base: str,
    history_path_base: str,
) -> bool:
    """
    Save scenario and history files.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Save history first
        history = []
        try:
            async for state in scenario_graph.aget_state_history(config):
                history.append(state)

            if history:
                history_path = (
                    f"{history_path_base}/{persona_job_filename}_history.json"
                )
                history_data = [
                    {
                        "values": state.values,
                        "metadata": state.metadata,
                        "created_at": state.created_at,
                    }
                    for i, state in enumerate(history)
                ]
                with open(history_path, "w") as f:
                    json.dump(history_data, f, indent=4)
                logger.info(f"Saved history for {persona_job_filename}")
        except Exception as e:
            logger.error(f"Error saving history for {persona_job_filename}: {e}")

        # Save scenario
        if scenario:
            scenario_path = f"{scenario_path_base}/{persona_job_filename}.json"
            with open(scenario_path, "w") as f:
                json.dump(scenario, f, indent=4)
            logger.info(f"Saved scenario for {persona_job_filename}")
            return True
        else:
            logger.warning(f"No scenario to save for {persona_job_filename}")
            return False

    except Exception as e:
        logger.error(f"Error saving files for {persona_job_filename}: {e}")
        return False


async def process_batch_with_timeout(
    persona_jobs: List[str],
    scenario_graph,
    game_name: str,
    scenario_path_base: str,
    history_path_base: str,
    batch_index: int,
    total_batches: int,
) -> Tuple[int, int]:
    """
    Process a batch of persona jobs with timeout and error handling.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    batch_start_time = time.time()
    successful_count = 0
    failed_count = 0

    logger.info(
        f"Starting batch {batch_index + 1}/{total_batches} with {len(persona_jobs)} jobs"
    )

    try:
        # Process each job individually to avoid one failure affecting others
        for i, persona_job in enumerate(persona_jobs):
            job_start_time = time.time()

            # Check batch timeout
            elapsed_batch_time = time.time() - batch_start_time
            if elapsed_batch_time > BATCH_TIMEOUT:
                logger.error(
                    f"Batch timeout exceeded ({elapsed_batch_time:.1f}s > {BATCH_TIMEOUT}s)"
                )
                raise BatchTimeoutError("Batch processing timed out")

            # Create unique thread_id for this job
            job_key = "-".join([persona_job, persona_job])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            thread_id = f"{game_name}_{job_key}_{timestamp}"

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50,
            }

            persona_job_camel_case = "".join(
                word.capitalize() for word in persona_job.split(" ")
            )

            try:
                # Create scenario with timeout and retry
                scenario = await create_scenario_with_timeout(
                    graph=scenario_graph,
                    game_name=game_name,
                    participants=["You", "Bob"],
                    participant_jobs=[persona_job, persona_job],
                    config=config,
                    timeout=TASK_TIMEOUT,
                )

                # Save scenario and history
                success = await save_scenario_and_history(
                    scenario=scenario,
                    scenario_graph=scenario_graph,
                    config=config,
                    persona_job_filename=persona_job_camel_case,
                    scenario_path_base=scenario_path_base,
                    history_path_base=history_path_base,
                )

                if success:
                    successful_count += 1
                    job_time = time.time() - job_start_time
                    logger.info(
                        f"Job {i+1}/{len(persona_jobs)} completed successfully in {job_time:.1f}s: {persona_job}"
                    )
                else:
                    failed_count += 1
                    logger.error(
                        f"Job {i+1}/{len(persona_jobs)} failed to save: {persona_job}"
                    )

            except (TaskTimeoutError, Exception) as e:
                failed_count += 1
                job_time = time.time() - job_start_time
                logger.error(
                    f"Job {i+1}/{len(persona_jobs)} failed after {job_time:.1f}s: {persona_job} - {e}"
                )

                # Continue with next job instead of stopping the entire batch
                continue

    except BatchTimeoutError as e:
        logger.error(f"Batch {batch_index + 1} timed out: {e}")
        # Return current counts even if batch timed out
        return successful_count, failed_count + (
            len(persona_jobs) - successful_count - failed_count
        )

    except Exception as e:
        logger.error(f"Unexpected error in batch {batch_index + 1}: {e}")
        return successful_count, failed_count + (
            len(persona_jobs) - successful_count - failed_count
        )

    batch_time = time.time() - batch_start_time
    logger.info(
        f"Batch {batch_index + 1} completed in {batch_time:.1f}s: {successful_count} successful, {failed_count} failed"
    )

    return successful_count, failed_count


async def main():
    """
    Main function with restart capability and robust error handling.
    """
    logger.info("Starting scenario generation process...")

    try:
        with open("data_creation/persona_jobs.jsonl", "r") as f:
            persona_jobs = [json.loads(line)["item"] for line in f][:10]
    except Exception as e:
        logger.error(f"Failed to load persona jobs: {e}")
        return

    game_name = GameNames.ESCALATION_GAME.value
    timestamp = datetime.now().strftime("%Y%m%d")
    scenario_path_base = f"data_creation/scenario_creation/langgraph_creation/scenarios/{game_name}_{timestamp}"
    history_path_base = f"data_creation/scenario_creation/langgraph_creation/histories/{game_name}_{timestamp}"

    # Create directories
    Path(scenario_path_base).mkdir(parents=True, exist_ok=True)
    Path(history_path_base).mkdir(parents=True, exist_ok=True)

    # Check for existing processed personas and filter them out
    logger.info("Checking for already processed personas...")
    processed_personas = get_existing_processed_personas(scenario_path_base)
    unprocessed_jobs, skipped_count = filter_unprocessed_personas(
        persona_jobs, processed_personas
    )

    logger.info(f"Total personas: {len(persona_jobs)}")
    logger.info(f"Already processed (skipped): {skipped_count}")
    logger.info(f"Remaining to process: {len(unprocessed_jobs)}")

    if len(unprocessed_jobs) == 0:
        logger.info("All personas have already been processed. Exiting.")
        return

    # Use filtered list for processing
    persona_jobs = unprocessed_jobs

    # Build graph once at the start
    logger.info("Building and compiling the scenario creation graph...")
    try:
        scenario_graph = build_scenario_creation_graph()
        logger.info("Graph built and compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to build scenario graph: {e}")
        return

    batch_size = 10
    num_batches = (len(persona_jobs) + batch_size - 1) // batch_size
    total_successful = 0
    total_failed = 0

    logger.info(
        f"Preparing to create {len(persona_jobs)} scenarios in {num_batches} batches..."
    )
    logger.info(
        f"Configuration: Task timeout={TASK_TIMEOUT}s, Batch timeout={BATCH_TIMEOUT}s, Max retries={MAX_RETRIES}"
    )

    # Process batches with individual error handling
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(persona_jobs))
        current_jobs = persona_jobs[start_index:end_index]

        batch_start_time = time.time()
        max_batch_retries = 2  # Allow batch retries for robustness

        for batch_retry in range(max_batch_retries + 1):
            try:
                logger.info(
                    f"Processing batch {batch_index + 1}/{num_batches} (attempt {batch_retry + 1}/{max_batch_retries + 1})"
                )

                # Process the batch with timeout handling
                successful_count, failed_count = await process_batch_with_timeout(
                    persona_jobs=current_jobs,
                    scenario_graph=scenario_graph,
                    game_name=game_name,
                    scenario_path_base=scenario_path_base,
                    history_path_base=history_path_base,
                    batch_index=batch_index,
                    total_batches=num_batches,
                )

                total_successful += successful_count
                total_failed += failed_count

                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Batch {batch_index + 1} completed in {batch_time:.1f}s: {successful_count} successful, {failed_count} failed"
                )

                # Break out of retry loop if batch completed
                break

            except Exception as e:
                logger.error(
                    f"Batch {batch_index + 1} failed on attempt {batch_retry + 1}: {e}"
                )

                if batch_retry < max_batch_retries:
                    logger.info(
                        f"Retrying batch {batch_index + 1} after {RETRY_DELAY * 2} seconds..."
                    )
                    await asyncio.sleep(RETRY_DELAY * 2)

                    # Rebuild graph on retry to handle potential corruption
                    try:
                        logger.info("Rebuilding scenario graph for retry...")
                        scenario_graph = build_scenario_creation_graph()
                        logger.info("Graph rebuilt successfully.")
                    except Exception as graph_error:
                        logger.error(f"Failed to rebuild graph: {graph_error}")
                        # If we can't rebuild the graph, mark all jobs in this batch as failed
                        total_failed += len(current_jobs)
                        break
                else:
                    logger.error(
                        f"Batch {batch_index + 1} failed after {max_batch_retries + 1} attempts"
                    )
                    total_failed += len(current_jobs)

        # Check for remaining unprocessed personas after each batch
        # This allows for true restart capability if the process is stopped and restarted
        remaining_processed_personas = get_existing_processed_personas(
            scenario_path_base
        )
        current_unprocessed, _ = filter_unprocessed_personas(
            persona_jobs, remaining_processed_personas
        )

        if len(current_unprocessed) == 0:
            logger.info("All remaining personas have been processed. Process complete.")
            break

        logger.info(
            f"Batch {batch_index + 1}/{num_batches} finished. Progress: {total_successful} successful, {total_failed} failed"
        )
        logger.info(f"Remaining unprocessed personas: {len(current_unprocessed)}")

    # Final summary
    total_processed = total_successful + total_failed
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total personas processed: {total_processed}")
    logger.info(f"Successful: {total_successful}")
    logger.info(f"Failed: {total_failed}")
    logger.info(
        f"Success rate: {(total_successful/total_processed*100):.1f}%"
        if total_processed > 0
        else "N/A"
    )

    # Check final state
    final_processed_personas = get_existing_processed_personas(scenario_path_base)
    final_unprocessed, _ = filter_unprocessed_personas(
        persona_jobs, final_processed_personas
    )

    if len(final_unprocessed) > 0:
        logger.warning(
            f"Process completed but {len(final_unprocessed)} personas remain unprocessed."
        )
        logger.info(
            "You can restart the script to continue processing the remaining personas."
        )
    else:
        logger.info("All personas have been successfully processed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(
            "Process interrupted by user. You can restart to continue from where it left off."
        )
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}")
        logger.info("You can restart the script to continue processing.")
