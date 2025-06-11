import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
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
# Suppress verbose HTTP logs from Azure SDK and httpx
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes per task
DEFAULT_BATCH_TIMEOUT = 1800  # 30 minutes per batch
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 5  # seconds between retries
DEFAULT_MAX_ITERATIONS = 5  # Maximum iterations for scenario refinement


class TaskTimeoutError(Exception):
    """Custom exception for task timeouts"""

    pass


class BatchTimeoutError(Exception):
    """Custom exception for batch timeouts"""

    pass


class ScenarioCreationConfig:
    """Configuration class for scenario creation parameters"""

    def __init__(self, **kwargs):
        """Initialize configuration with defaults and overrides"""

        # Set defaults
        self.persona_jobs_file = kwargs.get(
            "persona_jobs_file", "data_creation/persona_jobs_all.jsonl"
        )
        self.game_name = kwargs.get("game_name", GameNames.ESCALATION_GAME.value)
        self.num_personas = kwargs.get("num_personas", 20)
        self.batch_size = kwargs.get("batch_size", 20)
        self.task_timeout = kwargs.get("task_timeout", DEFAULT_TASK_TIMEOUT)
        self.batch_timeout = kwargs.get("batch_timeout", DEFAULT_BATCH_TIMEOUT)
        self.max_retries = kwargs.get("max_retries", DEFAULT_MAX_RETRIES)
        self.retry_delay = kwargs.get("retry_delay", DEFAULT_RETRY_DELAY)
        self.max_iterations = kwargs.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        self.llm_model = kwargs.get("llm_model", "gpt-4.1-mini")
        self.llm_temp_propose = kwargs.get("llm_temp_propose", 0.7)
        self.llm_temp_verify = kwargs.get("llm_temp_verify", 0.3)
        self.llm_temp_payoff = kwargs.get("llm_temp_payoff", 0.1)
        self.azure_mode = kwargs.get("azure_mode", True)
        self.debug_mode = kwargs.get("debug_mode", True)
        self.debug_num_personas = kwargs.get("debug_num_personas", 2)
        self.output_dir = kwargs.get(
            "output_dir", "data_creation/scenario_creation/langgraph_creation"
        )
        self.resume = kwargs.get("resume", False)
        self.verbose = kwargs.get("verbose", False)
        self.verification_nodes = kwargs.get(
            "verification_nodes", ["narrative", "preference_order", "pay_off"]
        )

    def to_dict(self):
        """Convert configuration to dictionary"""
        return self.__dict__


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
    timeout: int,
    max_retries: int,
    retry_delay: int,
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
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        Scenario dict or None if failed
    """
    persona_job = participant_jobs[0]

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"Creating scenario for {persona_job}, attempt {attempt + 1}/{max_retries + 1}"
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

            if scenario is None:
                logger.warning(f"Scenario creation returned None for {persona_job}")
                continue

            return scenario

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout creating scenario for {persona_job} (attempt {attempt + 1})"
            )
            if attempt < max_retries:
                logger.info(f"Retrying after {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Max retries exceeded for {persona_job}")
                raise TaskTimeoutError(
                    f"Task timed out after {max_retries + 1} attempts"
                )

        except Exception as e:
            logger.error(
                f"Error creating scenario for {persona_job} (attempt {attempt + 1}): {e}"
            )
            break

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


async def process_single_job(
    persona_job,
    scenario_graph,
    game_name,
    scenario_path_base,
    history_path_base,
    config: ScenarioCreationConfig,
):
    job_start_time = time.time()
    job_key = "-".join([persona_job, persona_job])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    thread_id = f"{game_name}_{job_key}_{timestamp}"

    graph_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }

    persona_job_camel_case = "".join(
        word.capitalize() for word in persona_job.split(" ")
    )

    try:
        scenario = await create_scenario_with_timeout(
            graph=scenario_graph,
            game_name=game_name,
            participants=(
                ["You", "Bob"]
                if GameNames.from_string(game_name).is_symmetric()
                else ["Alice", "Bob"]
            ),
            participant_jobs=[persona_job, persona_job],
            config=graph_config,
            timeout=config.task_timeout,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )
        success = await save_scenario_and_history(
            scenario=scenario,
            scenario_graph=scenario_graph,
            config=graph_config,
            persona_job_filename=persona_job_camel_case,
            scenario_path_base=scenario_path_base,
            history_path_base=history_path_base,
        )
        job_time = time.time() - job_start_time
        if success:
            logger.info(f"Job completed successfully in {job_time:.1f}s: {persona_job}")
            return True
        else:
            logger.error(f"Job failed to save: {persona_job}")
            return False
    except Exception as e:
        job_time = time.time() - job_start_time
        logger.error(f"Job failed after {job_time:.1f}s: {persona_job} - {e}")
        return False


async def process_batch_with_timeout(
    persona_jobs: List[str],
    scenario_graph,
    game_name: str,
    scenario_path_base: str,
    history_path_base: str,
    batch_index: int,
    total_batches: int,
    config: ScenarioCreationConfig,
) -> Tuple[int, int]:
    """
    Process a batch of persona jobs in parallel using asyncio.gather.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    batch_start_time = time.time()
    logger.info(
        f"Starting batch {batch_index + 1}/{total_batches} with {len(persona_jobs)} jobs"
    )

    # Create a task for each job
    tasks = [
        process_single_job(
            persona_job,
            scenario_graph,
            game_name,
            scenario_path_base,
            history_path_base,
            config,
        )
        for persona_job in persona_jobs
    ]

    # Run all jobs in parallel with a progress bar
    results = await tqdm_asyncio.gather(
        *tasks, desc=f"Processing Batch {batch_index + 1}/{total_batches}"
    )
    successful_count = sum(results)
    failed_count = len(results) - successful_count

    batch_time = time.time() - batch_start_time
    logger.info(
        f"Batch {batch_index + 1} completed in {batch_time:.1f}s: {successful_count} successful, {failed_count} failed"
    )

    return successful_count, failed_count


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Configurable LangGraph Scenario Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_scenario_langgraph.py --game-name "Prisoners_Dilemma" --num-personas 50
  python create_scenario_langgraph.py --config config/scenario_creation_config.yaml --debug
  python create_scenario_langgraph.py --llm-model "gpt-4" --max-iterations 10 --verbose
        """,
    )

    # Input parameters
    parser.add_argument(
        "--persona-jobs-file",
        type=str,
        default="data_creation/persona_jobs_all.jsonl",
        help="Persona jobs file path",
    )
    parser.add_argument(
        "--game-name",
        type=str,
        default=GameNames.ESCALATION_GAME.value,
        help="Game name",
    )
    parser.add_argument(
        "--num-personas", type=int, default=20, help="Number of personas to process"
    )

    # Processing parameters
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size for parallel processing"
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT,
        help="Timeout per task (seconds)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=DEFAULT_BATCH_TIMEOUT,
        help="Timeout per batch (seconds)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per task",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=DEFAULT_RETRY_DELAY,
        help="Delay between retries (seconds)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum refinement iterations",
    )

    # LLM parameters
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4.1-mini", help="LLM model name"
    )
    parser.add_argument(
        "--llm-temp-propose",
        type=float,
        default=0.7,
        help="Temperature for scenario proposal",
    )
    parser.add_argument(
        "--llm-temp-verify",
        type=float,
        default=0.3,
        help="Temperature for verification",
    )
    parser.add_argument(
        "--llm-temp-payoff",
        type=float,
        default=0.1,
        help="Temperature for payoff validation",
    )
    parser.add_argument(
        "--azure-mode",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use Azure OpenAI",
    )

    # Verification parameters
    parser.add_argument(
        "--verification-nodes",
        nargs="+",
        default=["narrative", "preference_order", "pay_off"],
        choices=["narrative", "preference_order", "pay_off"],
        help="Which verification nodes to use in the graph.",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_creation/scenario_creation/langgraph_creation",
        help="Output base directory",
    )

    # Runtime parameters
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to run a small sample case",
    )
    parser.add_argument(
        "--debug-num-personas",
        type=int,
        default=2,
        help="Number of personas to process in debug mode. Overrides --num-personas.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous run"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--auto-debug",
        action="store_true",
        help="Automatically run a debug batch before the full run. If debug succeeds, proceed to full run.",
    )

    return parser.parse_args()


def setup_configuration(
    auto_debug_mode=False, debug_mode=False, debug_num_personas=2, output_dir=None
) -> ScenarioCreationConfig:
    """Setup configuration from command line arguments or override for debug/auto-debug."""
    args = parse_arguments()
    # Allow override for auto-debug
    if auto_debug_mode:
        # Use a separate output dir for debug
        debug_output_dir = (output_dir or args.output_dir) + "_debug"
        return ScenarioCreationConfig(
            persona_jobs_file=args.persona_jobs_file,
            game_name=args.game_name,
            num_personas=debug_num_personas,
            batch_size=min(args.batch_size, debug_num_personas),
            task_timeout=args.task_timeout,
            batch_timeout=args.batch_timeout,
            max_retries=0,
            retry_delay=args.retry_delay,
            max_iterations=args.max_iterations,
            llm_model=args.llm_model,
            llm_temp_propose=args.llm_temp_propose,
            llm_temp_verify=args.llm_temp_verify,
            llm_temp_payoff=args.llm_temp_payoff,
            azure_mode=(args.azure_mode == "true"),
            debug_mode=True,
            debug_num_personas=debug_num_personas,
            output_dir=debug_output_dir,
            resume=False,
            verbose=args.verbose,
            verification_nodes=args.verification_nodes,
        )
    else:
        return ScenarioCreationConfig(
            persona_jobs_file=args.persona_jobs_file,
            game_name=args.game_name,
            num_personas=args.num_personas,
            batch_size=args.batch_size,
            task_timeout=args.task_timeout,
            batch_timeout=args.batch_timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_iterations=args.max_iterations,
            llm_model=args.llm_model,
            llm_temp_propose=args.llm_temp_propose,
            llm_temp_verify=args.llm_temp_verify,
            llm_temp_payoff=args.llm_temp_payoff,
            azure_mode=(args.azure_mode == "true"),
            debug_mode=debug_mode,
            debug_num_personas=args.debug_num_personas,
            output_dir=args.output_dir,
            resume=args.resume,
            verbose=args.verbose,
            verification_nodes=args.verification_nodes,
        )


def configure_logging(config: ScenarioCreationConfig):
    """Configure logging based on configuration"""
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting scenario generation process...")

    if config.verbose:
        logger.info("Configuration:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")


def load_persona_jobs(config: ScenarioCreationConfig) -> List[str]:
    """Load persona jobs from file"""
    try:
        with open(config.persona_jobs_file, "r") as f:
            all_persona_jobs = [json.loads(line)["item"] for line in f]
            if config.num_personas > 0:
                return all_persona_jobs[: config.num_personas]
            else:
                return all_persona_jobs
    except Exception as e:
        logger.error(
            f"Failed to load persona jobs from {config.persona_jobs_file}: {e}"
        )
        raise


def setup_directories(config: ScenarioCreationConfig) -> Tuple[str, str]:
    """Setup output directories and return paths"""
    game_name = config.game_name
    timestamp = datetime.now().strftime("%Y%m%d")
    scenario_path_base = f"{config.output_dir}/scenarios/{game_name}_{timestamp}"
    history_path_base = f"{config.output_dir}/histories/{game_name}_{timestamp}"

    # Create directories
    Path(scenario_path_base).mkdir(parents=True, exist_ok=True)
    Path(history_path_base).mkdir(parents=True, exist_ok=True)

    return scenario_path_base, history_path_base


def check_existing_work(persona_jobs: List[str], scenario_path_base: str) -> List[str]:
    """Check for existing work and filter unprocessed personas"""
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
        return []

    return unprocessed_jobs


def build_scenario_graph(config: ScenarioCreationConfig):
    """Build and configure the scenario creation graph"""
    logger.info("Building and compiling the scenario creation graph...")
    try:
        # Build scenario graph
        logger.info("Config arguments before building scenario graph:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")
        scenario_graph = build_scenario_creation_graph(
            debug_mode=config.debug_mode,
            llm_config={
                "model": config.llm_model,
                "temp_propose": config.llm_temp_propose,
                "temp_verify": config.llm_temp_verify,
                "temp_payoff": config.llm_temp_payoff,
                "azure_mode": config.azure_mode,
                "max_iterations": config.max_iterations,
            },
            verification_nodes=config.verification_nodes,
        )
        logger.info("Graph built and compiled successfully.")
        return scenario_graph
    except Exception as e:
        logger.error(f"Failed to build scenario graph: {e}")
        raise


async def process_single_batch(
    batch_index: int,
    current_jobs: List[str],
    scenario_graph,
    config: ScenarioCreationConfig,
    scenario_path_base: str,
    history_path_base: str,
    num_batches: int,
) -> Tuple[int, int]:
    """Process a single batch with retry logic"""
    batch_start_time = time.time()
    max_batch_retries = config.max_retries

    for batch_retry in range(max_batch_retries + 1):
        try:
            logger.info(
                f"Processing batch {batch_index + 1}/{num_batches} (attempt {batch_retry + 1}/{max_batch_retries + 1})"
            )

            # Process the batch with timeout handling
            successful_count, failed_count = await process_batch_with_timeout(
                persona_jobs=current_jobs,
                scenario_graph=scenario_graph,
                game_name=config.game_name,
                scenario_path_base=scenario_path_base,
                history_path_base=history_path_base,
                batch_index=batch_index,
                total_batches=num_batches,
                config=config,
            )

            batch_time = time.time() - batch_start_time
            logger.info(
                f"Batch {batch_index + 1} completed in {batch_time:.1f}s: {successful_count} successful, {failed_count} failed"
            )

            return successful_count, failed_count

        except Exception as e:
            logger.error(
                f"Batch {batch_index + 1} failed on attempt {batch_retry + 1}: {e}"
            )

            if batch_retry < max_batch_retries:
                logger.info(
                    f"Retrying batch {batch_index + 1} after {config.retry_delay * 2} seconds..."
                )
                await asyncio.sleep(config.retry_delay * 2)

                # Rebuild graph on retry to handle potential corruption
                try:
                    logger.info("Rebuilding scenario graph for retry...")
                    scenario_graph = build_scenario_graph(config)
                    logger.info("Graph rebuilt successfully.")
                except Exception as graph_error:
                    logger.error(f"Failed to rebuild graph: {graph_error}")
                    # If we can't rebuild the graph, mark all jobs in this batch as failed
                    return 0, len(current_jobs)
            else:
                logger.error(
                    f"Batch {batch_index + 1} failed after {max_batch_retries + 1} attempts"
                )
                return 0, len(current_jobs)

    return 0, len(current_jobs)


async def process_all_batches(
    persona_jobs: List[str],
    scenario_graph,
    config: ScenarioCreationConfig,
    scenario_path_base: str,
    history_path_base: str,
) -> Tuple[int, int]:
    """Process all batches of persona jobs"""
    batch_size = config.batch_size
    num_batches = (len(persona_jobs) + batch_size - 1) // batch_size
    total_successful = 0
    total_failed = 0

    logger.info(
        f"Preparing to create {len(persona_jobs)} scenarios in {num_batches} batches..."
    )
    logger.info(
        f"Configuration: Task timeout={config.task_timeout}s, Batch timeout={config.batch_timeout}s, Max retries={config.max_retries}"
    )

    # Process batches with individual error handling
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(persona_jobs))
        current_jobs = persona_jobs[start_index:end_index]

        successful_count, failed_count = await process_single_batch(
            batch_index=batch_index,
            current_jobs=current_jobs,
            scenario_graph=scenario_graph,
            config=config,
            scenario_path_base=scenario_path_base,
            history_path_base=history_path_base,
            num_batches=num_batches,
        )

        total_successful += successful_count
        total_failed += failed_count

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

    return total_successful, total_failed


def print_final_summary(
    total_successful: int,
    total_failed: int,
    persona_jobs: List[str],
    scenario_path_base: str,
):
    """Print final summary and check completion status"""
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


async def main():
    """
    Main function with restart capability and robust error handling.
    """
    args = parse_arguments()
    if getattr(args, "auto_debug", False):
        # 1. Run debug batch
        debug_config = setup_configuration(
            auto_debug_mode=True,
            debug_mode=True,
            debug_num_personas=args.debug_num_personas,
            output_dir=args.output_dir,
        )
        configure_logging(debug_config)
        logger.info("=" * 60)
        logger.info("AUTO-DEBUG MODE: Running debug batch before full run.")
        logger.info("=" * 60)
        try:
            all_persona_jobs = load_persona_jobs(debug_config)
            scenario_path_base, history_path_base = setup_directories(debug_config)
            persona_jobs_to_process = all_persona_jobs
            scenario_graph = build_scenario_graph(debug_config)
            total_successful, total_failed = await process_all_batches(
                persona_jobs=persona_jobs_to_process,
                scenario_graph=scenario_graph,
                config=debug_config,
                scenario_path_base=scenario_path_base,
                history_path_base=history_path_base,
            )
            print_final_summary(
                total_successful,
                total_failed,
                persona_jobs_to_process,
                scenario_path_base,
            )
            if total_failed > 0:
                logger.error("AUTO-DEBUG: Debug batch failed. Aborting full run.")
                return
            logger.info("AUTO-DEBUG: Debug batch succeeded. Proceeding to full run.")
        except Exception as e:
            logger.error(f"AUTO-DEBUG: Debug batch encountered an error: {e}")
            logger.error("Aborting full run.")
            return
        # 2. Run full batch
        full_config = setup_configuration(
            auto_debug_mode=False,
            debug_mode=False,
            output_dir=args.output_dir,
        )
        configure_logging(full_config)
        logger.info("=" * 60)
        logger.info("AUTO-DEBUG MODE: Running full batch after successful debug batch.")
        logger.info("=" * 60)
        try:
            all_persona_jobs = load_persona_jobs(full_config)
            scenario_path_base, history_path_base = setup_directories(full_config)
            persona_jobs_to_process = check_existing_work(
                all_persona_jobs, scenario_path_base
            )
            if not persona_jobs_to_process:
                logger.info("No new personas to process. Exiting.")
                return
            scenario_graph = build_scenario_graph(full_config)
            total_successful, total_failed = await process_all_batches(
                persona_jobs=persona_jobs_to_process,
                scenario_graph=scenario_graph,
                config=full_config,
                scenario_path_base=scenario_path_base,
                history_path_base=history_path_base,
            )
            print_final_summary(
                total_successful,
                total_failed,
                persona_jobs_to_process,
                scenario_path_base,
            )
        except Exception as e:
            logger.error(f"AUTO-DEBUG: Full batch encountered an error: {e}")
            return
        return
    # Normal (non-auto-debug) mode
    config = setup_configuration(debug_mode=args.debug)
    configure_logging(config)
    try:
        if config.debug_mode:
            logger.info("=" * 60)
            logger.info("DEBUG MODE: Running a small test case.")
            logger.info("=" * 60)
            # Override config for a small, fast run
            config.num_personas = config.debug_num_personas
            config.batch_size = min(config.batch_size, config.debug_num_personas)
            config.max_retries = 0
            logger.info(
                f"Debug Overrides: num_personas={config.num_personas}, batch_size={config.batch_size}"
            )

        # Load persona jobs based on config
        all_persona_jobs = load_persona_jobs(config)

        # Setup directories
        scenario_path_base, history_path_base = setup_directories(config)

        # Determine which jobs to run
        persona_jobs_to_process = []
        if config.debug_mode:
            logger.info(
                "Running a small sample for debug mode and skipping existing work check."
            )
            persona_jobs_to_process = all_persona_jobs
        elif config.resume:
            persona_jobs_to_process = check_existing_work(
                all_persona_jobs, scenario_path_base
            )
        else:
            logger.info(
                "Starting a fresh run. Previously processed personas will be skipped."
            )
            persona_jobs_to_process = check_existing_work(
                all_persona_jobs, scenario_path_base
            )

        if not persona_jobs_to_process:
            logger.info("No new personas to process. Exiting.")
            return

        # Build scenario graph
        logger.info("Config arguments before building scenario graph:")
        for key, value in config.to_dict().items():
            logger.info(f"  {key}: {value}")
        scenario_graph = build_scenario_graph(config)

        # Process all batches
        total_successful, total_failed = await process_all_batches(
            persona_jobs=persona_jobs_to_process,
            scenario_graph=scenario_graph,
            config=config,
            scenario_path_base=scenario_path_base,
            history_path_base=history_path_base,
        )

        # Print final summary
        print_final_summary(
            total_successful,
            total_failed,
            persona_jobs_to_process,
            scenario_path_base,
        )

        if config.debug_mode:
            logger.info("=" * 60)
            logger.info("DEBUG MODE run finished.")
            logger.info(f"Check outputs in {scenario_path_base}")
            logger.info(
                "To run the full generation, run the script again without the --debug flag."
            )
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        raise


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
