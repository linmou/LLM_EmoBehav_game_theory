import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from games.game import Game
from games.game_configs import get_game_config


# State definition
class ScenarioCreationState(TypedDict):
    # Input requirements
    game_name: str
    participants: List[str]
    participant_jobs: Optional[List[str]]
    # Working data
    scenario_draft: Optional[Dict[str, Any]]
    feedback: Optional[List[str]]
    iteration_count: int
    # Output
    final_scenario: Optional[Dict[str, Any]]
    converged: bool


# Initialize the LLM
def get_llm(temperature=1.4, json_mode=True):
    """Get the LLM to use for the scenario creation."""
    # Using ChatOpenAI, but you can replace with other LLM providers
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None,
    )


# Node functions
def propose_scenario(state: ScenarioCreationState) -> ScenarioCreationState:
    """Create a scenario draft based on the requirements."""
    game_name = state["game_name"]
    participants = state["participants"]
    participant_jobs = state.get("participant_jobs", [])
    iteration_count = state["iteration_count"]
    previous_draft = state.get("scenario_draft")
    previous_feedback = state.get("feedback", [])

    # Get game config and example scenario
    game_cfg = get_game_config(game_name)
    game = Game(
        name=game_name,
        scenario_class=game_cfg["scenario_class"],
        decision_class=game_cfg["decision_class"],
        payoff_matrix=game_cfg["payoff_matrix"],
    )
    example_scenario = game.example_scenario
    payoff_description = game.payoff_matrix.get_natural_language_description(
        participants
    )

    # Create system prompt
    system_prompt = f"""
    You are a scenario creator for game theory experiments.
    Your task is to create a realistic scenario that masks the underlying game theory structure ({game_name}).
    The scenario should be unique and ensure participants won't immediately recognize the underlying game structure.
    
    Current iteration: {iteration_count + 1}
    
    Always return your response as a valid JSON object matching the example format.
    """

    # Create human prompt
    if iteration_count == 0:
        # First iteration
        human_prompt = f"""
        Create a unique scenario that masks the {game_name} structure under a new context that readers can not recognize it as a {game_name} at first glance.
        
        The created scenario should contain the following participants:
        Participants: {participants}
        {"Participant jobs: " + str(participant_jobs) if participant_jobs else ""}
        Do not talk anything about participants' previous relationship.
        
        The created scenario should follow the payoff matrix:
        {payoff_description}
        Do not use the specific digits in the payoff matrix, but please make sure the payoff in your created scenario is at the proper level.
         
        You should follow this example format, note that when writing payoff_matrix, you should first use digital payoff, then write the natural language description of the payoff in this scenario.
        {json.dumps(example_scenario, indent=2)}
        
        Write in English.Return the scenario as a valid JSON object.
        """
    else:
        # Subsequent iterations with feedback
        human_prompt = f"""
        Refine the previous scenario draft based on the feedback provided.
        
        Previous draft:
        {json.dumps(previous_draft, indent=2)}
        
        Feedback:
        {" ".join(f"{fid+1}. {feedback}" for fid, feedback in enumerate(previous_feedback))}
       
        Participants: {participants}
        {"Participant jobs: " + str(participant_jobs) if participant_jobs else ""} 
        
        Follow this example format:
        {json.dumps(example_scenario, indent=2)}
        
        Return the improved scenario as a valid JSON object.
        """

    # Get LLM response
    llm = get_llm(temperature=0.7, json_mode=True)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)

    # With json_mode=True, response.content is already a JSON string
    scenario_draft = response.content
    if isinstance(scenario_draft, str):
        try:
            scenario_draft = json.loads(scenario_draft)
            scenario_draft["payoff_description"] = payoff_description
        except json.JSONDecodeError:
            scenario_draft = {"error": "Failed to parse JSON from response"}

    # Update the state
    return {
        **state,
        "scenario_draft": scenario_draft,
        "iteration_count": iteration_count + 1,
        "feedback": [],  # Reset feedback for next iteration
    }


def verify_scenario(state: ScenarioCreationState) -> ScenarioCreationState:
    """Verify if the created scenario matches the requirements and provide feedback."""
    game_name = state["game_name"]
    scenario_draft = state["scenario_draft"]
    players = state["participants"]
    # Get game config for verification
    game_cfg = get_game_config(game_name)
    game = Game(
        name=game_name,
        scenario_class=game_cfg["scenario_class"],
        decision_class=game_cfg["decision_class"],
        payoff_matrix=game_cfg["payoff_matrix"],
    )
    example_scenario = game.example_scenario

    # Create system prompt
    system_prompt = """
    You are a critical reviewer for game theory scenarios.
    Your task is to verify if the proposed scenario correctly implements the required game theory structure
    while masking its true nature from participants.
    Provide detailed feedback on what needs to be improved.
    
    Format your response as a JSON object with exactly two fields:
    - "feedback": an array of feedback points as strings
    - "converged": a boolean, true if no changes needed, false otherwise
    """

    # Create human prompt
    human_prompt = f"""
    Please review this scenario for a {game_name} game:
    {json.dumps(scenario_draft, indent=2)}
    
    
    Evaluate the scenario on these criteria:
    1. Does it properly mask the {game_name} structure?
    2. Does the scenario make sense and realistic?
    3. Are the participants' named {state["participants"]} and jobs {state["participant_jobs"]} correctly assigned?
    4. Are the behavior choices correctly representing the game's strategies?
    5. For {players[0]}, the rank of the payoff should follow the order: {game.payoff_matrix.ordered_payoff_leaves[0]}
    6. For {players[1]}, the rank of the payoff should follow the order: {game.payoff_matrix.ordered_payoff_leaves[1]}
    7. Do not talk anything about participants' previous relationship.
    
    
    Return a list of specific feedback points for improvement. 
    If the scenario is perfect and requires no changes, state that it has converged.
    Format your response as a JSON with two fields:
    - "feedback": a list of feedback points
    - "converged": true if no changes needed, false otherwise
    """

    # Get LLM response
    llm = get_llm(
        temperature=0.3, json_mode=True
    )  # Lower temperature for more consistent evaluation
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)

    # With json_mode=True, response.content is already a JSON string
    result = response.content
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {
                "feedback": [
                    "Error parsing verification result, please check the scenario format"
                ],
                "converged": False,
            }

    # Ensure the result has the expected structure
    if (
        not isinstance(result, dict)
        or "feedback" not in result
        or "converged" not in result
    ):
        result = {
            "feedback": ["Invalid verification result format"],
            "converged": False,
        }

    # Update the state
    return {**state, "feedback": result["feedback"], "converged": result["converged"]}


def should_continue(state: ScenarioCreationState) -> str:
    """Decide whether to continue refining or finalize the scenario."""
    # Check if we've converged or reached max iterations
    if state["converged"] or state["iteration_count"] >= 5:
        # Save the final scenario
        return "finalize"
    else:
        # Continue refining
        return "refine"


def finalize_scenario(state: ScenarioCreationState) -> ScenarioCreationState:
    """Finalize the scenario and save it."""
    if not state["converged"]:
        return {**state, "final_scenario": None}

    game_name = state["game_name"]
    scenario_draft = state["scenario_draft"]
    auto_save_path = state.get("auto_save_path", None)
    file_path = None
    if auto_save_path:
        # Create directory if it doesn't exist
        scenario_dir = Path(auto_save_path)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = f"{game_name}_{timestamp}"
        file_path = scenario_dir / f"{scenario_name}.json"

        # Save the scenario
        with open(file_path, "w") as f:
            json.dump(scenario_draft, f, indent=2)

    # Update the state
    return {**state, "final_scenario": scenario_draft, "scenario_path": str(file_path)}


# Build the graph
def build_scenario_creation_graph():
    """Build the scenario creation graph."""
    # Create the graph
    graph = StateGraph(ScenarioCreationState)

    # Add nodes
    graph.add_node("propose_scenario", propose_scenario)
    graph.add_node("verify_scenario", verify_scenario)
    graph.add_node("finalize_scenario", finalize_scenario)

    # Add edges
    graph.add_edge("propose_scenario", "verify_scenario")
    graph.add_conditional_edges(
        "verify_scenario",
        should_continue,
        {"refine": "propose_scenario", "finalize": "finalize_scenario"},
    )
    graph.add_edge("finalize_scenario", END)

    # Set the entry point
    graph.set_entry_point("propose_scenario")

    # Compile the graph with checkpointer
    # MemorySaver is used for in-memory checkpointing
    memory = MemorySaver()

    # The configurable fields (thread_id, checkpoint_ns, or checkpoint_id)
    # will be provided when invoking the graph
    return graph.compile(checkpointer=memory)


# Function to run the scenario creation process
def create_scenario(
    game_name: str,
    participants: List[str],
    participant_jobs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a scenario for the specified game and participants.

    Args:
        game_name: The name of the game (e.g., "Prisoners_Dilemma")
        participants: List of participant names
        participant_jobs: Optional list of participant jobs

    Returns:
        The created scenario
    """
    # Initialize the graph
    graph = build_scenario_creation_graph()

    # Initialize the state
    initial_state: ScenarioCreationState = {
        "game_name": game_name,
        "participants": participants,
        "participant_jobs": participant_jobs,
        "scenario_draft": None,
        "feedback": [],
        "iteration_count": 0,
        "final_scenario": None,
        "converged": False,
    }

    # Create config with thread_id for the checkpointer
    config = {
        "configurable": {
            "thread_id": f"{game_name}_{'-'.join(participant_jobs)}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    }

    # Run the graph with the config
    final_state = graph.invoke(initial_state, config)

    # Return the final scenario
    return final_state["final_scenario"]


async def a_create_scenario(
    graph: Any,  # Add graph as an argument
    game_name: str,
    participants: List[str],
    participant_jobs: Optional[List[str]] = None,
    auto_save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a scenario for the specified game and participants using a pre-compiled graph.

    Args:
        graph: The pre-compiled LangGraph object.
        game_name: The name of the game (e.g., "Prisoners_Dilemma")
        participants: List of participant names
        participant_jobs: Optional list of participant jobs

    Returns:
        The created scenario or None if an error occurs during execution.
    """
    # Initialize the state
    initial_state: ScenarioCreationState = {
        "game_name": game_name,
        "participants": participants,
        "participant_jobs": participant_jobs,
        "scenario_draft": None,
        "feedback": [],
        "iteration_count": 0,
        "final_scenario": None,
        "converged": False,
        "auto_save_path": auto_save_path,
    }

    # Create config with thread_id for the checkpointer
    # Ensure participant_jobs is not None and is a list for joining
    job_key = "-".join(participant_jobs) if participant_jobs else "no_jobs"
    timestamp = (
        __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )  # Added microseconds for uniqueness
    thread_id = f"{game_name}_{job_key}_{timestamp}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run the graph with the config
        final_state = await graph.ainvoke(initial_state, config)
        # Return the final scenario
        return final_state.get("final_scenario")  # Use .get for safety
    except Exception as e:
        # Log the error or handle it as needed
        print(f"Error invoking graph for thread {thread_id}: {e}")
        # Potentially log the initial_state as well for debugging
        # print(f"Initial state for failed invocation: {initial_state}")
        return None  # Indicate failure


if __name__ == "__main__":
    # Example usage
    scenario = create_scenario(
        game_name="Prisoners_Dilemma",
        participants=["You", "Bob"],
        participant_jobs=["immigration lawyer", "immigration lawyer"],
    )
    print(json.dumps(scenario, indent=2))
