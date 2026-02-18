import json
import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict, Union
import runpy

# Optional Azure imports; fall back gracefully if not installed.
try:
    from azure.core.credentials import AzureKeyCredential  # noqa: F401
except Exception:
    AzureKeyCredential = None  # type: ignore
try:
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
except Exception:
    AzureAIChatCompletionsModel = None  # type: ignore
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
except Exception:  # pragma: no cover - minimal fallbacks for test environments
    AIMessage = HumanMessage = SystemMessage = object  # type: ignore
    JsonOutputParser = object  # type: ignore
# langchain_openai may require azure deps for AzureChatOpenAI; fall back gracefully.
try:
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
except Exception:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:  # pragma: no cover - allow tests without langchain_openai
        ChatOpenAI = object  # type: ignore
    AzureChatOpenAI = None  # type: ignore
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - allow importing for unit tests
    MemorySaver = object  # type: ignore
    END = "END"  # type: ignore
    START = "START"  # type: ignore

from pydantic import BaseModel

from constants import GameType
from games.game import Game
from games.game_configs import get_game_config
from games.payoff_matrices import PayoffMatrix

PAYOFF_VALIDATION_QUESTION_FORMAT = (
    "- If {behavior_description}, please first imagine how much possible that these two behaviors happen together in the scenario,"
    "Then image the outcome of participants behaviors, "
    "as mentioned in the scenario description, after participants decision, reward for participant 1 is '{p1_outcome}' and reward for participant 2 is '{p2_outcome}', is this outcome match your imagination? "
    "Is the outcome a plausible consequence in the context of the scenario? "
    "In your response, please first analyze the probability of these two behaviors happen together in the scenario, then write your imagination of the outcome and analysis if it matches the outcome described in the scenario description, if everything is reasonable, finally answer YES, otherwise answer NO."
)


# Define a custom reducer to replace the value instead of adding
def replace_reducer(existing_value, new_value):
    """Reducer that simply returns the new value, overwriting the old."""
    return new_value


# State definition
class ScenarioCreationState(TypedDict):
    # Input requirements
    game_name: str
    participants: List[str]
    # Diplomacy raw input (optional)
    raw_record: Optional[Dict[str, Any]]
    map_summary: Optional[str]
    # Working data
    scenario_draft: Optional[Dict[str, Any]]
    gradient_options: Optional[List[Dict[str, Any]]]
    narrative_feedback: Annotated[
        List[str], replace_reducer
    ]  # Feedback from narrative verification
    behavior_feedback: Annotated[
        List[str], replace_reducer
    ]  # Feedback from behavior neutrality verification
    preference_feedback: Annotated[
        List[str], replace_reducer
    ]  # Feedback from mechanics verification
    payoff_feedback: Annotated[
        List[str], replace_reducer
    ]  # Feedback from payoff validation
    iteration_count: int
    # Output
    final_scenario: Optional[Dict[str, Any]]
    narrative_converged: bool  # Convergence flag from narrative verification
    behavior_converged: bool  # Convergence flag from behavior neutrality verification
    preference_converged: bool  # Convergence flag from mechanics verification
    payoff_converged: bool  # Convergence flag from payoff validation
    all_converged: Optional[
        bool
    ]  # Flag indicating if all verification steps have converged
    auto_save_path: Optional[str]  # Path for auto-saving scenarios


from api_configs import AZURE_OPENAI_CONFIG, OAI_CONFIG

# Global LLM configuration
_global_llm_config = {
    "model": "gpt-4.1-mini",
    "temp_propose": 0.7,
    "temp_verify": 0.3,
    "temp_payoff": 0.1,
    "azure_mode": True,
    "max_iterations": 8,
}


def set_global_llm_config(llm_config: dict):
    """Set global LLM configuration for all nodes"""
    global _global_llm_config
    _global_llm_config.update(llm_config)


def get_global_llm_config():
    """Get current global LLM configuration"""
    return _global_llm_config.copy()


# Helper used to derive the dominant participant label
def _primary_participant_label(players: List[Any]) -> str:
    raw = players[0] if players else "Player 1"
    if isinstance(raw, dict):
        raw = raw.get("name") or "Player 1"
    if not isinstance(raw, str):
        raw = str(raw)
    return raw


# Initialize the LLM
def get_llm(model=None, temperature=None, json_mode=True, azure_mode=None):
    """Get LLM instance with configuration from global config if not specified"""
    if model is None:
        model = _global_llm_config["model"]
    if temperature is None:
        temperature = _global_llm_config["temp_propose"]  # Default temperature
    if azure_mode is None:
        azure_mode = _global_llm_config["azure_mode"]

    if azure_mode:
        # If Azure libs unavailable, fall back to OpenAI-compatible client
        if AzureAIChatCompletionsModel is None:
            # Use explicit API key and base_url from OAI_CONFIG if available
            try:
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=OAI_CONFIG.get("api_key"),
                    base_url=OAI_CONFIG.get("base_url"),
                )
            except Exception:
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                )
        else:
            return AzureChatOpenAI(
                api_version=AZURE_OPENAI_CONFIG["api_version"],
                azure_endpoint=AZURE_OPENAI_CONFIG["azure_endpoint"],
                api_key=AZURE_OPENAI_CONFIG["api_key"],
                deployment_name=model,
                temperature=temperature,
            )
    else:
        # Ensure environment variables for OpenAI SDK are set from api_configs
        try:
            if isinstance(OAI_CONFIG, dict):
                ak = OAI_CONFIG.get("api_key")
                bu = OAI_CONFIG.get("base_url")
                if ak:
                    os.environ.setdefault("OPENAI_API_KEY", ak)
                if bu:
                    # Some SDKs read OPENAI_BASE_URL, some read OPENAI_API_BASE; set both.
                    os.environ.setdefault("OPENAI_BASE_URL", bu)
                    os.environ.setdefault("OPENAI_API_BASE", bu)
        except Exception:
            pass
        try:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=OAI_CONFIG.get("api_key"),
                base_url=OAI_CONFIG.get("base_url"),
            )
        except Exception:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
            )


# Node functions
def prepare_diplomacy_from_raw(state: ScenarioCreationState) -> ScenarioCreationState:
    """
    If 'raw_record' exists, set participants strictly from involved powers in the map record.
    Do not use persona profiles. Expect exactly two participants for payoff verification;
    if more are listed, take the first two; if fewer, fallback conservatively.
    """
    raw = state.get("raw_record")
    participants = []
    nl_lines = []
    if isinstance(raw, dict):
        # Participants from provided field (strict)
        inv = raw.get("involved_powers") or raw.get("involved_power") or []
        if isinstance(inv, list):
            # Keep order; filter empties; take at most two
            cleaned = [str(x) for x in inv if str(x).strip()]
            if len(cleaned) >= 2:
                participants = cleaned[:2]
            elif len(cleaned) == 1:
                participants = [cleaned[0], "Opponent"]
        # Phase mapping (e.g., S1901M)
        phase = str(raw.get("phase", ""))
        season = "Spring" if phase.startswith("S") else "Fall" if phase.startswith("F") else "Unknown"
        year = "".join(ch for ch in phase if ch.isdigit()) or "????"
        subphase = "Orders" if phase.endswith("M") else "Unknown"
        dest = raw.get("destination") or "an area"
        nl_lines.append(f"Phase: {season} {year} ({subphase}). Contesting {dest}.")
        # # Moves to destination
        # od = raw.get("orders_to_dest") or []
        # if od:
        #     mv = "; ".join(f"{o.get('power','?')} {o.get('order','')}" for o in od)
        #     nl_lines.append(f"Orders to destination: {mv}.")
        if not participants:
            orders = raw.get("orders_to_dest") or raw.get("moves") or []
            if isinstance(orders, list):
                seen = set()
                for order in orders:
                    if not isinstance(order, dict):
                        continue
                    power = str(order.get("power", "")).strip()
                    if power and power not in seen:
                        participants.append(power)
                        seen.add(power)
                    if len(participants) == 2:
                        break
                if len(participants) == 1:
                    participants.append("Opponent")
        # Units near dest per power
        und = raw.get("units_near_dest") or {}
        if isinstance(und, dict) and und:
            parts = []
            for pw, lst in und.items():
                lst2 = ", ".join(lst) if isinstance(lst, list) else str(lst)
                parts.append(f"{pw}: {lst2}")
            nl_lines.append(f"Units near destination: " + " | ".join(parts) + ".")
        
        # # Outcome/flags
        # had_sup = raw.get("had_support_to_dest", False)
        # nxt = raw.get("next_phase_outcome") or "unknown"
        # msgs = raw.get("phase_has_msgs_about_dest", False)
        # nl_lines.append(f"Support to destination present: {'yes' if had_sup else 'no'}.")
        # nl_lines.append(f"Next phase outcome tag: {nxt}.")
        # nl_lines.append(f"Messages about destination this phase: {'yes' if msgs else 'no'}.")
         
    map_summary = "\n".join(nl_lines) if nl_lines else None
    return {
        **state,
        "participants": participants or ["Player 1", "Player 2"],
        "map_summary": map_summary,
    }

def propose_scenario(state: ScenarioCreationState) -> ScenarioCreationState:
    """Create a scenario draft based on the requirements."""
    game_name = state["game_name"]
    participants = state["participants"]
    iteration_count = state["iteration_count"]
    previous_draft = state.get("scenario_draft")
    # Combine feedback from all verification steps for refinement
    previous_feedback = []
    for key in state.keys():
        if key.endswith("_feedback"):
            feedback_category = key.split("_")[0]
            if not state[feedback_category + "_converged"]:
                feedback_items = state[key]

                # Ensure feedback_items is a list
                if not isinstance(feedback_items, list):
                    print(f"Warning: {key} is not a list, converting to list")
                    feedback_items = [feedback_items]

                # Ensure all feedback items are strings
                for i, item in enumerate(feedback_items):
                    if isinstance(item, str):
                        previous_feedback.append(item)
                    elif isinstance(item, dict):
                        # Convert dict to string representation
                        print(f"Warning: Found dict in {key}[{i}]: {item}")
                        previous_feedback.append(str(item))
                    else:
                        # Convert any other type to string
                        print(f"Warning: Found {type(item)} in {key}[{i}]: {item}")
                        previous_feedback.append(str(item))

    previous_feedback = "\n".join(previous_feedback)

    # Get game config and example scenario
    game_cfg = get_game_config(game_name)
    game = Game(
        name=game_name,
        scenario_class=game_cfg["scenario_class"],
        decision_class=game_cfg["decision_class"],
        payoff_matrix=game_cfg["payoff_matrix"],
    )
    # TODO:
    example_scenario = game.scenario_class.diplomacy_example() if hasattr(game.scenario_class, 'diplomacy_example') else game.example_scenario
    print( "is using diplomacy example: ", hasattr(game.scenario_class, 'diplomacy_example'))
    payoff_description = ""
    if isinstance(game.payoff_matrix, PayoffMatrix):
        payoff_description = game.payoff_matrix.get_natural_language_description(
            participants
            if game.game_type == GameType.SIMULTANEOUS
            else ["player 1", "player 2"]
        )

    further_instructions = (
        f"An important feature of the {game_name} is : {game_cfg.get('game_description', '')}"
        if game_cfg.get("game_description", "")
        else ""
    )
    # Create system prompt
    system_prompt = f"""
    You are a scenario creator for game theory experiments.
    Your task is to create a realistic scenario that masks the underlying game theory structure ({game_name}).
    The scenario should be unique and ensure participants won't immediately recognize the underlying game structure.
    
    Current iteration: {iteration_count + 1}
    
    Always return your response in English and as a valid JSON object matching the example format.
    """

    # Create human prompt
    if iteration_count == 0:
        # First iteration
        # Diplomacy enrichment: if map_summary is provided, include it to ground the narrative.
        raw_ctx = state.get("map_summary") or ""
        human_prompt = f"""
        Create a unique scenario that masks the {game_name} structure under a new context that readers can not recognize it as a {game_name} at first glance.
        {further_instructions}
        
        The created scenario should contain the following participants:
        Participants: {participants}
        Do not talk anything about participants' previous relationship.
        If the following Diplomacy map summary is provided, ground the scenario details in it (units, regions, contested destination, typical outcomes like bounces/support/cover/tempo), but do not include any negotiation:
        {raw_ctx}
        
        {f'The created scenario should follow the payoff matrix: {payoff_description}' if payoff_description else ''}
        Do not contain digits as payoff in the description. Ensure the consequence descriptions are concrete and plausible from the map context.
        

        You should follow this example format, note that when writing payoff_matrix_description, you should first use digital payoff, then write the natural language description of the payoff in this scenario.
        {json.dumps(example_scenario, indent=2)}
                    
        When you design the behavior choices, keys should strictly follow the example format, don't use other keys.
        When you create the behavior choices, do not use ambiguous words like 'collaberate, cooperate' or 'defect', please provide specific behavior and more details settings in the scenario to make the behavior -> outcome causal chain robust and reasonable.
        When you create the behavior choices, do not use words with moral tendency like 'generous, selfish, volunatry, altruistic, etc.', don't use adjective or adverb to describe the behavior, just describe the behavior in a neutral way.
        
        {game.scenario_class.specific_prompt() if hasattr(game.scenario_class, 'specific_prompt') else ''}
        
        Write in English. Return the scenario as a valid JSON object.
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
        
        
        Follow this example format:
        {json.dumps(example_scenario, indent=2)}
        
        Return the improved scenario as a valid JSON object.
        """

    # Get LLM response
    llm = get_llm(temperature=_global_llm_config["temp_propose"], json_mode=True)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    if AzureAIChatCompletionsModel is not None and isinstance(llm, AzureAIChatCompletionsModel):
        response = llm.invoke(messages)
    else:
        response = llm.invoke(messages, response_format={"type": "json_object"})

    # With json_mode=True, response.content is already a JSON string
    scenario_draft_content = response.content
    parsed_scenario_draft = None
    if isinstance(scenario_draft_content, str):
        try:
            parsed_scenario_draft = json.loads(scenario_draft_content)
            # Ensure payoff_description is added if parsing is successful
            if isinstance(parsed_scenario_draft, dict):
                parsed_scenario_draft["payoff_description"] = payoff_description
            else:
                # Handle case where JSON is valid but not a dictionary
                parsed_scenario_draft = {
                    "error": "Parsed JSON is not an object",
                    "raw_content": scenario_draft_content,
                }

        except json.JSONDecodeError:
            parsed_scenario_draft = {
                "error": "Failed to parse JSON from response",
                "raw_content": scenario_draft_content,
            }
    elif isinstance(
        scenario_draft_content, dict
    ):  # Handle cases where the LLM might directly return a dict
        parsed_scenario_draft = scenario_draft_content
        parsed_scenario_draft["payoff_description"] = payoff_description
    else:
        parsed_scenario_draft = {
            "error": "Unexpected response type from LLM",
            "raw_content": str(scenario_draft_content),
        }

    # Enforce participants to match the involved powers (state.participants) and sanitize names in texts
    try:
        parsed_scenario_draft = normalize_scenario_participants(
            parsed_scenario_draft or {}, participants
        )
        parsed_scenario_draft = sanitize_names(parsed_scenario_draft, participants)
        parsed_scenario_draft = normalize_scenario_title(
            parsed_scenario_draft, participants, state.get("raw_record") or {}
        )
    except Exception:
        pass

    # Update the state
    return {
        **state,
        "scenario_draft": parsed_scenario_draft,
        "iteration_count": iteration_count + 1,
        "narrative_feedback": [],  # Reset feedback for next iteration
        "preference_feedback": [],  # Unused in diplomacy flow
        "payoff_feedback": [],  # Reset feedback for next iteration
        "narrative_converged": False,  # Reset convergence flags
        "preference_converged": True,  # Skip preference in diplomacy flow
        "payoff_converged": False,
    }


def normalize_scenario_participants(
    scenario_draft: Dict[str, Any], players: List[str]
) -> Dict[str, Any]:
    """
    Ensure scenario_draft['participants'] matches the provided players exactly.
    Produces a list of dicts: [{"name": players[0]}, {"name": players[1]}]
    """
    fixed = dict(scenario_draft or {})
    if not players or len(players) < 2:
        return fixed
    fixed["participants"] = [{"name": players[0]}, {"name": players[1]}]
    return fixed

def sanitize_names(scenario_draft: Dict[str, Any], players: List[str]) -> Dict[str, Any]:
    """
    Replace 'Alice'/'Ben' tokens in free text with provided players, to avoid leaking example names.
    """
    if not isinstance(scenario_draft, dict) or len(players) < 2:
        return scenario_draft
    p0, p1 = players[0], players[1]
    def _swap(s: Any) -> Any:
        if isinstance(s, str):
            return s.replace("Alice", p0).replace("Ben", p1)
        return s
    out = dict(scenario_draft)
    if "description" in out:
        out["description"] = _swap(out["description"])
    if "payoff_matrix_description" in out and isinstance(out["payoff_matrix_description"], dict):
        pmd = {}
        for k, v in out["payoff_matrix_description"].items():
            if isinstance(v, list):
                pmd[k] = [_swap(x) for x in v]
            else:
                pmd[k] = _swap(v)
        out["payoff_matrix_description"] = pmd
    return out

def normalize_scenario_title(
    scenario_draft: Dict[str, Any], players: List[str], raw: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Derive a deterministic, grounded title to avoid irrelevant/copy artifacts.
    Format: Diplomacy_<P1>_vs_<P2>_<DEST>_<Season><Year>_<Subphase>
    """
    fixed = dict(scenario_draft or {})
    if not players or len(players) < 2:
        return fixed
    dest = str((raw or {}).get("destination") or "AREA")
    phase = str((raw or {}).get("phase") or "")
    season = "Spring" if phase.startswith("S") else "Fall" if phase.startswith("F") else "Season"
    year = "".join(ch for ch in phase if ch.isdigit()) or "Year"
    subphase = "Orders" if phase.endswith("M") else "Phase"
    fixed["scenario"] = f"Diplomacy_{players[0]}_vs_{players[1]}_{dest}_{season}{year}_{subphase}"
    return fixed

def verify_narrative(state: ScenarioCreationState) -> Dict[str, Any]:
    """Verify the narrative aspects of the scenario."""
    game_name = state["game_name"]
    scenario_draft = state["scenario_draft"]
    players = state["participants"]

    # Handle potential error in scenario draft
    if not scenario_draft or "error" in scenario_draft:
        error_msg = (
            scenario_draft.get("error") if scenario_draft else "scenario_draft is None"
        )
        raw_content = (
            scenario_draft.get("raw_content")
            if scenario_draft
            else "No raw content available"
        )

        return {
            "narrative_feedback": [
                "Cannot verify narrative due to error in scenario draft generation.",
                f"Error: {error_msg}",
                f"Raw Content: {raw_content}",
            ],
            "narrative_converged": False,
        }

    # Create system prompt
    system_prompt = """
    You are a critical reviewer focusing on the narrative quality of game theory scenarios.
    Your task is to verify if the proposed scenario is realistic, coherent, and correctly uses participant details,
    while also making an initial assessment of how well it masks the underlying game.
    Provide detailed feedback on what needs to be improved regarding the narrative.

    Format your response as a JSON object with exactly two fields:
    - "feedback": an array of narrative-specific feedback points as strings
    - "converged": a boolean, true if narrative aspects need no changes, false otherwise
    """

    # Create human prompt
    human_prompt = f"""
    Please review the narrative aspects of this scenario for a {game_name} game:
    {json.dumps(scenario_draft, indent=2)}

    Evaluate the scenario based on these narrative criteria:
    1. Is the scenario description realistic and coherent? Does the story make sense? If the story is almost the same as the example scenario, no need to further improve it. ('You' is a proper name, no need to replace it) 
    2. Are the participants' names ({players}) correctly and plausibly integrated? Please strictly follow the {players}.
    3. Does the scenario avoid mentioning any negotiation or prior relationships between participants?
    4. Do the `behavior_choices` accurately reflect neutral 'withdraw' and 'escalate' strategies in this map context?
    5. Does the narrative effectively mask the underlying {game_name} structure? (Initial check)

    Return a list of specific feedback points for narrative improvement.
    If the narrative aspects are perfect and require no changes, state that it has converged.
    Format your response as a JSON with two fields:
    - "feedback": a list of feedback points
    - "converged": true if no changes needed, false otherwise
    """

    # Get LLM response
    llm = get_llm(temperature=_global_llm_config["temp_verify"], json_mode=True)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    if AzureAIChatCompletionsModel is not None and isinstance(llm, AzureAIChatCompletionsModel):
        response = llm.invoke(messages)
    else:
        response = llm.invoke(messages, response_format={"type": "json_object"})

    # Process response
    result_content = response.content
    result = {}
    if isinstance(result_content, str):
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            result = {
                "feedback": ["Error parsing narrative verification result"],
                "converged": False,
            }
    elif isinstance(result_content, dict):
        result = result_content
    else:
        result = {
            "feedback": ["Unexpected format for narrative verification result"],
            "converged": False,
        }

    # Ensure the result has the expected structure
    if (
        not isinstance(result, dict)
        or "feedback" not in result
        or "converged" not in result
    ):
        result = {
            "feedback": ["Invalid narrative verification result format"],
            "converged": False,
        }

    # Return only the fields we want to update, not the entire state
    return {
        "narrative_feedback": result["feedback"],
        "narrative_converged": result["converged"],
    }


def verify_preference_order(state: ScenarioCreationState) -> Dict[str, Any]:
    return {
        "preference_feedback": [],
        "preference_converged": True,
    }

    """Verify the game mechanics aspects of the scenario."""
    game_name = state["game_name"]
    scenario_draft = state["scenario_draft"]
    players = state["participants"]

    # Handle potential error in scenario draft directly
    if not scenario_draft or "error" in scenario_draft:
        error_msg = (
            scenario_draft.get("error") if scenario_draft else "scenario_draft is None"
        )
        return {
            "preference_feedback": [
                f"Cannot verify preference order due to error in scenario draft: {error_msg}"
            ],
            "preference_converged": False,
        }

    # Get game config for verification
    try:
        game_cfg = get_game_config(game_name)
        game = Game(
            name=game_name,
            scenario_class=game_cfg["scenario_class"],
            decision_class=game_cfg["decision_class"],
            payoff_matrix=game_cfg["payoff_matrix"],
        )
    except Exception as e:
        return {
            "preference_feedback": [
                f"Error loading game config for preference validation: {e}"
            ],
            "preference_converged": False,
        }

    # Check structure needed for this verification
    if "payoff_matrix" not in scenario_draft or not isinstance(
        scenario_draft["payoff_matrix_description"], dict
    ):
        return {
            "preference_feedback": [
                "Scenario draft missing 'payoff_matrix' or it is not a dict."
            ],
            "preference_converged": False,
        }

    # Create system prompt
    system_prompt = """
    You are a critical reviewer specializing in the game-theoretic mechanics of scenarios.
    Your task is to verify if the proposed scenario correctly implements the game's strategies and payoff structure
    and effectively masks the game's true nature.
    Provide detailed feedback on what needs to be improved regarding game mechanics implementation.

    Format your response as a JSON object with exactly two fields:
    - "feedback": an array of mechanics-specific feedback points as strings
    - "converged": a boolean, true if mechanics aspects need no changes, false otherwise
    """

    # Create human prompt
    human_prompt = f"""
    Please review the game mechanics implementation in this scenario for a {game_name} game:
    {json.dumps(scenario_draft, indent=2)}

    Evaluate the scenario based on these game mechanics criteria:
    1. For {players[0]}, does the described payoff rank order implied by `payoff_matrix` descriptions match the required order: {game.payoff_matrix.ordered_payoff_leaves[0]}? (Focus on the rank order, not exact values).
    2. For {players[1]}, does the described payoff rank order implied by `payoff_matrix` descriptions match the required order: {game.payoff_matrix.ordered_payoff_leaves[1]}? (Focus on the rank order, not exact values).

    Return a list of specific feedback points for mechanics improvement.
    If the mechanics are implemented correctly and require no changes, state that it has converged.
    Format your response as a JSON with two fields:
    - "feedback": a list of feedback points
    - "converged": true if no changes needed, false otherwise
    """

    # Get LLM response
    llm = get_llm(
        temperature=_global_llm_config["temp_verify"], json_mode=True
    )  # Temperature for verification
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = (
        llm.invoke(messages)
        if isinstance(llm, AzureAIChatCompletionsModel)
        else llm.invoke(messages, response_format={"type": "json_object"})
    )

    # Process response
    result_content = response.content
    result = {}
    if isinstance(result_content, str):
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            result = {
                "feedback": ["Error parsing mechanics verification result"],
                "converged": False,
            }
    elif isinstance(result_content, dict):
        result = result_content
    else:
        result = {
            "feedback": ["Unexpected format for mechanics verification result"],
            "converged": False,
        }

    # Ensure the result has the expected structure
    if (
        not isinstance(result, dict)
        or "feedback" not in result
        or "converged" not in result
    ):
        result = {
            "feedback": ["Invalid mechanics verification result format"],
            "converged": False,
        }

    # Return only the fields we need to update
    return {
        "preference_feedback": result["feedback"],
        "preference_converged": result["converged"],
    }


def verify_pay_off(
    state: ScenarioCreationState, debug_mode: bool = False
) -> Dict[str, Any]:
    """Verify if the payoffs described in the scenario are plausible given the actions."""
    game_name = state["game_name"]
    scenario_draft = state["scenario_draft"]
    players = state["participants"]

    # Handle potential error in scenario draft directly
    if not scenario_draft or "error" in scenario_draft:
        error_msg = (
            scenario_draft.get("error") if scenario_draft else "scenario_draft is None"
        )
        return {
            "payoff_feedback": [
                f"Cannot verify payoff plausibility due to error in scenario draft: {error_msg}"
            ],
            "payoff_converged": False,
        }

    # Get game config for payoff structure
    try:
        game_cfg = get_game_config(game_name)
        game = Game(
            name=game_name,
            scenario_class=game_cfg["scenario_class"],
            decision_class=game_cfg["decision_class"],
            payoff_matrix=game_cfg["payoff_matrix"],
        )
    except Exception as e:
        return {
            "payoff_feedback": [
                f"Error loading game config for payoff validation: {e}"
            ],
            "payoff_converged": False,
        }

    # Check if scenario_draft has the required keys for this specific check
    if (
        "behavior_choices" not in scenario_draft
        or "payoff_matrix_description" not in scenario_draft
        or not isinstance(scenario_draft["payoff_matrix_description"], dict)
    ):
        return {
            "payoff_feedback": [
                "Scenario draft is missing required keys ('behavior_choices', 'payoff_matrix_description') or payoff_matrix is not a dictionary for payoff validation."
            ],
            "payoff_converged": False,
        }

    # Generate validation questions
    validation_questions = []
    # Ensure payoff_leaves exists and is iterable
    if (
        hasattr(game.payoff_matrix, "payoff_leaves")
        and game.payoff_matrix.payoff_leaves
    ):
        try:
            for payoff_leaf in game.payoff_matrix.payoff_leaves:
                # Get behavior descriptions from scenario draft
                leaf_actions = payoff_leaf.actions
                behaviors = []
                for action in leaf_actions:
                    if action in scenario_draft["behavior_choices"]:
                        behaviors.append(scenario_draft["behavior_choices"][action])
                    else:
                        error_msg = f"Required action '{action}' not found in behavior_choices. Available actions: {list(scenario_draft['behavior_choices'].keys())}"
                        if debug_mode:
                            raise KeyError(error_msg)
                        else:
                            return {
                                "payoff_feedback": [f"Format error: {error_msg}"],
                                "payoff_converged": False,
                            }

                # Make behavior description more informative with player names
                # For sequential games, players take turns, so we use modulo to get the current player
                behavior_description_parts = []
                for i, behav in enumerate(behaviors):
                    player_index = i % len(players)
                    behavior_description_parts.append(
                        f"Participant {player_index + 1} ({players[player_index]}) chooses '{behav}'"
                    )
                behavior_description = " and ".join(behavior_description_parts)

                # Construct the key for the payoff matrix dictionary
                # Assumes leaf_actions order matches players order
                outcome_key_parts = []
                for i, action in enumerate(leaf_actions):
                    player_index = i % len(players)
                    outcome_key_parts.append(f"player {player_index + 1}: {action}")
                outcome_key = " , ".join(outcome_key_parts)

                # Get corresponding payoff description from scenario draft
                outcome_list = scenario_draft["payoff_matrix_description"].get(
                    outcome_key
                )

                if outcome_list is None:
                    # Try to find a matching key with different spacing/formatting
                    matching_keys = []
                    for key in scenario_draft["payoff_matrix_description"].keys():
                        # Normalize both keys by removing extra spaces and comparing
                        normalized_outcome_key = " ".join(outcome_key.split())
                        normalized_key = " ".join(key.split())
                        if normalized_outcome_key == normalized_key:
                            matching_keys.append(key)

                    if matching_keys:
                        # Use the first matching key
                        outcome_list = scenario_draft["payoff_matrix_description"][
                            matching_keys[0]
                        ]
                    else:
                        error_msg = f"Payoff outcome not found in scenario draft for key '{outcome_key}'. Available keys: {list(scenario_draft['payoff_matrix_description'].keys())}"
                        if debug_mode:
                            raise ValueError(error_msg)
                        else:
                            return {
                                "payoff_feedback": [f"Format error: {error_msg}"],
                                "payoff_converged": False,
                            }

                if isinstance(outcome_list, list) and len(outcome_list) >= 2:
                    p1_outcome = outcome_list[0]
                    p2_outcome = outcome_list[1]
                elif isinstance(outcome_list, str):
                    p1_outcome = outcome_list
                    p2_outcome = outcome_list
                else:
                    error_msg = f"Invalid payoff outcome format for key '{outcome_key}'. Expected list with 2+ elements or string, got: {type(outcome_list)} with value: {outcome_list}"
                    if debug_mode:
                        raise ValueError(error_msg)
                    else:
                        return {
                            "payoff_feedback": [f"Format error: {error_msg}"],
                            "payoff_converged": False,
                        }

                # Format the question including the outcome using the constant
                question = PAYOFF_VALIDATION_QUESTION_FORMAT.format(
                    behavior_description=behavior_description,
                    p1_outcome=p1_outcome,
                    p2_outcome=p2_outcome,
                )
                validation_questions.append(question)
        except Exception as e:
            if debug_mode:
                raise
            else:
                return {
                    "payoff_feedback": [
                        f"Unexpected error during payoff validation setup: {str(e)}"
                    ],
                    "payoff_converged": False,
                }
    else:
        return {
            "payoff_feedback": [
                "Game config payoff matrix has no 'payoff_leaves' or it is empty."
            ],
            "payoff_converged": False,
        }

    validation_prompt = "\n".join(validation_questions)
    if not validation_prompt:
        return {
            "payoff_feedback": [
                "Could not generate any validation questions for payoffs."
            ],
            "payoff_converged": False,
        }

    # Create system prompt
    system_prompt = f"""
    You are a critical reviewer evaluating the realism of payoff outcomes in a game theory scenario. You will be given a scenario description and a list of questions beginning with "-" about the payoffs.
    Based *only* on the provided scenario description, evaluate if the described outcomes are plausible consequences of the participants' actions.
    Focus solely on the cause-and-effect realism within the story.

    Respond in JSON format with two fields:
    - "feedback": An array of strings, containing your answer for each question OR specific reasons why the payoffs are unrealistic. Please answer each question separately.
    - "converged": A boolean. Set to true ONLY if ALL questions are answered YES (meaning all payoffs are plausible). Set to false otherwise.
    """

    # Create human prompt
    if "description" not in scenario_draft:
        error_msg = "Required 'description' field not found in scenario_draft"
        if debug_mode:
            raise KeyError(error_msg)
        else:
            return {
                "payoff_feedback": [f"Format error: {error_msg}"],
                "payoff_converged": False,
            }

    human_prompt = f"""
    Image you are in the following scenario:
    {scenario_draft["description"]}

    Participants: {players}

    Evaluate the following questions based *only* on the scenario description above. Are the described outcomes plausible consequences of the actions?

    {validation_prompt}

    Return your evaluation as a JSON object with "feedback" and "converged" fields.
    """

    # Get LLM response
    llm = get_llm(
        temperature=_global_llm_config["temp_payoff"], json_mode=True
    )  # Temperature for payoff validation
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    try:
        response = (
            llm.invoke(messages)
            if isinstance(llm, AzureAIChatCompletionsModel)
            else llm.invoke(messages, response_format={"type": "json_object"})
        )
        result_content = response.content
    except Exception as e:
        return {
            "payoff_feedback": [f"Error calling LLM for payoff validation: {e}"],
            "payoff_converged": False,
        }

    # Process response
    result = {}
    if isinstance(result_content, str):
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            result = {
                "feedback": ["Error parsing payoff validation result"],
                "converged": False,
            }
    elif isinstance(result_content, dict):
        result = result_content
    else:
        result = {
            "feedback": ["Unexpected format for payoff validation result"],
            "converged": False,
        }

    # Ensure the result has the expected structure
    if (
        not isinstance(result, dict)
        or "feedback" not in result
        or "converged" not in result
    ):
        result = {
            "feedback": ["Invalid payoff validation result format"],
            "converged": False,
        }

    # Return only the fields we need to update
    return {
        "payoff_feedback": result["feedback"],
        "payoff_converged": result["converged"],
    }


def _allowed_behavior_keys_for_game(game_name: str) -> Optional[List[str]]:
    """
    Best-effort helper to derive the canonical behavior_choices keys from the
    game's BehaviorChoices.example() definition, e.g. EGBehaviorChoices.example().
    """
    try:
        game_cfg = get_game_config(game_name)
        scenario_cls = game_cfg.get("scenario_class")
        if scenario_cls is None:
            return None
        annotations = getattr(scenario_cls, "__annotations__", {})
        behavior_type = annotations.get("behavior_choices")
        example_fn = getattr(behavior_type, "example", None)
        if not callable(example_fn):
            return None
        example = example_fn()
        if not isinstance(example, dict):
            return None
        keys = list(example.keys())
        return keys if keys else None
    except Exception:
        return None


def verify_behavior(state: ScenarioCreationState) -> Dict[str, Any]:
    """
    LLM-based verification that behavior_choices descriptions avoid emotional,
    moral, or coordination-style wording and stay objective.

    This node first enforces that behavior_choices field names match the
    canonical template defined by GameClass.behavior_choices.example().
    Only if the keys match will it call the LLM-based verifier to check
    wording neutrality.

    The LLM prompt construction and result parsing are delegated to the
    behavior_choices_verifier helper module; this function wires in the
    game context and LLM client.
    """
    # Static check: enforce behavior_choices keys match template, if available.
    draft = state.get("scenario_draft") or {}
    behaviors = draft.get("behavior_choices")
    allowed_keys = _allowed_behavior_keys_for_game(state.get("game_name", ""))
    if allowed_keys is not None and isinstance(behaviors, dict):
        current_keys = list(behaviors.keys())
        if set(current_keys) != set(allowed_keys):
            missing = [k for k in allowed_keys if k not in behaviors]
            extra = [k for k in current_keys if k not in allowed_keys]
            parts: List[str] = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"unexpected keys: {extra}")
            parts.append(f"expected behavior_choices keys: {allowed_keys}")
            msg = "Behavior_choices keys do not match template: " + "; ".join(parts)
            return {"behavior_feedback": [msg], "behavior_converged": False}

    try:
        mod = runpy.run_path(str(Path(__file__).with_name("behavior_choices_verifier.py")))
        build_prompt = mod.get("build_behavior_verification_prompt")
        parse_result = mod.get("parse_behavior_verification_result")
        if not callable(build_prompt) or not callable(parse_result):
            raise KeyError("behavior_choices_verifier helpers not available")

        # Build prompts specific to this state's behavior choices.
        system_prompt, human_prompt = build_prompt(state)  # type: ignore[misc]

        llm = get_llm(
            temperature=_global_llm_config["temp_verify"],
            json_mode=True,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        if AzureAIChatCompletionsModel is not None and isinstance(
            llm, AzureAIChatCompletionsModel
        ):
            response = llm.invoke(messages)
        else:
            response = llm.invoke(messages, response_format={"type": "json_object"})

        raw = getattr(response, "content", response)
        parsed = parse_result(raw)  # type: ignore[misc]
        feedback = parsed.get("feedback", [])
        converged = bool(parsed.get("converged"))
    except Exception as e:
        return {
            "behavior_feedback": [f"Behavior verifier unavailable: {e}"],
            "behavior_converged": False,
        }

    if not isinstance(feedback, list):
        feedback = [str(feedback)]
    return {"behavior_feedback": feedback, "behavior_converged": converged}


# Alias to match the requested node/function name in documentation and searches.
def verify_behavior_choices(state: ScenarioCreationState) -> Dict[str, Any]:
    return verify_behavior(state)


def aggregate_verification(state: ScenarioCreationState) -> Dict[str, Any]:
    """
    Aggregation step after parallel verification.

    This node combines results from all verification steps that ran in parallel
    and determines if all verification steps have converged.
    """
    print("Aggregating verification results...")

    # Check all verification results - must explicitly check for True
    required_convergence_flags = [
        "narrative_converged",
        "preference_converged",
        "payoff_converged",
    ]
    all_converged = True

    for flag in required_convergence_flags:
        converged = state.get(flag, False)  # Default to False if missing
        print(f"{flag}: {converged}")
        # Only consider it converged if it's explicitly True
        if converged is not True:
            all_converged = False

    # Safely get scenario name for logging
    scenario_name = "Unknown"
    if state.get("scenario_draft") and isinstance(state["scenario_draft"], dict):
        scenario_name = state["scenario_draft"].get("scenario", "Unknown")

    print(f"All converged for {scenario_name}: {all_converged}")

    # Return only the field we're updating
    return {"all_converged": all_converged}


def should_continue(state: ScenarioCreationState) -> str:
    """Decide whether to continue refining or finalize the scenario."""
    iteration_count = state["iteration_count"]

    # Check if we've converged on ALL aspects or reached max iterations
    all_converged = state.get("all_converged", False)
    max_iterations_reached = iteration_count >= _global_llm_config["max_iterations"]

    if all_converged or max_iterations_reached:
        print(
            f"Iteration {iteration_count}: All converged: {all_converged}, Max iterations reached: {max_iterations_reached}. Moving to finalize."
        )
        return "finalize"
    else:
        print(
            f"Iteration {iteration_count}: All converged: {all_converged}, Max iterations reached: {max_iterations_reached}. Refining."
        )
        return "refine"


def finalize_scenario(state: ScenarioCreationState) -> ScenarioCreationState:
    """Finalize the scenario and save it."""
    all_converged = state.get("all_converged")
    if all_converged is None and state.get("scenario_draft") is not None:
        all_converged = True

    final_scenario = state["scenario_draft"] if all_converged else None

    if final_scenario is not None:
        gradient_opts = state.get("gradient_options") or []
        whose_option = _primary_participant_label(state.get("participants") or [])
        try:
            final_scenario = {
                **final_scenario,
                "gradient_options": list(gradient_opts),
                "whose_option": whose_option,
            }
        except Exception:
            pass

        if gradient_opts:
            try:
                final_scenario = {**final_scenario, "options": gradient_opts}
            except Exception:
                pass
    auto_save_path = state.get("auto_save_path", None)

    file_path = None
    # Save to disk if a path is provided
    if auto_save_path and final_scenario and state["all_converged"]:
        try:
            # Create directory if it doesn't exist
            scenario_dir = Path(auto_save_path)
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = f"{state['game_name']}_{timestamp}"
            file_path = scenario_dir / f"{scenario_name}.json"

            # Save the scenario
            with open(file_path, "w") as f:
                json.dump(final_scenario, f, indent=2)
            print(f"Scenario saved to: {file_path}")

        except Exception as e:
            print(f"Error saving scenario to {auto_save_path}: {e}")
            file_path = None  # Ensure file_path is None if saving failed

    # Update the state with the scenario
    result = {
        **state,
        "final_scenario": final_scenario,
        "scenario_path": str(file_path) if file_path else None,
    }

    return result


# Build the graph
def build_scenario_creation_graph(
    debug_mode: bool = False,
    llm_config: Optional[Dict] = None,
    verification_nodes: Optional[List[str]] = None,
):
    """Build the scenario creation graph with parallel verification steps."""

    # Set global LLM configuration if provided
    if llm_config:
        set_global_llm_config(llm_config)

    if verification_nodes is None:
        # Diplomacy flow: skip preference ordering, verify narrative + behavior wording + payoff
        verification_nodes = ["narrative", "behavior", "pay_off"]

    # Create the graph
    graph = StateGraph(ScenarioCreationState)

    # Add nodes
    graph.add_node("prepare_diplomacy_from_raw", prepare_diplomacy_from_raw)
    graph.add_node("propose_scenario", propose_scenario)
    # Stage 2 nodes for gradient options
    def propose_gradient_options(state: ScenarioCreationState) -> Dict[str, Any]:
        """
        Generate 1-5 neutral, concrete order options for Participant 1 (your_country),
        ordered from restraint to pressure. No negotiation or affect words.
        """
        scenario = state.get("scenario_draft") or {}
        players = state.get("participants") or ["Player 1", "Player 2"]
        raw = state.get("raw_record")
        system_prompt = """
        You generate natural-language order options for a single Diplomacy participant.
        Requirements:
        - 3 to 5 options, ordered from restraint to pressure.
        - Neutral wording; avoid 'cooperate', 'defect', 'aggressive', 'decisive', 'threaten'.
        - Concrete, plausible orders referencing units/regions from context.
        - No negotiation; do not mention deals, alliances, truce, promises, messaging.
        Return a JSON object: {"options": [{"id":1,"text":"..."}, ...]}.
        """
        ctx = {
            "participant": players[0],
            "opponent": players[1] if len(players) > 1 else "Opponent",
            "scenario_description": scenario.get("description", ""),
            "map_record": raw or {},
        }
        human_prompt = json.dumps(ctx, indent=2)
        llm = get_llm(temperature=_global_llm_config["temp_propose"], json_mode=True)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        if AzureAIChatCompletionsModel is not None and isinstance(llm, AzureAIChatCompletionsModel):
            resp = llm.invoke(messages)
        else:
            resp = llm.invoke(messages, response_format={"type": "json_object"})
        content = resp.content
        opts = []
        try:
            obj = json.loads(content) if isinstance(content, str) else content
            raw_opts = obj.get("options") or []
            # normalize ids 1..N, cap at 5
            for i, o in enumerate(raw_opts[:5], start=1):
                text = o.get("text") if isinstance(o, dict) else str(o)
                if text and isinstance(text, str) and len(text) > 8:
                    opts.append({"id": i, "text": text})
        except Exception:
            pass
        return {**state, "gradient_options": opts, "gradient_feedback": [], "gradient_converged": False}

    def verify_gradient(state: ScenarioCreationState) -> Dict[str, Any]:
        """
        Verify monotonic pressure ordering, concreteness, and neutrality of options.
        """
        options = state.get("gradient_options") or []
        scenario = state.get("scenario_draft") or {}
        players = state.get("participants") or ["Player 1", "Player 2"]
        system_prompt = """
        You are verifying a set of Diplomacy order options for a single participant.
        Criteria:
        - Options must be ordered restraint -> pressure (1 is most restrained; last is most pressuring).
        - Orders should be concrete (hold/move/advance/position/contest) and reference plausible units/regions.
        - Neutral wording (no 'cooperate', 'defect', 'aggressive', 'decisive', 'threaten'); no negotiation.
        Return JSON: {"feedback": [...], "converged": true/false}
        """
        ctx = {
            "participant": players[0],
            "scenario_description": scenario.get("description", ""),
            "options": options,
        }
        human_prompt = json.dumps(ctx, indent=2)
        llm = get_llm(temperature=_global_llm_config["temp_verify"], json_mode=True)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        try:
            if AzureAIChatCompletionsModel is not None and isinstance(llm, AzureAIChatCompletionsModel):
                resp = llm.invoke(messages)
            else:
                resp = llm.invoke(messages, response_format={"type": "json_object"})
            content = resp.content
            obj = json.loads(content) if isinstance(content, str) else content
            fb = obj.get("feedback") or []
            conv = bool(obj.get("converged")) and len(options) >= 3
            return {**state, "gradient_feedback": fb, "gradient_converged": conv}
        except Exception as e:
            return {**state, "gradient_feedback": [f"Verifier error: {e}"], "gradient_converged": False}

    def aggregate_gradient(state: ScenarioCreationState) -> Dict[str, Any]:
        return {"all_converged": bool(state.get("gradient_converged"))}

    graph.add_node("propose_gradient_options", propose_gradient_options)
    graph.add_node("verify_gradient", verify_gradient)
    graph.add_node("aggregate_gradient", aggregate_gradient)
    graph.add_node("finalize_scenario", finalize_scenario)

    # Conditionally add verification nodes
    if "narrative" in verification_nodes:
        graph.add_node("verify_narrative", verify_narrative)
    if "behavior" in verification_nodes:
        graph.add_node("verify_behavior", verify_behavior)
    if "pay_off" in verification_nodes:
        graph.add_node(
            "verify_pay_off", lambda state: verify_pay_off(state, debug_mode)
        )

    def aggregate_verification_dynamic(state: ScenarioCreationState) -> Dict[str, Any]:
        """
        Dynamically aggregates results based on the selected verification nodes.
        """
        required_convergence_flags = [f"{v}_converged" for v in verification_nodes]
        all_converged = True
        for flag in required_convergence_flags:
            if state.get(flag) is not True:
                all_converged = False
        return {"all_converged": all_converged}

    graph.add_node("aggregate_verification", aggregate_verification_dynamic)

    # Start -> Propose
    graph.add_edge(START, "prepare_diplomacy_from_raw")
    graph.add_edge("prepare_diplomacy_from_raw", "propose_scenario")

    # Parallel verification steps (fan-out from propose_scenario) if any selected
    for verifier in (verification_nodes or []):
        graph.add_edge("propose_scenario", f"verify_{verifier}")
        graph.add_edge(f"verify_{verifier}", "aggregate_verification")
    # If none selected, route directly to aggregation
    if not (verification_nodes or []):
        graph.add_edge("propose_scenario", "aggregate_verification")

    # Conditional edge from the aggregation node (stage1) always routes to gradient proposals
    graph.add_conditional_edges(
        "aggregate_verification",
        should_continue,
        {"refine": "propose_scenario", "finalize": "propose_gradient_options"},
    )

    # Stage 2: gradient verification loop (only if requested)
    if "gradient" in (verification_nodes or []):
        graph.add_edge("propose_gradient_options", "verify_gradient")
        graph.add_edge("verify_gradient", "aggregate_gradient")
        graph.add_conditional_edges(
            "aggregate_gradient",
            lambda s: "finalize" if s.get("all_converged") else "refine",
            {"refine": "propose_gradient_options", "finalize": "finalize_scenario"},
        )
    else:
        graph.add_edge("propose_gradient_options", "finalize_scenario")

    # Always end after finalizing
    graph.add_edge("finalize_scenario", END)

    # Compile the graph with checkpointer
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def create_scenario(
    game_name: str,
    participants: List[str],
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Create a scenario for the specified game and participants.

    This function builds a LangGraph with parallel verification steps
    to efficiently validate narrative, preference ordering, and payoffs
    simultaneously rather than sequentially.

    Args:
        game_name: The name of the game (e.g., "Prisoners_Dilemma")
        participants: List of participant names
        debug_mode: If True, raise exceptions on format errors; if False, add to feedback

    Returns:
        The created scenario
    """
    # Initialize the graph
    graph = build_scenario_creation_graph(debug_mode)

    # Initialize the state
    initial_state: ScenarioCreationState = {
        "game_name": game_name,
        "participants": participants,
        "scenario_draft": None,
        "gradient_options": None,
        "narrative_feedback": [],
        "behavior_feedback": [],
        "preference_feedback": [],
        "payoff_feedback": [],
        "iteration_count": 0,
        "final_scenario": None,
        "narrative_converged": False,
        "behavior_converged": False,
        "preference_converged": True,
        "payoff_converged": False,
        "all_converged": None,
        "auto_save_path": None,  # Not used in sync version by default
    }

    # Create config with thread_id for the checkpointer
    config = {
        "configurable": {
            "thread_id": f"{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "recursion_limit": 10,  # Set a reasonable recursion limit
        }
    }

    # Run the graph with the config
    final_state = graph.invoke(initial_state, config)

    # Return the final scenario
    return final_state["final_scenario"]


async def a_create_scenario(
    graph: Any,
    game_name: str,
    participants: List[str],
    auto_save_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Create a scenario for the specified game and participants using a pre-compiled graph.

    Args:
        graph: The pre-compiled LangGraph object
        game_name: The name of the game (e.g., "Prisoners_Dilemma")
        participants: List of participant names
        auto_save_path: Optional path to save scenarios
        config: Optional configuration dictionary for the graph

    Returns:
        The created scenario or None if an error occurs during execution
    """
    # Initialize the state
    initial_state: ScenarioCreationState = {
        "game_name": game_name,
        "participants": participants,
        "scenario_draft": None,
        "gradient_options": None,
        "narrative_feedback": [],
        "behavior_feedback": [],
        "preference_feedback": [],
        "payoff_feedback": [],
        "iteration_count": 0,
        "final_scenario": None,
        "narrative_converged": False,
        "behavior_converged": False,
        "preference_converged": True,
        "payoff_converged": False,
        "all_converged": None,
        "auto_save_path": auto_save_path,
    }

    # Create config with thread_id for the checkpointer if not provided
    if config is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        thread_id = f"{game_name}_{timestamp}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": 10,  # Set a reasonable recursion limit
            }
        }

    try:
        # Run the graph with the config
        final_state = await graph.ainvoke(initial_state, config)
        return final_state.get("final_scenario")
    except Exception as e:
        print(
            f"Error invoking graph for thread {config.get('configurable', {}).get('thread_id', 'unknown')}: {e}"
        )
        return None


if __name__ == "__main__":
    # Minimal example: run the Diplomacy graph on one enriched JSONL record
    import time

    # Path to PD-like contests enriched file (adjust if needed)
    sample_path = os.path.expandvars("${USER_HOME}/dipllm/data/pd_like_contests_sample.enriched.jsonl").replace("${USER_HOME}", "/home/jjl7137")
    try:
        lines = Path(sample_path).read_text().splitlines()
        raw_record = json.loads(lines[0]) if lines else {}
    except Exception as e:
        print(f"Failed to load sample record from {sample_path}: {e}")
        raw_record = {}

    # Determine game name without hard-coding: prefer from raw or env, else fallback
    game_name_in = raw_record.get("game_name") or os.getenv("DIPLOMACY_GAME_NAME") or "Escalation_Game"

    # Build graph (uses narrative + payoff first, then generates gradient options)
    graph = build_scenario_creation_graph(
        debug_mode=True,
        llm_config={
            "model": _global_llm_config["model"],
            "temp_propose": _global_llm_config["temp_propose"],
            "temp_verify": _global_llm_config["temp_verify"],
            "temp_payoff": _global_llm_config["temp_payoff"],
            "azure_mode": _global_llm_config["azure_mode"],
            "max_iterations": 6,
        },
        verification_nodes=[],
    )

    # Initial state; participants inferred from raw_record in prepare_diplomacy_from_raw
    initial_state: ScenarioCreationState = {
        "game_name": game_name_in,
        "participants": [],
        "raw_record": raw_record,
        "map_summary": None,
        "scenario_draft": None,
        "gradient_options": None,
        "narrative_feedback": [],
        "behavior_feedback": [],
        "preference_feedback": [],
        "payoff_feedback": [],
        "iteration_count": 0,
        "final_scenario": None,
        "narrative_converged": False,
        "behavior_converged": False,
        "preference_converged": True,
        "payoff_converged": False,
        "all_converged": None,
        "auto_save_path": None,
    }

    config = {
        "configurable": {
            "thread_id": f"Diplomacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "recursion_limit": 10,
        }
    }

    t0 = time.time()
    final_state = graph.invoke(initial_state, config)
    took = time.time() - t0

    scenario = final_state.get("final_scenario")
    print(json.dumps(scenario, indent=2))
    print(f"Participants: {final_state.get('participants')}")
    print(f"Time taken: {took:.2f}s")
