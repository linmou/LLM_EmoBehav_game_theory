# LangGraph Scenario Creation

This module handles the creation of game theory scenarios using LangGraph, a library for building stateful, multi-step AI applications with LLMs.

## Overview

The scenario creation process uses a sequential workflow to generate and refine game theory scenarios:

1. **Propose Scenario**: Generate an initial scenario draft for the specified game
2. **Verify Narrative**: Check if the scenario narrative is realistic and coherent
3. **Verify Preference Order**: Ensure the game mechanics are correctly implemented
4. **Verify Pay Off**: Validate that the outcomes described are plausible 
5. **Aggregate Verification**: Synchronize the verification results
6. **Decision Point**: Continue refining or finalize the scenario

## Implementation Details

The graph was originally designed with parallel verification steps, but due to LangGraph concurrency limitations where multiple nodes can't update the same state keys simultaneously, we've adopted a sequential approach instead.

### Key Components

- **StateGraph**: Manages the state transitions between different stages of scenario creation
- **State Definition**: `ScenarioCreationState` holds input requirements, working data, and output
- **Node Functions**: Each function processes the state and produces new values
- **Edge Logic**: Conditional transitions based on verification results

## Usage

```python
from data_creation.scenario_creation import create_scenario, a_create_scenario
from data_creation.scenario_creation.langgraph_creation import build_scenario_creation_graph

# Synchronous approach (for simple use cases)
scenario = create_scenario(
    game_name="Prisoners_Dilemma",
    participants=["Alice", "Bob"],
    participant_jobs=["lawyer", "lawyer"]
)

# Asynchronous approach (recommended for web applications)
async def create_scenario_async():
    graph = build_scenario_creation_graph()
    
    # Configure with recursion limit to avoid errors in complex scenarios
    config = {
        "configurable": {"thread_id": "unique_id"},
        "recursion_limit": 50  # Important to prevent recursion limit errors
    }
    
    scenario = await a_create_scenario(
        graph=graph,
        game_name="Prisoners_Dilemma", 
        participants=["Alice", "Bob"],
        participant_jobs=["lawyer", "lawyer"],
        auto_save_path="./scenarios",  # Optional path to save generated scenarios
        config=config  # Pass the config with recursion_limit
    )
    return scenario
```

## Troubleshooting

### Recursion Limit Errors

If you encounter this error: 
```
Error invoking graph: Recursion limit of 25 reached without hitting a stop condition
```

Make sure to:
1. Pass a `config` parameter to `a_create_scenario` with a higher `recursion_limit` (e.g., 50)
2. Ensure the `finalize_scenario` function properly sets the `final_scenario` even when verification hasn't fully converged

### Concurrency Errors

If you encounter the error: `Can receive only one value per step. Use an Annotated key to handle multiple values`,
make sure you're using the sequential graph structure that avoids parallel updates to the same state keys.

### API Gateway Issues

For handling API gateway issues (like 502 Bad Gateway), consider implementing retry mechanisms and
adjusting the frequency of API calls to stay within rate limits.

## Dependencies

- langgraph
- langchain
- openai

# Scenario Creation Graph

This module uses LangGraph to create game theory scenarios with parallel verification steps.

## Architecture

The scenario creation process follows these main steps:

1. **propose_scenario**: Creates an initial scenario draft based on game requirements
2. **verification** (parallel steps):
   - **verify_narrative**: Checks the narrative quality and coherence
   - **verify_preference_order**: Validates game-theoretic mechanics
   - **verify_pay_off**: Ensures payoff plausibility
3. **aggregate_verification**: Combines results from parallel verification steps
4. **conditional branching**:
   - If all verifications passed OR max iterations reached: **finalize_scenario**
   - Otherwise: Loop back to **propose_scenario** for refinement

## Parallel Processing

The verification steps run in parallel, which offers several advantages:
- Faster execution as all verifications happen simultaneously
- Independent validation of different aspects of the scenario
- Comprehensive feedback collected in a single iteration

### Implementation Details

- Uses LangGraph's fan-out/fan-in pattern for parallel execution
- State reducers (specifically `operator.add`) combine feedback from parallel nodes
- Node functions only return the specific fields they modify, not the entire state
- The `aggregate_verification` node consolidates verification results

## State Management

The `ScenarioCreationState` class includes reducers for feedback lists to properly handle updates from parallel nodes. Instead of accumulating feedback across iterations using `operator.add`, we now use a custom `replace_reducer` to ensure only the *latest* feedback from the verification steps is kept in the state:

```python
# Custom reducer function
def replace_reducer(existing_value, new_value):
    return new_value

class ScenarioCreationState(TypedDict):
    # ...
    narrative_feedback: Annotated[List[str], replace_reducer]
    preference_feedback: Annotated[List[str], replace_reducer]
    payoff_feedback: Annotated[List[str], replace_reducer]
    # ...
```

### Handling Concurrency Issues

To avoid the error: `Can receive only one value per step. Use an Annotated key to handle multiple values`, we:

1. Modified node functions to return only the specific fields they update
2. Used reducers (like the `replace_reducer` shown above) for any fields that might be updated in parallel
3. Created a dedicated `all_converged` field to track overall convergence

## Troubleshooting

If you encounter concurrency errors, ensure:
- Node functions only return fields they need to modify
- Any fields that might be updated in parallel use reducers
- The `recursion_limit` is set to a reasonable value in the configuration
