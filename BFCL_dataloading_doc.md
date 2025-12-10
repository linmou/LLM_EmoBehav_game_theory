  BFCL Evaluation: Quick Guide
  The root directory is /home/jjl7137/gorilla/berkeley-function-call-leaderboard, it should be attached before all path in this file.

  - Offline evaluation reads saved model outputs and local JSON datasets. It does not call model APIs during scoring.
  - Two phases:
      - Generation: bfcl_eval/_llm_response_generation.py queries models (API or local) and writes result/<model>/.../BFCL_v*_<_category_>_result.json.
      - Evaluation: bfcl_eval/eval_checker/eval_runner.py loads prompts/ground-truth from bfcl_eval/data and scores the results offline, writing score/<model>/.../BFCL_v*_<_category_>_score.json plus CSV summaries.

  Key Code Paths

  - Runner: bfcl_eval/eval_checker/eval_runner.py
      - runner(...) walks result/ and calls evaluate_task(...).
      - evaluate_task(...) dispatches by category to:
      - Single-turn AST: `ast_file_runner(...)`
      - Multi-turn: `multi_turn_runner(...)`
      - Agentic: `agentic_runner(...)`
      - Relevance/Irrelevance and Format Sensitivity use dedicated paths.
  - Dataset loader (file-based “dataloader”): bfcl_eval/utils.py
      - load_dataset_entry(category): loads prompts from bfcl_eval/data.
      - load_ground_truth_entry(category): loads expected values from bfcl_eval/data/possible_answer.
  - Checker (single-turn): bfcl_eval/eval_checker/ast_eval/ast_checker.py
      - simple_function_checker(...) enforces function name, required/optional params, types, and value matching.
      - Strings normalized via standardize_string to be robust to case/spacing/punctuation.
      - Optional params: allowed omission is encoded by "" in ground truth.

  Input/Output Shape (live_simple)

  - Input (from bfcl_eval/data/BFCL_v4_live_simple.json):
      - id: unique test id.
      - question: list of one turn with {role, content}.
      - function: one function spec with name and JSON-schema params.
  - Expected model output:
      - A list with a single dict: [{ "<func_name>": { "<param>": <value>, ... } }].
  - Ground truth (from bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json):
      - {"id": "...", "ground_truth": [{ "<func_name>": { "<param>": [allowed_values...] }}]}
      - Optional parameter omission is allowed if the list contains "".

  Example

  - ID: live_simple_4-3-0
      - Input: ask weather in “Tel Aviv” with Fahrenheit.
      - Function: get_current_weather(location: string, unit: enum["celsius","fahrenheit"] default "fahrenheit")
      - Ground truth: {"location": ["Tel Aviv, Israel"], "unit": ["fahrenheit", ""]}
      - Accepted outputs:
      - `[{"get_current_weather": {"location": "Tel Aviv, Israel", "unit": "fahrenheit"}}]`
      - `[{"get_current_weather": {"location": "Tel Aviv, Israel"}}]`  (omit unit; default allowed by `""`)

  What live_simple Tests

  - Not tool selection (only one function). It tests:
      - Schema conformance: exactly one function call, correct JSON shape.
      - Argument extraction: correct values, enums/booleans/ints/types match schema.
      - Defaults/optionals: required present; optionals omitted only if "" present.
      - Canonicalization: string normalization (e.g., case/format differences OK).

  Other Categories (Context)

  - multiple/parallel: multiple functions and ordering.
  - irrelevance: ensuring no tool call when none is appropriate.
  - multi_turn: multi-step reasoning + intermediate tool calls.
  - format_sensitivity: robustness to prompt/format variations.
  - agentic (web_search, memory): multi-step agent workflows.

  Running It

  - Generate: bfcl generate --model <model_name> --test-category <category>
  - Evaluate: bfcl evaluate --model <model_name> --test-category <category>
  - Results live under:
      - Results: result/<model>/.../BFCL_v*_<_category_>_result.json
      - Scores: score/<model>/.../BFCL_v*_<_category_>_score.json, plus CSVs in score/.

  Takeaway

  - “Expected outputs” aren’t stored as separate canonical renderings. Flexibility (e.g., defaultable params) is encoded in ground truth value lists (including "") and enforced by simple_function_checker and helpers.

  Example of formatted data point:

   Here’s a fully assembled prompt for entry live_simple_4-3-0 (system + tools + user), exactly as a prompting-style chat model would receive it:

  - role: system
  content:
  You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used,
  point it out. If the given question lacks the parameters required by the function, also point it out.
  You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used,
  point it out. If the given question lacks the parameters required by the function, also point it out.

  You should only return the function calls in your response.

  If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.

  At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no
  more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

  Here is a list of functions in json format that you can invoke.
  [
        {
            "name": "get_current_weather",
            "description": "Retrieves the current weather conditions for a specified city and state. If using state, then use short form like CA.",
            "parameters": {
                "type": "dict",
                "required": [
                    "location"
                ],
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which to get the weather, in the format of 'City, State (abbr)', such as 'San Francisco, CA' if State for the city exists. 'City, Country' if State for the city doesn't exist."
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit of temperature for the weather report.",
                        "enum": [
                            "celsius",
                            "fahrenheit"
                        ],
                        "default": "fahrenheit"
                    }
                }
            }
        }
  ]

  - role: user
  content: What are the current weather conditions in Tel Aviv, and could you provide that in Fahrenheit, please?