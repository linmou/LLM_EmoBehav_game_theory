# SC2 dataset generator

Generates StarCraft scenarios using Gemini 2.5 Flash, seeded by `few_shot_examples.json`.

## Usage

```bash
export GEMINI_API_KEY="..."
python generate_sc2_scenarios_with_gemini.py --n 50 --out generated_scenarios.json
```

Parallel batches:

```bash
export GEMINI_API_KEY="..."
python generate_sc2_scenarios_with_gemini.py --n 200 --batch_size 10 --concurrency 8 --out generated_scenarios.json
```

Disable progress bar:

```bash
python generate_sc2_scenarios_with_gemini.py --no_progress --n 200 --out generated_scenarios.json
```
