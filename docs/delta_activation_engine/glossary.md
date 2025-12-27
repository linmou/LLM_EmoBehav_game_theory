# Delta Activation Engine Glossary
Last updated: 2024-03-19 (working copy)

- **Delta activation**: Vector difference between a steered activation and the baseline activation for the same prompt set.
- **Baseline vector**: Last-layer, last-token representation averaged across probes when steering is disabled.
- **Steered vector**: Same representation computed while injecting an emotion-specific activation direction scaled by intensity.
- **Control layers**: Middle third of decoder layers where RepE directions are injected (e.g., layers 4–7 of a 12-layer model).
- **RepE reader**: Emotion-specific direction map loaded from RepE artifacts; keyed by emotion and layer.
- **Intensity**: Scalar multiplier applied to the activation direction before injection.
- **Prompt probe**: Short instruction template used to elicit a generic response; baseline pipeline uses raw templates, chat pipeline renders them via chat templates.
- **PromptFormat**: Utility from `neuro_manipulation` that applies the model’s chat template to user/system messages.
- **DeltaProbesDataset**: Lightweight dataset that adapts a list of probe strings into `BenchmarkItem` records for prompt rendering.
- **Prompt wrapper**: Callable that assembles system/user text for PromptFormat; here `DeltaProbesPromptWrapper`.
- **Probe hash**: SHA-256 digest of the probe texts used in a run, stored in metadata for reproducibility.
- **Job config**: Parsed YAML describing model path, emotions, intensities, output directory, and backend configs.
- **Chat job config**: Job config extended with prompt configuration (benchmark name, task type, probe source/list, thinking flag).
