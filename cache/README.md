Last updated: 2025-09-28 (commit 09808adf1db2101a19438cc733833f640f05be48)

# SWE-bench Lite Cache Artifacts

This directory holds the offline retrieval results and text datasets required for the SWE-bench Lite experiments under `emotion_experiment_engine`. Regenerate the cache by running the upstream SWE-bench CLI commands below and then pointing the experiments at the outputs preserved here.

## Prerequisites
- Active conda environment: `llm_fresh`
- Python dependencies include `pyserini` (install with `pip install pyserini` inside the environment)
- A modern JDK available to Pyserini. These instructions assume Temurin 21 downloaded to `/data/home/jjl7137/java/jdk-21.0.8+9`.

```bash
# one-time JDK install (already done on this machine)
mkdir -p /data/home/jjl7137/java
wget https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.8%2B9/OpenJDK21U-jdk_x64_linux_hotspot_21.0.8_9.tar.gz -O /data/home/jjl7137/java/OpenJDK21.tar.gz
tar -C /data/home/jjl7137/java -xzf /data/home/jjl7137/java/OpenJDK21.tar.gz
```

## 1. Generate BM25 Retrieval Hits
Run from the SWE-bench repository root so git-based helpers resolve correctly.

```bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate llm_fresh
export JAVA_HOME=/data/home/jjl7137/java/jdk-21.0.8+9
export PATH="$JAVA_HOME/bin:$PATH"
cd /data/home/jjl7137/SWE-bench
python -m swebench.inference.make_datasets.bm25_retrieval \
    --dataset_name_or_path SWE-bench/SWE-bench_Lite \
    --document_encoding_style file_name_and_contents \
    --output_dir /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/cache/retrieval_results \
    --splits test
```

Outputs:
- `cache/retrieval_results/SWE-bench__SWE-bench_Lite/file_name_and_contents.retrieval.jsonl`
- companion index directory (`file_name_and_contents_indexes`) used by Pyserini

A compatibility symlink is expected at `cache/retrieval_results/SWE-bench_SWE-bench_Lite.retrieval.jsonl`. Create or refresh it with:

```bash
ln -sf /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/cache/retrieval_results/SWE-bench__SWE-bench_Lite/file_name_and_contents.retrieval.jsonl \
       /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/cache/retrieval_results/SWE-bench_SWE-bench_Lite.retrieval.jsonl
```

## 2. Materialize the Text Dataset
Continue in the SWE-bench repo with the same environment variables set.

```bash
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path SWE-bench/SWE-bench_Lite \
    --output_dir /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/cache/datasets \
    --retrieval_file /data/home/jjl7137/LLM_EmoBehav_game_theory_flexible_dataset/cache/retrieval_results/SWE-bench__SWE-bench_Lite/file_name_and_contents.retrieval.jsonl \
    --prompt_style style-3 \
    --file_source bm25 \
    --k 20 \
    --max_context_len 32768 \
    --tokenizer_name llama \
    --splits test \
    --validation_ratio 0.0
```

This produces `cache/datasets/SWE-bench__SWE-bench_Lite__style-3__fs-bm25__k-20__mcc-32768-llama`, the dataset consumed by the emotion experiment runner.

## 3. Using the Cached Data
When launching experiments, configure the dataset paths to point at the cache directories above. Any future helper scripts should skip the retrieval step and reuse these prepared artifacts.

## Verification Checklist
- Retrieval JSONL exists and contains 300 entries (one per SWE-bench Lite test instance).
- Text dataset `test` split reports 300 rows when loaded with `datasets.load_from_disk`.
- `JAVA_HOME` is exported to a JDK ≥17 before invoking Pyserini-based scripts.
