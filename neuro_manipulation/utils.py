import base64
import glob
import hashlib
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from emotion_experiment_engine.data_models import VLLMLoadingConfig  # pragma: no cover
else:
    VLLMLoadingConfig = Any  # type: ignore[misc,assignment]

import torch
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]
from neuro_manipulation.threading_env import ensure_mkl_threading_layer
try:
    ensure_mkl_threading_layer()
    from vllm import LLM  # type: ignore
except Exception:
    # Provide a minimal stub to avoid hard dependency during dry-runs or CPU-only envs
    class LLM:  # type: ignore
        pass

# from neuro_manipulation.repe import repe_pipeline_registry
# import pickle
# repe_pipeline_registry()

def _ensure_repo_root_on_pythonpath() -> None:
    """
    vLLM v1 uses multiprocessing workers whose CWD / sys.path may not include this repo.
    Ensure the repo root is on PYTHONPATH so worker_extension_cls can be imported.
    """
    repo_root = str(Path(__file__).resolve().parent.parent)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if repo_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root] + parts) if parts else repo_root


def get_emotion_images(
    image_dir: Union[str, Path], emotions: Optional[List[str]] = None
) -> Dict[str, List[Image.Image]]:
    """
    Load emotion-specific images from organized directory structure.

    Args:
        image_dir: Directory containing emotion subdirectories (e.g., anger/, happiness/, etc.)
        emotions: List of emotions to load. If None, loads standard 6 emotions.

    Returns:
        Dictionary mapping emotion names to lists of PIL Images

    Directory structure expected:
        image_dir/
        ├── anger/
        │   ├── image1.jpg
        │   └── image2.png
        ├── happiness/
        └── sadness/
    """
    if emotions is None:
        emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

    image_dir = Path(image_dir)
    emotion_images: Dict[str, List[Image.Image]] = {}

    # Supported image extensions
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]

    for emotion in emotions:
        emotion_path = image_dir / emotion
        emotion_images[emotion] = []

        if not emotion_path.exists():
            print(f"Warning: Emotion directory not found: {emotion_path}")
            continue

        # Load all images from emotion directory
        for ext in image_extensions:
            image_files = list(emotion_path.glob(ext))
            for image_file in image_files:
                try:
                    pil_image = Image.open(image_file).convert("RGB")
                    emotion_images[emotion].append(pil_image)
                except Exception as e:
                    print(f"Warning: Failed to load image {image_file}: {e}")

        print(f"Loaded {len(emotion_images[emotion])} images for emotion: {emotion}")

    return emotion_images


@dataclass
class AnswerProbabilities:
    # query: str
    # ans: str
    ans_probabilities: List[float]
    ans_ids: List[int]
    input_ids: List[int]
    input_text: str
    emotion: str
    emotion_activation: Dict[int, float]

    @property
    def emotion_activation_mean_last_layer(self):
        return np.mean(list(self.emotion_activation[-1].values()))


def is_huggingface_model_name(model_path):
    """
    Check if the given path is a HuggingFace model name.
    A HuggingFace model name typically has the format 'organization/model_name'.
    """
    return "/" in model_path and not os.path.exists(model_path)


def get_model_config(model_path):
    """
    Get model configuration from either local path or HuggingFace.
    Returns the config dictionary and the actual model path to use.
    """
    try:
        if is_huggingface_model_name(model_path):
            # Download config file from HuggingFace
            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.json",
                cache_dir=None,  # Use default cache
            )
            actual_path = model_path  # Use the HuggingFace model name directly
        else:
            config_path = os.path.join(model_path, "config.json")
            actual_path = model_path

        # Read config file
        with open(config_path, "r") as f:
            config = json.load(f)

        return config, actual_path

    except Exception as e:
        raise


def get_optimal_tensor_parallel_size(model_path):
    """
    Calculate the optimal tensor parallel size based on model architecture and available GPUs.
    Returns the maximum possible tensor parallel size that divides the number of attention heads evenly.
    """
    try:
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()

        if num_gpus == 0:
            return 1

        # Get model config
        config, _ = get_model_config(model_path)

        # Get number of attention heads from config
        # Different models might store this in different keys
        num_heads = None
        possible_keys = ["num_attention_heads", "n_head", "num_heads", "n_heads"]
        for key in possible_keys:
            if key in config:
                num_heads = config[key]
                break

        if num_heads is None:
            return 1

        # Find the largest divisor of num_heads that is <= num_gpus
        optimal_size = 1
        for i in range(1, min(num_heads + 1, num_gpus + 1)):
            if num_heads % i == 0:
                optimal_size = i

        return optimal_size

    except Exception as e:
        return 1


def primary_emotions_concept_dataset(
    data_dir,
    model_name=None,
    tokenizer=None,
    system_prompt=None,
    seed=0,
    multimodal_intent=False,
    enable_thinking=False,
):
    """
    Create dataset of emotion scenarios using proper model-specific prompt formatting.
    Supports both text-only and multimodal (text+image) emotion data generation.

    Args:
        data_dir: Directory containing emotion JSON files (anger.json, happiness.json, etc.)
                 - For text mode: data_dir = "data/stimulus/text/" containing text scenarios in JSONs
                 - For image mode: data_dir = "data/stimulus/image/" containing image paths in JSONs
        model_name: Name of the model to format prompts for
        tokenizer: Optional tokenizer for more accurate prompt formatting
        system_prompt: Optional system prompt, if None no system prompt will be used
        seed: Random seed for shuffling
        multimodal_intent: Whether to use multimodal intent
        enable_thinking: Whether to enable thinking mode

    Returns:
        Dictionary of formatted emotion datasets
        - Text-only mode: strings formatted for the model
        - Multimodal mode: dicts with 'images' and 'text' keys
    """
    import json
    import os
    import random

    import numpy as np
    from transformers import AutoTokenizer

    from neuro_manipulation.prompt_formats import ManualPromptFormat, PromptFormat

    random.seed(seed)

    # Setup prompt format
    if tokenizer is None and model_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer is not None:
        prompt_format = PromptFormat(tokenizer)
    elif model_name is not None:
        # Use ManualPromptFormat as fallback
        format_cls = ManualPromptFormat.get(model_name)
        user_tag = format_cls.user_tag
        assistant_tag = format_cls.assistant_tag
        prompt_format = None  # Will use manual formatting
    else:
        # Default to empty tags if no model info provided
        user_tag = ""
        assistant_tag = ""
        prompt_format = None  # Will use manual formatting

    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

    # Auto-detect data type directly from content in data_dir
    data_status = detect_emotion_data_type(data_dir, emotions)

    # Determine final processing mode
    is_multimodal_data = data_status["is_multimodal_data"]
    should_use_multimodal = multimodal_intent and is_multimodal_data

    if multimodal_intent and not is_multimodal_data:
        print(
            f"⚠️  Multimodal processing requested but data is '{data_status['data_type']}' type - falling back to text-only"
        )
        should_use_multimodal = False
    elif not multimodal_intent and is_multimodal_data:
        print(
            f"✓ Image data detected but multimodal_intent=False - using text-only mode"
        )
        print(f"  Set multimodal_intent=True to enable multimodal processing")

    if should_use_multimodal:
        print(
            f"✓ Multimodal mode enabled: processing {data_status['valid_emotions_count']} emotions with image data"
        )
    else:
        print(
            f"✓ Text-only mode: processing {data_status['valid_emotions_count']} emotions"
        )

    # Load raw data directly from data_dir
    raw_data = {}
    for emotion in emotions:
        emotion_file = os.path.join(data_dir, f"{emotion}.json")
        if os.path.exists(emotion_file):
            with open(emotion_file, "r", encoding="utf-8") as file:
                raw_data[emotion] = list(set(json.load(file)))

    formatted_data = {}
    for emotion in emotions:
        if emotion not in raw_data or len(raw_data[emotion]) == 0:
            continue  # Skip emotions that don't have data files or have empty data

        other_emotions_data = [
            v for k, v in raw_data.items() if k != emotion and len(v) > 0
        ]
        if not other_emotions_data:
            continue  # Skip if no other emotions have data

        c_e, o_e = raw_data[emotion], np.concatenate(other_emotions_data)
        random.shuffle(o_e)

        data = [[c, o] for c, o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c, o] for c, o in zip(c_e, o_e)]).tolist()

        # Helper function to format a single scenario
        def format_scenario(scenario):
            if should_use_multimodal and is_multimodal_data:
                # Scenario is an image path - load the image relative to data_dir
                try:
                    # Image paths in JSON are relative to data_dir
                    full_image_path = Path(data_dir) / scenario

                    if full_image_path.exists():
                        pil_image = Image.open(full_image_path).convert("RGB")

                        # Use same user_message structure as text mode for consistency
                        user_message = f"Consider the emotion of the following scenario:\nScenario: [IMAGE]\nAnswer:"

                        # Format text prompt
                        if prompt_format is not None:
                            formatted_prompt = prompt_format.build(
                                system_prompt, [user_message], [], [pil_image]
                            )
                        else:
                            if system_prompt:
                                formatted_prompt = f"{user_tag} {system_prompt}\n\n{user_message} {assistant_tag} "
                            else:
                                formatted_prompt = (
                                    f"{user_tag} {user_message} {assistant_tag} "
                                )

                        return {"images": [pil_image], "text": formatted_prompt}
                    else:
                        print(f"⚠️  Image not found: {full_image_path}")
                        return None

                except Exception as e:
                    print(f"⚠️  Failed to load image {scenario}: {e}")
                    return None
            else:
                # Text-only processing (scenario is text content)
                user_message = f"Consider the emotion of the following scenario:\nScenario: {scenario}\nAnswer:"

                # Format text prompt
                if prompt_format is not None:
                    formatted_prompt = prompt_format.build(
                        system_prompt,
                        [user_message],
                        [],
                        enable_thinking=enable_thinking,
                    )
                else:
                    if system_prompt:
                        formatted_prompt = f"{user_tag} {system_prompt}\n\n{user_message} {assistant_tag} "
                    else:
                        formatted_prompt = f"{user_tag} {user_message} {assistant_tag} "

                return formatted_prompt

        # Format the data using prompt format
        emotion_test_data = []
        emotion_train_data = []

        for scenario in data_:
            formatted_data_item = format_scenario(scenario)
            if formatted_data_item is not None:
                emotion_test_data.append(formatted_data_item)

        for scenario in data:
            formatted_data_item = format_scenario(scenario)
            if formatted_data_item is not None:
                emotion_train_data.append(formatted_data_item)

        formatted_data[emotion] = {
            "train": {"data": emotion_train_data, "labels": train_labels},
            "test": {
                "data": emotion_test_data,
                "labels": [[1, 0] * len(emotion_test_data)],
            },
        }

    return formatted_data


def test_direction(
    hidden_layers,
    rep_reading_pipeline,
    rep_reader,
    test_data,
    rep_token=-1,
    batch_size: int = 32,
):
    H_tests = rep_reading_pipeline(
        test_data["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        rep_reader=rep_reader,
        batch_size=batch_size,
    )

    results: Dict[int, float] = {}
    rep_readers_means: Dict[int, float] = {}

    for layer in hidden_layers:
        H_test = [H[layer] for H in H_tests]
        rep_readers_means[layer] = float(np.mean(H_test))
        H_test = [H_test[i : i + 2] for i in range(0, len(H_test), 2)]

        sign = rep_reader.direction_signs[layer]

        eval_func = min if sign == -1 else max
        cors = np.mean([eval_func(H) == H[0] for H in H_test])

        results[layer] = float(cors)

    return results, rep_readers_means


def get_rep_reader(
    rep_reading_pipeline,
    train_data,
    test_data,
    hidden_layers,
    rep_token=-1,
    n_difference=1,
    direction_method="pca",
    batch_size: int = 32,
):
    rep_reader = rep_reading_pipeline.get_directions(
        train_data["data"],
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        n_difference=n_difference,
        batch_size=batch_size,
        train_labels=train_data["labels"],
        direction_method=direction_method,
    )

    result, _ = test_direction(
        hidden_layers=hidden_layers,
        rep_reading_pipeline=rep_reading_pipeline,
        rep_reader=rep_reader,
        test_data=test_data,
        rep_token=rep_token,
        batch_size=batch_size,
    )
    print(result)

    return rep_reader, result


def prob_cal_record(
    prob_cal_pipeline,
    dataset,
    emotion,
    rep_token,
    hidden_layers,
    rep_reader,
    save_path="record.pkl",
):
    assert save_path.endswith(".pkl")

    records = []
    for b_out in prob_cal_pipeline(
        dataset,
        rep_token=rep_token,
        hidden_layers=hidden_layers,
        rep_reader=rep_reader,
        batch_size=4,
    ):
        bsz = len(b_out["ans_ids"])
        for bid in range(bsz):
            ans_record = AnswerProbabilities(
                ans_probabilities=b_out["ans_probabilities"][bid],
                ans_ids=b_out["ans_ids"][bid],
                input_ids=b_out["input_ids"][bid],
                emotion=emotion,
                input_text=prob_cal_pipeline.tokenizer.decode(b_out["input_ids"][bid]),
                emotion_activation={
                    layer: b_out[layer][bid] for layer in hidden_layers
                },
            )
            records.append(ans_record)

    # Save all records
    print(f"Saving records to {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Records saved to {save_path}")


def detect_multimodal_model(model_name_or_path):
    """
    Detect if a model is multimodal based on name patterns and config.

    Args:
        model_name_or_path: Model name or path

    Returns:
        bool: True if multimodal model detected
    """
    # Method 1: Name pattern detection
    multimodal_patterns = [
        "VL",
        "Vision",
        "CLIP",
        "BLIP",
        "LLaVA",
        "Flamingo",
        "multimodal",
        "Qwen2.5-VL",
        "GPT-4V",
        "InstructBLIP",
        "gemma-3-4b",
        "gemma-3-12b",
        "gemma-3-27b",
    ]

    model_name_upper = str(model_name_or_path).upper()
    name_indicates_multimodal = any(
        pattern.upper() in model_name_upper for pattern in multimodal_patterns
    )

    if name_indicates_multimodal:
        return True

    # Method 2: Try to load AutoProcessor to check multimodal capability
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        # Check if processor has image processing capabilities
        has_image_processor = hasattr(processor, "image_processor") or hasattr(
            processor, "feature_extractor"
        )
        return has_image_processor
    except Exception:
        pass

    return False


def auto_load_processor(model_name_or_path, vram_optimized=True):
    """
    Automatically load AutoProcessor for multimodal models with VRAM optimization.

    Args:
        model_name_or_path: Model name or path
        vram_optimized: Whether to apply VRAM-saving optimizations

    Returns:
        AutoProcessor or None if not available/not multimodal
    """
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        # Apply VRAM optimizations
        if vram_optimized and hasattr(processor, "image_processor"):
            # EXTREME VRAM REDUCTION: Reduce to tiny images to prevent 9GB allocation
            # Original: 12,845,056 pixels (~3582x3582)
            # Ultra-tiny: 65,536 pixels (~256x256) - minimal for processing
            original_max = getattr(processor.image_processor, "max_pixels", 12845056)
            optimized_max = 65536  # Extremely small to prevent large allocations

            if hasattr(processor.image_processor, "max_pixels"):
                processor.image_processor.max_pixels = optimized_max

            # Force very small image dimensions
            if hasattr(processor.image_processor, "size"):
                if isinstance(processor.image_processor.size, dict):
                    # Force maximum dimensions to be very small
                    processor.image_processor.size = {
                        "shortest_edge": 224,  # Very small
                        "longest_edge": 256,  # Very small maximum
                    }

            # Add image pre-compression if possible
            if hasattr(processor.image_processor, "do_resize"):
                processor.image_processor.do_resize = True
            if hasattr(processor.image_processor, "do_rescale"):
                processor.image_processor.do_rescale = True

        print(f"✓ Auto-loaded AutoProcessor for {model_name_or_path}")
        return processor
    except Exception as e:
        print(f"⚠️  Could not load AutoProcessor for {model_name_or_path}: {e}")
        return None


def is_awq_model(model_name_or_path):
    """
    Detect if a model is AWQ quantized based on naming conventions.
    """
    model_name_upper = str(model_name_or_path).upper()
    awq_indicators = ["-AWQ", "AWQ-", "/AWQ", "_AWQ"]
    return any(indicator in model_name_upper for indicator in awq_indicators)


def load_tokenizer_only(
    model_name_or_path="gpt2",
    user_tag="[INST]",
    assistant_tag="[/INST]",
    expand_vocab=False,
    auto_load_multimodal=True,
):
    """
    Load only tokenizer and processor (if multimodal) without any model loading.
    Lightweight function for validation and dry-run scenarios.

    Args:
        model_name_or_path: Model name or path
        user_tag: User tag for conversation format (for vocab expansion)
        assistant_tag: Assistant tag for conversation format (for vocab expansion)
        expand_vocab: Whether to expand vocabulary with tags
        auto_load_multimodal: Whether to auto-detect and load multimodal processor

    Returns:
        tuple: (tokenizer, processor_or_none)
               - processor_or_none: AutoProcessor for multimodal models, None for text-only
    """
    from transformers import AutoTokenizer  # type: ignore
    from pathlib import Path

    # Load tokenizer (same logic as in load_model_tokenizer)
    tokenizer_kwargs = dict(
        padding_side="left",
        legacy=False,
        token=True,
        trust_remote_code=True,
    )

    tokenizer_source = model_name_or_path
    local_files_only = False
    model_dir = Path(str(model_name_or_path))
    if model_dir.exists() and model_dir.is_dir():
        local_files_only = True
        has_tokenizer_files = any(
            (model_dir / filename).exists()
            for filename in (
                "tokenizer.json",
                "tokenizer.model",
                "vocab.json",
                "merges.txt",
            )
        )
        if not has_tokenizer_files:
            # Some local model directories (notably certain Mamba2 downloads) may
            # contain only weights + config, without tokenizer artifacts.
            # In that case, fall back to a sibling Mamba tokenizer if present.
            for sibling_name in (
                "mamba-790m-hf",
                "mamba-370m-hf",
                "mamba-1.4b-hf",
                "mamba-2.8b-hf",
            ):
                candidate = model_dir.parent / sibling_name
                if any(
                    (candidate / filename).exists()
                    for filename in ("tokenizer.json", "tokenizer.model")
                ):
                    tokenizer_source = str(candidate)
                    print(
                        f"⚠️  No tokenizer files in {model_dir}; using tokenizer from {candidate}"
                    )
                    break

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            local_files_only=local_files_only,
            **tokenizer_kwargs,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=False,
            local_files_only=local_files_only,
            **tokenizer_kwargs,
        )
    try:
        tokenizer.name_or_path = str(model_name_or_path)
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Expand vocabulary if requested (same logic as original)
    if expand_vocab:
        special_tokens_dict = {"additional_special_tokens": [user_tag, assistant_tag]}
        tokenizer.add_special_tokens(special_tokens_dict)

    # Auto-load processor for multimodal models (same logic as original)
    processor = None
    if auto_load_multimodal:
        is_multimodal = detect_multimodal_model(model_name_or_path)
        if is_multimodal:
            processor = auto_load_processor(model_name_or_path)
            if processor:
                print(f"✓ Detected multimodal model: {model_name_or_path}")
            else:
                raise Exception(
                    f"Multimodal model detected but processor loading failed: {model_name_or_path}"
                )
        else:
            print(f"✓ Detected text-only model: {model_name_or_path}")

    return tokenizer, processor


def load_model_only(
    model_name_or_path="gpt2",
    from_vllm=False,
    loading_config: "Optional[VLLMLoadingConfig]" = None,
):
    """
    Load only the model without tokenizer or processor.
    Used when tokenizer is already available from load_tokenizer_only.

    Args:
        model_name_or_path: Model name or path
        from_vllm: Whether to use vLLM for loading
        loading_config: Optional LoadingConfig object for vLLM parameters

    Returns:
        model: Loaded model (vLLM LLM or HuggingFace model)
    """
    model = None
    if from_vllm:
        try:
            _ensure_repo_root_on_pythonpath()
            # vLLM's CLI entrypoints force workers to use `spawn` because the
            # default `fork` is not CUDA-safe after torch has initialized CUDA.
            # We load a HF model on CUDA first (for rep-reader extraction) and
            # then start vLLM, so force the safe default here too.
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            # Use VLLMLoadingConfig.to_vllm_kwargs() method
            if loading_config and hasattr(loading_config, "to_vllm_kwargs"):
                vllm_kwargs = loading_config.to_vllm_kwargs()

                # Auto-detect tensor parallel size if not specified
                if vllm_kwargs.get("tensor_parallel_size") is None:
                    vllm_kwargs["tensor_parallel_size"] = (
                        get_optimal_tensor_parallel_size(model_name_or_path)
                    )
                if is_awq_model(model_name_or_path):
                    vllm_kwargs["quantization"] = "awq"
            else:
                # No loading config - use defaults
                vllm_kwargs = {
                    "model": model_name_or_path,
                    "tensor_parallel_size": get_optimal_tensor_parallel_size(
                        model_name_or_path
                    ),
                    "max_model_len": 32768,
                    "trust_remote_code": True,
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.90,
                    "dtype": "float16",
                    "seed": 42,
                    "disable_custom_all_reduce": False,
                }

            # vLLM selects attention backend via env var; allow forcing through config.
            attention_backend = vllm_kwargs.pop("attention_backend", None)
            if attention_backend:
                os.environ["VLLM_ATTENTION_BACKEND"] = str(attention_backend)

            model = LLM(**vllm_kwargs)

        except Exception as e:
            raise RuntimeError(f"vLLM loading failed: {e}") from e

    if not model:
        # Check if this should be a causal LM model based on its config.
        #
        # NOTE: Some state-spaces Mamba/Mamba2 checkpoints ship a non-Transformers
        # config.json (e.g. `d_model`, `n_layer`) which Transformers will otherwise
        # interpret using default Mamba(2)Config values, causing state_dict shape
        # mismatches. For local paths, translate that config to a proper
        # Transformers config object before loading.
        from transformers import (  # type: ignore
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            MambaConfig,
            Mamba2Config,
            MambaForCausalLM,
            Mamba2ForCausalLM,
        )

        # Determine HF torch dtype, defaulting to float16 when unspecified
        hf_torch_dtype = torch.float16
        if loading_config is not None:
            if isinstance(loading_config, dict):
                dtype_value = loading_config.get("dtype")
            else:
                dtype_value = getattr(loading_config, "dtype", None)

            if isinstance(dtype_value, torch.dtype):
                hf_torch_dtype = dtype_value
            elif isinstance(dtype_value, str):
                dtype_key = dtype_value.lower()
                if dtype_key in {"bfloat16", "bf16"}:
                    hf_torch_dtype = torch.bfloat16
                elif dtype_key in {"float32", "fp32"}:
                    hf_torch_dtype = torch.float32
                elif dtype_key in {"float16", "fp16"}:
                    hf_torch_dtype = torch.float16

        model_dir: Optional[Path] = None
        local_files_only = False
        try:
            p = Path(model_name_or_path)
            if p.exists() and p.is_dir():
                model_dir = p
                local_files_only = True
        except Exception:
            model_dir = None
            local_files_only = False

        hf_common_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if not local_files_only:
            hf_common_kwargs["token"] = True

        def _translate_state_spaces_ssm_config(
            directory: Optional[Path],
        ) -> Optional[Dict[str, Any]]:
            if directory is None:
                return None
            cfg_path = directory / "config.json"
            if not cfg_path.exists():
                return None
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                return None

            ssm_cfg = raw.get("ssm_cfg")
            if not isinstance(ssm_cfg, dict):
                return None
            layer = ssm_cfg.get("layer")
            if layer not in {"Mamba", "Mamba2"}:
                return None

            d_model = raw.get("d_model")
            n_layer = raw.get("n_layer")
            vocab_size = raw.get("vocab_size")
            if not isinstance(d_model, int) or not isinstance(n_layer, int):
                return None

            def _load_checkpoint_shapes() -> Dict[str, Any]:
                ckpt_path = directory / "pytorch_model.bin"
                if not ckpt_path.exists():
                    return {}
                try:
                    state_dict = torch.load(str(ckpt_path), map_location="cpu")
                except Exception:
                    return {}
                if (
                    isinstance(state_dict, dict)
                    and "state_dict" in state_dict
                    and isinstance(state_dict["state_dict"], dict)
                ):
                    state_dict = state_dict["state_dict"]
                if not isinstance(state_dict, dict):
                    return {}
                return state_dict

            common = {
                "hidden_size": d_model,
                "num_hidden_layers": n_layer,
            }
            if isinstance(vocab_size, int):
                common["vocab_size"] = vocab_size

            state_dict = _load_checkpoint_shapes()

            if layer == "Mamba2":
                # Prefer checkpoint-derived padded vocab size when available.
                emb = state_dict.get("backbone.embedding.weight")
                if hasattr(emb, "shape") and len(getattr(emb, "shape", ())) == 2:
                    common["vocab_size"] = int(emb.shape[0])

                # Transformers Mamba2Config requires: hidden_size * expand == num_heads * head_dim.
                # state-spaces checkpoints store per-head tensors with length = num_heads (e.g. dt_bias).
                dt = state_dict.get("backbone.layers.0.mixer.dt_bias")
                if hasattr(dt, "numel"):
                    num_heads = int(dt.numel())
                    expand = 2
                    common["num_heads"] = num_heads
                    common["expand"] = expand
                    head_dim = (d_model * expand) // max(num_heads, 1)
                    common["head_dim"] = int(head_dim)

                conv_w = state_dict.get("backbone.layers.0.mixer.conv1d.weight")
                if (
                    hasattr(conv_w, "shape")
                    and len(getattr(conv_w, "shape", ())) == 3
                    and hasattr(conv_w.shape[0], "__int__")
                ):
                    conv_dim = int(conv_w.shape[0])
                    intermediate_size = int(common.get("expand", 2) * d_model)
                    if conv_dim >= intermediate_size and (conv_dim - intermediate_size) % 2 == 0:
                        group_state_prod = (conv_dim - intermediate_size) // 2
                        # Prefer the common state_size=128 when it fits; otherwise keep product intact.
                        if group_state_prod % 128 == 0:
                            common["state_size"] = 128
                            common["n_groups"] = int(group_state_prod // 128)
                        else:
                            common["state_size"] = int(group_state_prod)
                            common["n_groups"] = 1

                if isinstance(raw.get("rms_norm"), bool):
                    common["rms_norm"] = raw["rms_norm"]
                if isinstance(raw.get("residual_in_fp32"), bool):
                    common["residual_in_fp32"] = raw["residual_in_fp32"]
                if isinstance(raw.get("tie_embeddings"), bool):
                    common["tie_word_embeddings"] = raw["tie_embeddings"]
                return {"config": Mamba2Config(**common), "state_dict": state_dict}

            # layer == "Mamba"
            if isinstance(raw.get("residual_in_fp32"), bool):
                common["residual_in_fp32"] = raw["residual_in_fp32"]
            if isinstance(raw.get("tie_embeddings"), bool):
                common["tie_word_embeddings"] = raw["tie_embeddings"]
            return {"config": MambaConfig(**common), "state_dict": state_dict}

        translated = _translate_state_spaces_ssm_config(model_dir)
        config = translated["config"] if translated else None
        state_dict = translated.get("state_dict") if translated else None

        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, **hf_common_kwargs)
            state_dict = None

        if isinstance(state_dict, dict):
            # state-spaces checkpoints use `backbone.embedding.weight` while Transformers uses `backbone.embeddings.weight`.
            if "backbone.embedding.weight" in state_dict and "backbone.embeddings.weight" not in state_dict:
                state_dict = dict(state_dict)
                state_dict["backbone.embeddings.weight"] = state_dict.pop("backbone.embedding.weight")

        def _load_with_config(model_cls):  # type: ignore[no-untyped-def]
            if state_dict is not None:
                # Transformers disallows passing `state_dict` together with a model path.
                return model_cls.from_pretrained(
                    None,
                    config=config,
                    state_dict=state_dict,
                    torch_dtype=hf_torch_dtype,
                    device_map="auto",
                ).eval()

            return model_cls.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=hf_torch_dtype,
                device_map="auto",
                **hf_common_kwargs,
            ).eval()

        model_type = getattr(config, "model_type", None)
        if model_type in {"mamba", "mamba2"}:
            if model_type == "mamba2":
                model = _load_with_config(Mamba2ForCausalLM)
            else:
                model = _load_with_config(MambaForCausalLM)
        elif hasattr(config, "architectures") and getattr(config, "architectures"):
            if any(
                isinstance(arch, str) and "ForCausalLM" in arch
                for arch in config.architectures
            ):
                model = _load_with_config(AutoModelForCausalLM)
            else:
                model = _load_with_config(AutoModel)
        else:
            model = _load_with_config(AutoModel)

    return model


def load_model_tokenizer(
    model_name_or_path="gpt2",
    user_tag="[INST]",
    assistant_tag="[/INST]",
    expand_vocab=False,
    from_vllm=False,
    auto_load_multimodal=True,
    enable_multi_gpu=True,
    loading_config: "Optional[VLLMLoadingConfig]" = None,
):
    """
    Enhanced model loading with automatic multimodal detection and processor loading.
    BACKWARD COMPATIBLE: Maintains existing API and behavior.

    Args:
        model_name_or_path: Model name or path
        user_tag: User tag for conversation format
        assistant_tag: Assistant tag for conversation format
        expand_vocab: Whether to expand vocabulary with tags
        from_vllm: Whether to use vLLM for loading
        auto_load_multimodal: Whether to auto-detect and load multimodal processor
        enable_multi_gpu: Legacy parameter (preserved for compatibility)
        loading_config: Optional LoadingConfig object or dict with vLLM loading parameters

    Returns:
        tuple: (model, tokenizer, processor_or_none)
               - processor_or_none: AutoProcessor for multimodal models, None for text-only
    """
    # Load tokenizer and processor first (lightweight)
    tokenizer, processor = load_tokenizer_only(
        model_name_or_path=model_name_or_path,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        expand_vocab=expand_vocab,
        auto_load_multimodal=auto_load_multimodal,
    )

    # Load model (heavy GPU operation)
    model = load_model_only(
        model_name_or_path=model_name_or_path,
        from_vllm=from_vllm,
        loading_config=loading_config,
    )

    return model, tokenizer, processor


def detect_emotion_data_type(data_dir, emotions=None):
    """
    Detect if emotion data contains text scenarios or image paths.

    Args:
        data_dir: Directory containing emotion JSON files
        emotions: List of emotions to check. If None, uses standard 6 emotions.

    Returns:
        dict: {
            'data_type': 'text' | 'image' | 'mixed' | 'none',
            'available_emotions': list of emotions with valid data,
            'total_samples': dict mapping emotion to sample count,
            'is_multimodal_data': bool
        }
    """
    if emotions is None:
        emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

    assert data_dir and Path(data_dir).exists(), f"The {data_dir} does not exist."

    available_emotions = []
    total_samples = {}
    data_types_found = set()

    for emotion in emotions:
        emotion_file = Path(data_dir) / f"{emotion}.json"
        if not emotion_file.exists():
            continue

        try:
            with open(emotion_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                continue

            # Sample first few items to determine data type
            sample_items = data[: min(3, len(data))]

            # Check if items look like file paths (contain common image extensions or path separators)
            image_indicators = [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
                ".gif",
                "/",
                "\\",
            ]
            looks_like_paths = 0

            for item in sample_items:
                if isinstance(item, str):
                    item_lower = item.lower()
                    if any(indicator in item_lower for indicator in image_indicators):
                        looks_like_paths += 1

            # Determine data type for this emotion
            if looks_like_paths >= len(sample_items) * 0.7:  # 70% threshold
                emotion_data_type = "image"
            else:
                emotion_data_type = "text"

            data_types_found.add(emotion_data_type)
            available_emotions.append(emotion)
            total_samples[emotion] = len(data)

        except Exception as e:
            print(f"Warning: Failed to process {emotion_file}: {e}")
            continue

    # Determine overall data type
    if len(data_types_found) == 0:
        overall_data_type = "none"
    elif len(data_types_found) == 1:
        overall_data_type = list(data_types_found)[0]
    else:
        overall_data_type = "mixed"  # Some emotions have text, others have image paths

    return {
        "data_type": overall_data_type,
        "available_emotions": available_emotions,
        "total_samples": total_samples,
        "is_multimodal_data": overall_data_type == "image",
        "valid_emotions_count": len(available_emotions),
    }


def validate_multimodal_experiment_feasibility(config):
    """
    Validate if multimodal experiment is feasible based on model, data, and configuration.

    Args:
        config: Experiment configuration dictionary

    Returns:
        dict: {
            'feasible': bool,
            'mode': 'multimodal' | 'text_only' | 'impossible',
            'reasons': list of reason strings,
            'data_status': dict from detect_emotion_data_type,
            'model_is_multimodal': bool
        }
    """
    model_name = config.get("model_name_or_path", "")
    data_dir = config.get("data_dir", "")
    multimodal_intent = config.get("multimodal_intent", False)

    # Check model capabilities
    model_is_multimodal = detect_multimodal_model(model_name)

    # Check data availability
    data_status = detect_emotion_data_type(data_dir)

    reasons = []

    # Determine feasible mode
    if multimodal_intent:
        # User explicitly wants multimodal experiment
        if not model_is_multimodal:
            return {
                "feasible": False,
                "mode": "impossible",
                "reasons": [
                    f"Multimodal experiment requested but model '{model_name}' is text-only"
                ],
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }

        if data_status["data_type"] != "image":
            return {
                "feasible": False,
                "mode": "impossible",
                "reasons": [
                    f"Multimodal experiment requested but data is '{data_status['data_type']}' type"
                ],
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }

        if data_status["valid_emotions_count"] < 2:
            return {
                "feasible": False,
                "mode": "impossible",
                "reasons": [
                    f"Multimodal experiment requested but only {data_status['valid_emotions_count']} emotions have valid image data (need ≥2)"
                ],
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }

        # All conditions met for multimodal
        reasons.append(f"Multimodal experiment explicitly requested and feasible")
        reasons.append(f"Model '{model_name}' supports multimodal processing")
        reasons.append(
            f"Found {data_status['valid_emotions_count']} emotions with image data"
        )

        return {
            "feasible": True,
            "mode": "multimodal",
            "reasons": reasons,
            "data_status": data_status,
            "model_is_multimodal": model_is_multimodal,
        }

    else:
        # No explicit multimodal intent - determine best mode automatically
        if (
            model_is_multimodal
            and data_status["data_type"] == "image"
            and data_status["valid_emotions_count"] >= 2
        ):
            # All conditions met for multimodal - suggest it
            reasons.append("Auto-detected: Multimodal model + image data available")
            reasons.append(f"Model '{model_name}' supports multimodal processing")
            reasons.append(
                f"Found {data_status['valid_emotions_count']} emotions with image data"
            )
            reasons.append(
                "Consider setting 'multimodal_intent: true' for multimodal experiment"
            )

            return {
                "feasible": True,
                "mode": "text_only",  # Default to text-only unless explicitly requested
                "reasons": reasons,
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }

        elif (
            data_status["data_type"] in ["text", "none"]
            or data_status["valid_emotions_count"] >= 2
        ):
            # Text-only mode is feasible
            reasons.append("Text-only experiment feasible")
            if data_status["data_type"] == "text":
                reasons.append(
                    f"Found {data_status['valid_emotions_count']} emotions with text scenarios"
                )

            return {
                "feasible": True,
                "mode": "text_only",
                "reasons": reasons,
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }

        else:
            # Not enough data for any experiment
            return {
                "feasible": False,
                "mode": "impossible",
                "reasons": [
                    f"Insufficient emotion data: {data_status['valid_emotions_count']} emotions found (need ≥2)"
                ],
                "data_status": data_status,
                "model_is_multimodal": model_is_multimodal,
            }


def all_emotion_rep_reader(
    data,
    emotions,
    rep_reading_pipeline,
    hidden_layers,
    rep_token,
    n_difference,
    direction_method,
    save_path="exp_records/emotion_rep_reader.pkl",
    read_args=None,
    batch_size: int = 32,
):

    # args = {
    #     'rep_token': rep_token,
    #     'hidden_layers': hidden_layers,
    #     'n_difference': n_difference,
    #     'direction_method': direction_method,
    # }

    # if save_path is not None and os.path.exists(save_path) and not rebuild:
    #     with open(save_path, 'rb') as f:
    #         emotion_rep_readers = pickle.load(f)
    #         if 'args' in emotion_rep_readers:
    #             if emotion_rep_readers['args'] == args:
    #                 return emotion_rep_readers

    emotion_rep_readers: Dict[str, Any] = {"layer_acc": {}}
    for emotion in tqdm(emotions):
        train_data = data[emotion]["train"]
        test_data = data[emotion]["test"]
        rep_reader, layer_acc = get_rep_reader(
            rep_reading_pipeline=rep_reading_pipeline,
            train_data=train_data,
            test_data=test_data,
            hidden_layers=hidden_layers,
            rep_token=rep_token,
            n_difference=n_difference,
            direction_method=direction_method,
            batch_size=batch_size,
        )

        emotion_rep_readers[emotion] = rep_reader
        emotion_rep_readers["layer_acc"][emotion] = layer_acc

    emotion_rep_readers["args"] = read_args
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(emotion_rep_readers, f)

    return emotion_rep_readers


def dict_to_unique_code(dictionary):
    # Step 1: Serialize the dictionary to a JSON string
    serialized_dict = json.dumps(dictionary, sort_keys=True)

    # Step 2: Hash the serialized string using SHA-256
    hash_object = hashlib.sha256(serialized_dict.encode())
    hash_hex = hash_object.hexdigest()

    # Step 3: Convert the hash to a base64 string (optional)
    hash_bytes = hash_hex.encode("utf-8")
    unique_code = base64.urlsafe_b64encode(hash_bytes).rstrip(b"=").decode("utf-8")

    return unique_code


def oai_response(
    prompt,
    client,
    model="gpt-4o",
    response_format=None,
):
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            # {'role': 'system', 'content': 'You are an avereage American.'},
            {"role": "user", "content": prompt}
        ],
        response_format=response_format,
        seed=42,
    )
    return response.choices[0].message.content


def main():
    from transformers import AutoTokenizer  # type: ignore

    emotions = [
        "happiness",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    ]
    # emotions = ["stress"]
    data_dir = "/home/jjl7137/representation-engineering/data/emotions"
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    user_tag = "[INST]"
    assistant_tag = "[/INST]"

    # model, tokenizer = load_model_tokenizer(model_name_or_path,user_tag=user_tag,
    #                                         assistant_tag=assistant_tag,
    #                                         expand_vocab=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        padding_side="left",
        legacy=False,
        token=True,
        trust_remote_code=True,
    )
    data = primary_emotions_concept_dataset(
        data_dir, model_name=model_name_or_path, tokenizer=tokenizer
    )

    # rep_token = -1
    # hidden_layers = list(range(-1, -model.config.num_hidden_layers+8, -1))
    # n_difference = 1
    # direction_method = 'pca'

    # rep_reading_pipeline = pipeline( "rep-reading", model=model, tokenizer=tokenizer)
    # prob_cal_pipeline = pipeline( "rep-reading&prob-calc", model=model, tokenizer=tokenizer, user_tag=user_tag, assistant_tag=assistant_tag)

    # emotion_rep_readers = all_emotion_rep_reader(data, emotions, rep_reading_pipeline, hidden_layers, rep_token, n_difference, direction_method)

    # for pid, emotional_prompt in tqdm(enumerate(Negative_SET)):
    #     dataset = EvalDatasets('MATH', prompt_modify_func=lambda question, answer: f' {emotional_prompt} {user_tag} {question} {assistant_tag}: Answer: {answer}')

    #     for emotion in emotions:
    #         rep_reader = emotion_rep_readers[emotion]
    #         prob_cal_record(prob_cal_pipeline=prob_cal_pipeline,
    #                         dataset=dataset,
    #                         emotion=emotion,
    #                         rep_token=None,
    #                         hidden_layers=hidden_layers,
    #                         rep_reader=rep_reader,
    #                         save_path=f'exp_records/MATH/prompt_{pid}_{emotion}_record.pkl')


if __name__ == "__main__":
    main()
