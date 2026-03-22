import pickle

from transformers import pipeline
from vllm import LLM

from constants import Emotions
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.repe.pipelines import get_pipeline
from neuro_manipulation.utils import (
    all_emotion_rep_reader,
    detect_emotion_data_type,
    dict_to_unique_code,
    load_model_tokenizer,
    load_tokenizer_only,
    load_model_only,
    primary_emotions_concept_dataset,
)


def setup_model_and_tokenizer(config, from_vllm=False):
    """
    Setup model and tokenizer with merged configuration.

    Args:
        config: Dict with model configuration (can contain model_name_or_path)
        from_vllm: Whether to load using vLLM
        loading_config: Optional LoadingConfig object with loading parameters

    Returns:
        tuple: (model, tokenizer, prompt_format, processor)
    """
    try:
        model_path = config.model_path
    except:
        model_path = config.get("model_name_or_path", config.get("model_path"))

    model, tokenizer, processor = load_model_tokenizer(
        model_path,
        expand_vocab=False,
        from_vllm=from_vllm,
        loading_config=config,
    )

    prompt_format = PromptFormat(tokenizer)

    return model, tokenizer, prompt_format, processor


def load_emotion_readers(
    config, model, tokenizer, hidden_layers, processor=None, enable_thinking=False
):
    """
    Load emotion readers with complete auto-detection for multimodal processing.

    Args:
        config: Configuration dictionary
        model: The model to use
        tokenizer: Model tokenizer
        hidden_layers: Hidden layers for emotion vector extraction
        processor: Optional processor (auto-loaded if None and multimodal model detected)

    Returns:
        Dictionary of emotion readers
    """
    from neuro_manipulation.utils import validate_multimodal_experiment_feasibility

    # Validate experiment feasibility and get recommended mode
    feasibility = validate_multimodal_experiment_feasibility(config)

    if not feasibility["feasible"]:
        print("❌ Experiment not feasible:")
        for reason in feasibility["reasons"]:
            print(f"   - {reason}")
        raise ValueError(
            "Cannot proceed with emotion reader loading - check configuration"
        )

    # Determine final processing mode
    experiment_mode = feasibility["mode"]
    multimodal_intent = config.get("multimodal_intent", False)
    emotion_data_seed = int(config.get("emotion_data_seed", 0))

    # Auto-load processor if needed and not provided
    if experiment_mode == "multimodal" and processor is None:
        from neuro_manipulation.utils import auto_load_processor

        processor = auto_load_processor(config["model_name_or_path"])
        if processor is None:
            model_lower = str(config.get("model_name_or_path", "")).lower()
            if "internvl" in model_lower:
                # InternVL AutoProcessor is often tokenizer-only, but we can still build
                # `pixel_values` via AutoImageProcessor in the InternVL adapter.
                # Also provide an image processor to transformers Pipeline so batching
                # can pad `pixel_values` without crashing.
                from transformers import AutoImageProcessor

                processor = AutoImageProcessor.from_pretrained(
                    config["model_name_or_path"], trust_remote_code=True
                )
                print(
                    "⚠️  InternVL detected: AutoProcessor unavailable; proceeding in multimodal mode "
                    "using AutoImageProcessor-based adapter."
                )
            else:
                raise ValueError(
                    "Multimodal mode requested but AutoProcessor is unavailable for this model."
                )

    print(f"✓ Experiment mode: {experiment_mode}")
    for reason in feasibility["reasons"]:
        print(f"  - {reason}")

    # Build args dict including multimodal parameters
    args = {
        "emotions": config.get("emotions", Emotions.get_emotions()),
        "data_dir": config["data_dir"],
        "model_name_or_path": config["model_name_or_path"],
        "rep_token": config["rep_token"],
        "hidden_layers": hidden_layers,
        "n_difference": config["n_difference"],
        "direction_method": config["direction_method"],
        "experiment_mode": experiment_mode,
        "multimodal_intent": multimodal_intent,
        "emotion_data_seed": emotion_data_seed,
    }

    arg_codes = dict_to_unique_code(args)
    cache_filename = f"neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl"

    # Try to load cached emotion readers
    try:
        if not config.get("rebuild", False):
            emotion_rep_readers = pickle.load(open(cache_filename, "rb"))
            cached_emotions = set(
                key for key in emotion_rep_readers.keys() if isinstance(key, str)
            )
            requested_emotions = set(args["emotions"])
            has_requested_readers = requested_emotions.issubset(cached_emotions)
            if emotion_rep_readers.get("args") == args and has_requested_readers:
                print("✓ Loaded cached emotion readers")
                return emotion_rep_readers
    except:
        pass

    # Generate emotion dataset with auto-detection
    requested_emotions = list(config.get("emotions", Emotions.get_emotions()))
    dataset_emotions = requested_emotions
    if len(requested_emotions) == 1:
        data_status = detect_emotion_data_type(config["data_dir"])
        available_emotions = data_status.get("available_emotions", [])
        if requested_emotions[0] in available_emotions and len(available_emotions) >= 2:
            dataset_emotions = available_emotions

    data = primary_emotions_concept_dataset(
        config["data_dir"],
        model_name=config["model_name_or_path"],
        tokenizer=tokenizer,
        seed=emotion_data_seed,
        enable_thinking=enable_thinking,
        multimodal_intent=(experiment_mode == "multimodal"),
        emotions=dataset_emotions,
    )

    # Create appropriate pipeline based on experiment mode
    if experiment_mode == "multimodal":
        print("✓ Creating multimodal rep-reading pipeline")
        rep_reading_pipeline = pipeline(
            "multimodal-rep-reading",
            model=model,
            tokenizer=tokenizer,
            image_processor=processor,  # Use AutoProcessor for multimodal
        )
    else:
        print("✓ Creating text-only rep-reading pipeline")
        rep_reading_pipeline = pipeline(
            "rep-reading",
            model=model,
            tokenizer=tokenizer,
            image_processor=processor,
        )

    return all_emotion_rep_reader(
        data,
        requested_emotions,
        rep_reading_pipeline,
        hidden_layers,
        config["rep_token"],
        config["n_difference"],
        config["direction_method"],
        read_args=args,
        save_path=cache_filename,
    )
