from vllm import LLM, SamplingParams

# --- Configuration ---
# Model path on your system
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
# Number of GPUs to use (if you have multiple and want to specify)
# For a 0.5B model, 1 GPU is likely sufficient.
TENSOR_PARALLEL_SIZE = 1
# Set to True if the model requires custom code from Hugging Face Hub
TRUST_REMOTE_CODE = True

# --- Sample Prompts ---
# You can modify these or add your own
prompts = [
    "Hello! Can you tell me a short story?",
    "What is the capital of France?",
    "Write a python function to calculate the factorial of a number.",
    "Translate the following English sentence to Chinese: 'The weather is nice today.'",
]

# --- Sampling Parameters ---
# These parameters control the generation process.
# Adjust them as needed for your use case.
# For more details, see: https://vllm.readthedocs.io/en/latest/dev/sampling_params.html
sampling_params = SamplingParams(
    n=1,  # Number of output sequences to return
    temperature=0.7, # Controls randomness. Higher is more random.
    top_p=0.95, # Nucleus sampling.
    max_tokens=200, # Maximum number of tokens to generate per output.
    # stop=["<|im_end|>", "<|endoftext|>"] # Add model-specific stop tokens if known
                                        # Qwen models often use <|im_end|>
)

def run_vllm_inference():
    """
    Initializes the vLLM engine, generates text for the sample prompts,
    and prints the outputs.
    """
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Using tensor_parallel_size: {TENSOR_PARALLEL_SIZE}")

    try:
        # Initialize the LLM engine
        # For smaller models like 0.5B, you might not need quantization or other advanced settings
        # unless you are resource-constrained.
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=TRUST_REMOTE_CODE,
            # You might need to specify dtype if you encounter issues, e.g., dtype="half" or dtype="bfloat16"
            # dtype="auto" is usually fine.
        )
        print("Model loaded successfully.")

        print("\n--- Starting Inference ---")
        # Generate text for the prompts
        outputs = llm.generate(prompts, sampling_params)
        print("--- Inference Complete ---")

        # Print the outputs
        print("\n--- Generated Outputs ---")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated Text: {generated_text}")
            print("-" * 30)

    except ImportError as e:
        print(f"Error: vLLM library not found. Please ensure it is installed.")
        print(f"Details: {e}")
        print("You can typically install it with: pip install vllm")
    except RuntimeError as e:
        print(f"Error during vLLM initialization or generation: {e}")
        print("This could be due to various reasons such as CUDA issues, model compatibility, or insufficient resources.")
        print("Check the vLLM documentation and GitHub issues for troubleshooting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_vllm_inference()
