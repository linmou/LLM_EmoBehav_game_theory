from vllm import LLM, SamplingParams

# Configuration for the full 128k context using RoPE Scaling
model_id = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-3B-Instruct"

# Define the RoPE scaling configuration as a dictionary
rope_scaling_config = {
    "rope_type": "linear",
    "factor": 4.0,
}

llm = LLM(
    model=model_id,
    rope_theta=1000000,
    max_model_len=131072,       # Set to 128 * 1024
    rope_scaling=rope_scaling_config,
    # Add other parameters like tensor_parallel_size if needed
    tensor_parallel_size=2
)

# Example usage with a potentially very long prompt
long_prompt = ' '.join(["I am good."] * 30000) # Imagine a prompt with > 32k tokens
prompts = [long_prompt]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated: {generated_text!r}")
