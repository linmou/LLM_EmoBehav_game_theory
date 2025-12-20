import os
from sys import argv
model_path = argv[1]

from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

assert os.path.exists(model_path), f"{model_path} no exists"
llm = LLM(model=model_path)
#llm = LLM(model="/data/home/jjl7137/huggingface_models/Zyphra/Zamba2-1.2B-Instruct")
try:
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    with open('supported_models.txt', 'a+') as f:
        f.write(f"{model_path} works\n")


except:
    print(f"{model_path} not work")



