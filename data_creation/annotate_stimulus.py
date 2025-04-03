import asyncio
import json
import multiprocessing
import os
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


class AnnotatedStimulus(BaseModel):
    stimulus: str
    trigger_type: Literal["None", "PlaceHolderA", "PlaceHolderB"]
    reasoning: str


async def annotate_sample(sample, stimulus_types):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are a psychologist. You will be given a sample of emotion stimuli. Please annotate with trigger type it belongs to. 
                            The trigger types are: {stimulus_types}
                            if the stimulus does not belong to any of the trigger types, return trigger_type as None
                            """,
            },
            {"role": "user", "content": sample},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnnotatedStimulus",
                "type": "object",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "stimulus": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "trigger_type": {
                            "type": "string",
                            "enum": stimulus_types + ["None"],
                        },
                    },
                    "required": ["stimulus", "trigger_type", "reasoning"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


async def annotate_stimulus(
    data: list[str], emotion2stimulus: dict, emotion: str
) -> list[dict]:
    stimulus_types = [
        str(trigger["tag"]) for trigger in emotion2stimulus[emotion]["triggers"]
    ]

    annotated_data = []
    bsz = 10
    for i in tqdm(range(0, len(data), bsz), desc="Processing batches"):
        batch = data[i : i + bsz]

        tasks = [
            asyncio.create_task(annotate_sample(sample, stimulus_types))
            for sample in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling exceptions individually.
        for res in results:
            if isinstance(res, Exception):
                # Log or handle the exception as needed
                print(f"Error during annotation: {res}")
                continue
            else:
                annotated_data.append(res)
    with open(f"data_creation/stimulus_data/{emotion}_annotated.json", "w") as f:
        json.dump(annotated_data, f, indent=4)

    return annotated_data


if __name__ == "__main__":
    # from pprint import pprint

    # pprint(AnnotatedStimulus.model_json_schema())

    emotion = "disgust"
    with open(f"data_creation/stimulus_data/{emotion}.json", "r") as f:
        data = json.load(f)

    with open("data_creation/stimulus_categories.json", "r") as f:
        emotion2stimulus = json.load(f)

    asyncio.run(annotate_stimulus(data, emotion2stimulus, emotion))
