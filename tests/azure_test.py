from openai import AzureOpenAI

from api_configs import AZURE_OPENAI_CONFIG

deployment = "gpt-4o"

client = AzureOpenAI(**AZURE_OPENAI_CONFIG)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see? return in json format",
        },
    ],
    max_completion_tokens=10000,
    model=deployment,
    response_format={"type": "json_object"},
)

# print(response.choices[0].message.content)


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


response = oai_response(
    "I am going to Paris, what should I see? return in json format",
    client,
    model=deployment,
    response_format={"type": "json_object"},
)

print(response)
