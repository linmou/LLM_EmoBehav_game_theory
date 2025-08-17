import os

from openai import OpenAI

print(os.getenv("DASHSCOPE_API_KEY"))

query = """

I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the  actions I can do Pick up a block Unstack a block from on top of another block Put down a block Stack a block on top of another block I have the following restrictions on my actions: I can only pick up or unstack one block at a time. I can only pick up or unstack a block if my hand is empty. I can only pick up a block if the block is on the table and the block is clear. A block is clear ,→ if the block has no other blocks on top of it and if the block is not picked up. I can only unstack a block from on top of another block if the block I am unstacking was really on ,→ top of the other block. I can only unstack a block from on top of another block if the block I am unstacking is clear. Once I pick up or unstack a block, I am holding the block. I can only put down a block that I am holding. I can only stack a block on top of another block if I am holding the block being stacked. I can only stack a block on top of another block if the block onto which I am stacking the block ,→ is clear. Once I put down or stack a block, my hand becomes empty. Once you stack a block on top of a second block, the second block is no longer clear. [STATEMENT] As initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table. ,
, My goal is to have that the orange block is on top of the blue block. My plan is as follows: [PLAN] unstack the blue block from on top of the orange block put down the blue block pick up the orange block stack the orange block on top of the blue block [PLAN END] [STATEMENT] As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table. ,→ ,→ My goal is to have that the orange block is on top of the red block. My plan is as follows:

"""

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print(
        "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
    )
