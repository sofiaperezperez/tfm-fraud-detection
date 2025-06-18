import os

from openai import OpenAI
from dotenv import dotenv_values

ENV_CONFIG = dotenv_values(".env")
API_KEY = ENV_CONFIG.get("OPENAI_API_KEY")
SERVER_URL = ENV_CONFIG.get("SERVER_URL", "http://http://g4.etsisi.upm.es:8833/v1")
MODEL_ID = ENV_CONFIG.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(
    api_key=API_KEY,
    base_url=SERVER_URL,
)

USER_PROMPT_TEMPLATE = """

""".strip()

def main():
    messages = [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(name="Pablo")
        }
    ]
            
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=256,
        temperature=0.9,
    )

    print("Response:")
    for choice in completion.choices:
        print(choice.message.content.strip())
            

if __name__ == "__main__":
    main()
