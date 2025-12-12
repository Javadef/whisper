import asyncio
from app.services.llm_service import llm_service


async def main():
    prompt = "Say hello in Uzbek"
    print("Prompt:", prompt)
    resp = await llm_service.generate(prompt, system_prompt=None)
    print("Response:\n", resp)


if __name__ == "__main__":
    asyncio.run(main())
