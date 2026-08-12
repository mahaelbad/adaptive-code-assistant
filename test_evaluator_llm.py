from src.llm.llm_client import LLMClient


def main():

    llm = LLMClient(
        temperature=0.0,
        max_tokens=100,
    )

    prompt = """
You are a classification system.

Question:
Write a Python function that filters strings with even length
and sorts them by length.

Retrieved example:
def by_length(arr):
    # sorts integers by value

Is this retrieved example useful for answering the question?

Reply with exactly:
RELEVANT
or:
NOT_RELEVANT

Final answer:
"""

    response = llm.generate(prompt)

    print("=" * 80)
    print("RAW RESPONSE:")
    print(repr(response))
    print("=" * 80)


if __name__ == "__main__":
    main()