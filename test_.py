from src.workflows.generation_workflow import GenerationWorkflow


def main():
    print("=" * 80)
    print("GENERATION WORKFLOW TEST")
    print("=" * 80)

    workflow = GenerationWorkflow(
        k=3
    )

    query = """
Write a Python function called sorted_list_sum(lst).

The function receives a list of strings.
Remove all strings that have odd lengths.
Return the remaining strings sorted by length in ascending order.
If two strings have the same length, preserve their original order.
""".strip()

    print("\nUser Query:")
    print("-" * 80)
    print(query)

    print("\nRunning Generation Workflow...")
    print("-" * 80)

    result = workflow.run(
        query=query
    )

    print("\n" + "=" * 80)
    print("WORKFLOW RESULT")
    print("=" * 80)

    print("\nGeneration Mode:")
    print(result["generation_mode"])

    print("\nRetrieval Relevant:")
    print(result["retrieval_relevant"])

    print("\nRetrieved Documents:")
    print(len(result["retrieved_documents"]))

    print("\n" + "=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)

    print(result["response"])


if __name__ == "__main__":
    main()