from src.workflows.generation_workflow import GenerationWorkflow


def main():
    print("=" * 80)
    print("RAG GENERATION WORKFLOW TEST")
    print("=" * 80)

    query = """
Write a Python function called remove_duplicates(numbers).

The function receives a list of integers.
Remove all elements that occur more than once.
Keep the order of the remaining elements unchanged.
Return the resulting list.
"""

    print("\nUser Query:")
    print(query)

    print("\nInitializing Generation Workflow...")

    workflow = GenerationWorkflow(
        k=3
    )

    print("\nRunning workflow...")

    result = workflow.run(
        query
    )

    print("\n" + "=" * 80)
    print("RESULT")
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