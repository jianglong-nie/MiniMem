"""Run the complete LoCoMo evaluation pipeline."""

from . import answer_question, construct_memory, evaluate_answer


def main():
    print("=" * 60)
    print("Stage 1/3: Constructing memories")
    print("=" * 60)
    construct_memory.main()

    print("\n" + "=" * 60)
    print("Stage 2/3: Answering questions")
    print("=" * 60)
    answer_question.main()

    print("\n" + "=" * 60)
    print("Stage 3/3: Evaluating answers")
    print("=" * 60)
    evaluate_answer.main()

    print("\nFull LoCoMo evaluation completed.")


if __name__ == "__main__":
    main()
