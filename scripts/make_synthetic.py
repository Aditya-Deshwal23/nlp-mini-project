"""Generate tiny, obviously fake transcript fixtures for tests only."""

from pathlib import Path

CALLS = [
    (
        "call_01_normal",
        "Operator: We will now begin the question and answer session.\nAda Example -- Analyst: What changed?\nCFO -- Executive: Costs improved.",
    ),
    ("call_02_no_qa", "Operator: Welcome to the call.\nCEO -- Executive: Results were strong."),
    (
        "call_03_multi_part",
        "Operator: Q&A\nBob Sample -- Analyst: What changed? And why did it happen?\nCEO -- Executive: Demand increased.",
    ),
    (
        "call_04_followup",
        "Operator: Q&A\nCara Demo -- Analyst: What is growth?\nCEO -- Executive: Growth was steady.\nCara Demo -- Analyst: Can you clarify margins?\nCFO -- Executive: Margins improved.",
    ),
    ("call_05_operator_only", "Operator: Q&A\nOperator: There are no analyst questions."),
    (
        "call_06_no_affiliation",
        "Operator: Q&A\nDrew Fake -- Analyst: What is next?\nCEO -- Executive: We have a plan.",
    ),
    (
        "call_07_two_responders",
        "Operator: Q&A\nEli Mock -- Analyst: How is demand?\nCEO -- Executive: Demand is healthy.\nCFO -- Executive: Cash conversion is also healthy.",
    ),
    (
        "call_08_trailing_question",
        "Operator: Q&A\nFay Placeholder -- Analyst: What should we watch?",
    ),
    (
        "call_09_interrupted",
        "Operator: Q&A\nGus Test -- Analyst: What happened--\nCEO -- Executive: We paused the program. [Interrupted]",
    ),
    (
        "call_10_second_normal",
        "Operator: question-and-answer session\nHana Fiction -- Analyst: Is pricing stable?\nCFO -- Executive: Pricing is stable.",
    ),
    (
        "call_11_second_followup",
        "Operator: Q&A\nIra Sample -- Analyst: How is revenue?\nCEO -- Executive: Revenue grew.\nIra Sample -- Analyst: What about costs?\nCFO -- Executive: Costs were controlled.",
    ),
    (
        "call_12_extra",
        "Operator: Q&A\nJo Example -- Analyst: Can you summarize?\nCEO -- Executive: We remain focused.",
    ),
]


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    output.mkdir(parents=True, exist_ok=True)
    for call_id, text in CALLS:
        (output / f"{call_id}.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
