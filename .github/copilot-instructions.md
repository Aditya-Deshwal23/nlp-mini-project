# Earnings-call NLP project context

This is a Python 3.11 coursework project. It analyses one exchange at a time:
one analyst question followed by the executive answers that follow.

Constraints:
- No GPU, paid APIs, or LLM calls.
- Use spaCy `en_core_web_sm`, scikit-learn, statsmodels, gender-guesser, and Streamlit.
- Argument labels are CLAIM, EVIDENCE, REBUTTAL, REQUEST, and OTHER.
- Quality is a transparent weighted rubric, not a black box.
- Treat gender as inferred, probabilistic, binary perceived gender from first names.
- Findings are associations, never proof of bias.

Keep changes readable, deterministic, tested, and documented.

