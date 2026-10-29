# Argument annotation guide

Annotate the **primary function** of each sentence. Do not assign multiple
labels: choose the label that best explains what the sentence is doing.

## Labels and fixture examples

### CLAIM

An assertion or stance that would benefit from support.

- “Results were strong.” (`call_02_no_qa.txt`)
- “Growth was steady.” (`call_04_followup.txt`)
- “We remain focused.” (`call_12_extra.txt`)

### EVIDENCE

A fact, number, example, or source offered as support.

- “Costs improved.” (`call_01_normal.txt`)
- “Demand increased.” (`call_03_multi_part.txt`)
- “Cash conversion is also healthy.” (`call_07_two_responders.txt`)

### REBUTTAL

Pushback, contrast, qualification, or a counter-point against an earlier
point. Contrast markers such as “however”, “but”, and “although” are useful
signals, but the sentence must actually contrast or qualify something.

- “However, demand changed.” (synthetic contrast)
- “But margins improved.” (synthetic contrast)
- “Although costs rose, cash was stable.” (synthetic contrast)

### REQUEST

The explicit analyst ask or question.

- “What changed?” (`call_01_normal.txt`)
- “What changed? And why did it happen?” (`call_03_multi_part.txt`)
- “Can you clarify margins?” (`call_04_followup.txt`)

### OTHER

Pleasantries, procedure, greetings, operator instructions, acknowledgements,
or filler that does not make an argument or ask a question.

- “Welcome to the call.” (`call_02_no_qa.txt`)
- “There are no analyst questions.” (`call_05_operator_only.txt`)
- “We will now begin the question and answer session.” (`call_01_normal.txt`)

## Tie-break rules

Rules and models may record several boolean signals, but the gold annotation
has one primary label. Use this priority when a sentence has several
functions: **REQUEST > REBUTTAL > EVIDENCE > CLAIM > OTHER**.

For compound sentences, label the sentence by its dominant function. If one
clause is an explicit question and another clause supplies context, use
REQUEST. If a number appears inside a question, use REQUEST rather than
EVIDENCE. A contrastive sentence is REBUTTAL even when it also contains a
number, unless the number is clearly the sentence’s sole purpose.

Annotate what is written, not what you believe the speaker intended. Mark
procedural operator language and conversational filler as OTHER.
