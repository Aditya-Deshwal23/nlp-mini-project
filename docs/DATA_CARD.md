# Data Card

## Source and licence

The checked-in data are 12 obviously synthetic transcript fixtures used for tests and demos. They
are not reported empirical data. An external transcript source, version, licence, URL, and access
date have not yet been recorded: **TODO**.

The project must not report results from a dataset unless its licence permits coursework use.

## Size and coverage

Numbers below are read from `results/tables/data_quality.csv` and generated parquet outputs:

- Calls loaded: **12**
- Calls with a detected Q&A section: **11**
- Exchanges: **12**
- Turns: **38**
- Sentence rows: **25**
- Merged feature rows: **12**
- Median question words: **3.0**
- Median answer words: **3.0**
- Share with empty answers: **0.08333333333333333**
- Share interrupted: **0.08333333333333333**

Analyst coverage in `gender_coverage.csv` is 8 F, 3 M, and 0 Unknown among the eligible analyst
rows. Executive coverage is 0 F, 0 M, and 11 Unknown in that table. These values demonstrate the
fixture pipeline, not population prevalence.

## Collection gaps and known biases

The fixture data are intentionally tiny and obviously fake. Public earnings-call corpora may
under-represent women analysts, vary in transcript formatting, omit affiliations, and use
inconsistent speaker roles. Calls without a detectable Q&A marker are excluded from later parsing.
The current fixture run found no Q&A marker for one call.

## Preprocessing

Loaders map source fields to a canonical turn schema. Cleaning normalises quotes and dashes,
removes bracketed stage directions, collapses whitespace, and preserves `text_raw`. Q&A markers
define the analysis section. Consecutive analyst turns are merged; subsequent executive turns
become the answer. Names are converted to salted IDs before result-facing outputs.
