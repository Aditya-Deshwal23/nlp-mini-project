# Data Dictionary

Generated from `configs/feature_spec.yaml`; the feature table is one row per exchange.

| Name | Description | Type | Family |
| --- | --- | --- | --- |
| `exchange_id` | Stable exchange identifier | `integer` | `control` |
| `call_id` | Call identifier | `string` | `control` |
| `analyst_id` | Salted analyst identifier | `string` | `control` |
| `analyst_gender` | Inferred perceived analyst gender category | `string` | `control` |
| `analyst_gender_conf` | Confidence of inferred analyst gender | `string` | `control` |
| `question_text` | Analyst question text | `string` | `structure` |
| `answer_text` | Executive answer text | `string` | `response` |
| `relevance_tfidf` | TF-IDF cosine similarity of question and answer | `float` | `response` |
| `keyword_overlap` | Jaccard overlap of noun-chunk lemmas | `float` | `response` |
| `courtesy_opener` | Answer begins with a courtesy phrase | `boolean` | `response` |
| `direct_opener` | Answer begins with a direct response | `boolean` | `response` |
| `deflection_rate` | Deflection lexicon matches per answer word | `float` | `response` |
| `hedge_rate` | Hedge lexicon matches per answer word | `float` | `response` |
| `certainty_rate` | Certainty lexicon matches per answer word | `float` | `response` |
| `answer_words` | Number of words in executive answer | `integer` | `response` |
| `question_words` | Number of words in analyst question | `integer` | `response` |
| `answer_to_question_ratio` | Answer words divided by question words | `float` | `response` |
| `answer_numeric_density` | Numeric-token share per 100 answer words | `float` | `specificity` |
| `n_responders` | Number of distinct responders | `integer` | `response` |
| `first_responder_role` | Normalized role of first responder | `string` | `response` |
| `top_responder_word_share` | Largest responder word count divided by answer words | `float` | `response` |
| `sentiment_question` | VADER compound sentiment of question | `float` | `response` |
| `sentiment_answer` | VADER compound sentiment of answer | `float` | `response` |
| `tone_gap` | Answer sentiment minus question sentiment | `float` | `response` |
| `was_interrupted` | Whether the exchange contains an interruption | `boolean` | `interaction` |
| `first_responder_id` | Salted first-responder identifier | `string` | `control` |
| `first_responder_gender` | Inferred perceived first-responder gender category | `string` | `control` |
| `responder_gender_mix` | Gender composition of responders | `string` | `control` |
| `order_in_call` | Zero-based exchange order within call | `integer` | `interaction` |
| `order_norm` | Exchange order normalized within call | `float` | `interaction` |
| `is_followup` | Whether the analyst has spoken earlier in the call | `boolean` | `interaction` |
| `n_followups_by_analyst_in_call` | Number of follow-ups by this analyst in the call | `integer` | `interaction` |
| `call_n_exchanges` | Number of exchanges in the call | `integer` | `interaction` |
| `call_year` | Calendar year of the call | `integer` | `interaction` |
| `call_quarter` | Calendar quarter of the call | `integer` | `interaction` |
| `structure_q` | AQS question structure component | `float` | `outcome` |
| `specificity_q` | AQS question specificity component | `float` | `outcome` |
| `clarity_q` | AQS question clarity component | `float` | `outcome` |
| `calibration_q` | AQS question calibration component | `float` | `outcome` |
| `focus` | AQS question focus component | `float` | `outcome` |
| `AQS_q` | Transparent question Argument Quality Score | `float` | `outcome` |
| `structure_a` | AQS answer structure component | `float` | `outcome` |
| `specificity_a` | AQS answer specificity component | `float` | `outcome` |
| `clarity_a` | AQS answer clarity component | `float` | `outcome` |
| `calibration_a` | AQS answer calibration component | `float` | `outcome` |
| `AQS_a` | Transparent answer Argument Quality Score | `float` | `outcome` |

## Missing-data policy

- Nullable columns: `answer_to_question_ratio, answer_numeric_density, top_responder_word_share, call_year, call_quarter, structure_q, specificity_q, clarity_q, calibration_q, AQS_q, structure_a, specificity_a, clarity_a, calibration_a, AQS_a`.
- Do not silently impute. Pairwise analyses drop NaN values for the variables used; report the retained n.
- There are no timestamps in the source turns, so latency and talk time are not measured.
- Word counts are only a proxy for airtime.
