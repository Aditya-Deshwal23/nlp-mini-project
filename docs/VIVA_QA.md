# Viva questions and short answers

1. **What is the unit of analysis?** An exchange: one analyst question and the executive answers that follow.
2. **Why not analyse whole calls?** Whole calls mix prepared remarks and Q&A; exchanges preserve the interaction.
3. **Why classical NLP?** It is offline, laptop-friendly, interpretable, and satisfies the no-LLM constraint.
4. **Why sentence level for arguments?** A single answer can contain claims, evidence, rebuttals, and procedure.
5. **What are the labels?** CLAIM, EVIDENCE, REBUTTAL, REQUEST, and OTHER.
6. **What is the label priority?** REQUEST, then REBUTTAL, EVIDENCE, CLAIM, OTHER.
7. **What are silver labels?** Rule-generated labels used as a transparent but imperfect training signal.
8. **What are gold labels?** Human annotations used to evaluate the rules and classifier.
9. **How can silver labels mislead?** Lexicons can miss context, over-trigger on finance terms, and encode their author’s assumptions.
10. **Why keep flags if one label wins?** Flags preserve audit evidence when a sentence has several functions.
11. **What is macro-F1?** The unweighted mean of per-class F1 scores, so rare classes matter.
12. **Why not accuracy alone?** Accuracy can hide failure on minority argument labels.
13. **How was kappa used?** It measures agreement between two annotators beyond chance; current completed agreement is TODO.
14. **What does AQS measure?** A documented proxy for structure, specificity, clarity, calibration, and question focus.
15. **What does AQS not measure?** Truth, intelligence, speaker worth, or objective quality.
16. **Why percentile components?** They place heterogeneous measures on a comparable 0–1 corpus-relative scale.
17. **What are the question weights?** Structure .30, specificity .25, clarity .20, calibration .10, focus .15.
18. **What are the answer weights?** Structure .35, specificity .25, clarity .20, calibration .20.
19. **Why Welch’s t-test?** It compares means without assuming equal group variances.
20. **Why Mann–Whitney?** It provides a rank-based sensitivity analysis for non-normal outcomes.
21. **Why permutation tests?** Shuffling analyst gender within call preserves call composition while testing the observed contrast.
22. **Why clustering?** Exchanges from the same call share context and are not independent.
23. **Why mixed models?** A random call intercept models shared call-level baseline variation.
24. **What does BH do?** Benjamini–Hochberg controls the expected false-discovery rate across a family of tests.
25. **What does a null result mean?** No statistically detectable difference under this design; it does not prove equality or absence of bias.
26. **Why report effect sizes?** A p-value alone does not show practical magnitude.
27. **Why bootstrap intervals?** They quantify uncertainty for effects without relying only on normal approximations.
28. **How does name-based gender fail?** Names are ambiguous, culturally variable, abbreviated, and do not reveal identity.
29. **Why include Unknown?** Excluding Unknown silently would hide coverage and selection limitations.
30. **What does the power analysis say?** It should show detectable effects for observed group sizes; current power table is TODO.
31. **Why use a synthetic fixture?** It makes tests deterministic and avoids presenting fake data as real evidence.
32. **How many fixture calls have Q&A?** 11 of 12.
33. **How many fixture exchanges exist?** 12.
34. **What would you do next?** Use a licensed larger corpus, annotate more sentences, add a second annotator, and complete the planned statistics.
35. **What is the main ethical warning?** Findings are aggregate associations in observational data, never proof of bias or causation.
