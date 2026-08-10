ButterflyAI v0.00051 — corrective alignment + benchmark suite v0.00042

WHY THIS EXISTS
---------------
v0.0005 showed a large real improvement but was correctly rejected because it still failed identity/state contrast, exact variable binding and arithmetic hard gates.

v0.00051 keeps the same 17,477,376-parameter architecture and the same tokenizer. It starts again from the accepted v0.0004 brain because rejected v0.0005 weights were deleted by policy.

WHAT CHANGED
------------
1) Benchmark v0.00042 is harder and less repetitive.
   - casual Spanish
   - missing accents/punctuation
   - q/que-style input
   - multiple surface forms per skill
   - identity vs state contrast
   - unseen exact-copy targets
   - held-out arithmetic pairs
   - better greeting semantics ("Hey" can be a valid reply to "Buenas")

2) Anti-leak is stronger.
   Benchmark prompts are normalized before comparison, so changing commas, question marks, accents or q/que does not sneak essentially the same sentence into training.

3) Validation is family-held-out.
   Validation wording templates are different families from training templates. We no longer call a random paraphrase split "generalization".

4) Corrective curriculum:
   ROBUST_DIALOGUE -> BINDING_MATH -> EPISTEMIC_CONTRAST -> MIXED_GENERALIZATION

5) Assistant-only objective remains.
   USER text is context. Loss is computed only on Butterfly response tokens.

ORDER
-----
10_PREPARE_V00051.bat
11_BUILD_CORRECTIVE_CORPUS_V00051.bat
12_TRAIN_BUTTERFLY_V00051.bat
13_COMPARE_AND_PROMOTE_V00051.bat

DO NOT rerun 00-09 for this update.
Do not delete models/, data/, .butterfly/ or benchmarks/.
