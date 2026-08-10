ButterflyAI v0.00041
====================

WHAT THIS VERSION IS
--------------------
v0.00041 is a SYSTEM / EVALUATION PATCH, not a new neural brain.

After the v0.0004 promotion we discovered that the old benchmark could give a
very high score to responses that looked like Spanish but did not actually
answer the user's question. v0.00041 fixes the ruler before we train v0.0005.

After installing this patch:

  system version : v0.00041
  active brain   : v0.0004
  brain weights  : unchanged
  tokenizer      : unchanged
  corpus         : unchanged
  memory         : unchanged

WHAT CHANGED
------------
1. Benchmark suite v0.00041 is semantics-first and deterministic.
   It uses top_k=1 during evaluation so repeated evaluations are stable.

2. It tests basic dialogue directly:
   - Hola / Buenas
   - Gracias
   - Como te llamas
   - Como estas
   - clarification and goodbye

3. It tests comprehension:
   - archivo
   - carpeta
   - API
   - neural-network parameter
   - token
   - epoch

4. It tests instruction following:
   - exact one-word answer
   - exactly one sentence
   - exactly two steps
   - missing-data behavior
   - strict word-count instruction

5. It tests epistemic behavior:
   - 2 + 2
   - rejection of false arithmetic
   - Internet/source verification
   - unknown fictional fact without invention
   - conflicting sources

6. Promotion now has HARD GATES.
   A future candidate cannot promote just because the overall number is higher.
   Critical failures block promotion.

7. IMPORTANT SAFETY FIX:
   Re-running 04_COMPARE_AND_PROMOTE after v0.0004 was already promoted could,
   in the old code, accidentally treat the active v0.0004 as its own candidate.
   v0.00041 makes that impossible. Active and candidate must be distinct and the
   active model path is explicitly protected from candidate deletion.

INSTALL
-------
Extract this ZIP directly over the permanent ButterflyAI folder, for example:

  D:\Downloads\ButterflyAI\

Accept file replacement.

DO NOT delete or rebuild:
  .butterfly\
  models\
  data\corpus\
  benchmarks\

RUN NOW
-------
Run:

  05_REEVALUATE_ACTIVE_V00041.bat

This only reads the active v0.0004 brain and saves a new strict baseline:

  benchmarks\baseline-v0.0004-suite-v0.00041.json

Do NOT retrain anything for v0.00041.
The result becomes the honest baseline we will use to design v0.0005.
