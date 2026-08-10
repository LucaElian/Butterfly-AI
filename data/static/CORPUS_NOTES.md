# ButterflyAI v0.0004 corpus notes

## Spanish Wikipedia / Wikimedia

`01_BUILD_LANGUAGE_CORPUS.bat` obtains plain-text articles through the Spanish Wikipedia MediaWiki API.
For every accepted article ButterflyAI keeps its page ID, title and source URL in:

`data/corpus/wikipedia_sources.jsonl`

Wikimedia text is reusable subject to the applicable Wikimedia project licenses and attribution/share-alike terms. Keep the source manifest with the corpus if you reuse or redistribute the corpus.

Important: material used for language-model pretraining is **not** automatically inserted into ButterflyAI's epistemic memory as a verified fact. Training material teaches statistical language patterns; factual verification remains a separate process.

## Butterfly-authored conversation templates

The basic conversational/epistemic templates in `butterfly/corpus/conversation.py` and `data/static/instruction_seed.txt` are part of ButterflyAI itself. The larger conversational corpus is produced from those templates plus short introductions derived from the corresponding Wikipedia training/validation documents.
