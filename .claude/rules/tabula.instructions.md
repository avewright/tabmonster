---
description: Always load these instructions. 
---

## Project purpose

**Tabula** is a from-scratch reimplementation and improvement of tabPFN — a transformer-based in-context learning model for tabular classification and regression. The goal is to beat tabPFN's score on standard benchmarks by experimenting with better architectures, richer feature representations, and larger/more diverse training corpora sourced from HuggingFace and Kaggle. Training is fully streaming (no full dataset snapshots in memory) so it runs on consumer VRAM. The model is evaluated autonomously in a continuous experiment loop.

You have access to 8gb of cuda vram. Try to use less than 32gb of ram (loaded in data)
---

LOOP FOREVER:

    Look at the git state: the current branch/commit we're on
    Come up with ideas to get validation score as low as possible
      - new model architecture
      - new data features 
      - new data sources (huggingface, kaggle, etc)
      - performing research
    Ensure the data is not leaking and well formatted 
    REPEAT

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Timeout: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!