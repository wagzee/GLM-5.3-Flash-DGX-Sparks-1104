# Long-context baselines

Three tools in `scripts/bench/` check an engine or recipe change against the latest results here, cheapest first.
All of them run on the head node from the recipe checkout, with the pair serving and nothing else on it.

| Check | Tool | Time | What it catches |
| --- | --- | --- | --- |
| Speed + exactness | `attn_baseline.py` | ~15 min | prefill and decode speed at 8K-256K, recall smoke test, reply hashes |
| Tier 1 quality | `divergence.py` | ~10 min | any numeric change, token by token, by depth; prediction loss at depth |
| Tier 2 quality | `longctx_suite.py` | ~1-2 h | task accuracy at 32K-256K, thinking off and on, and reasoning tokens spent |

**Which to run:**
- **A change that should be byte-identical** (a faster kernel with the same summation order, a scheduler change):
  the speed run with `compare --exact`. If the hashes match, quality is unchanged and tiers 1 and 2 are not needed.
- **A change that moves numbers** (another summation order, KV or dense width, a wider token selection): also
  tier 1. If tier 1 shows the change within tolerance, or better (lower loss at depth), run tier 2 to see whether
  the difference reaches answers.

## Tier 1: divergence from the reference

```bash
python3 scripts/bench/divergence.py run --label <label> --source <TensorFold clone> --out docs/speed/baselines/<date>-<label>-div.json
python3 scripts/bench/divergence.py compare docs/speed/baselines/<date>-<label>-div.json
```

The model reads two fixed texts, TensorFold's own Zig sources (275K tokens) and its Markdown docs (135K tokens) at
commit `f350cd3e`, in one `/v1/completions` call each with `prompt_logprobs`. Every position's log-probability
depends only on the tokens before it, so one call shows prediction quality at every depth.
- The run reports, by depth bin (0-8K, 8K-32K, 32K-128K, 128K-), the mean negative log-likelihood (NLL, nats a token)
  and how often the model's top choice is the actual token.
- Against the reference (`reference/ref-{code,prose}.bin.gz`, every token's log-probability and top choice) it adds
  top-1 agreement, |delta log-probability| (mean, p99, max) and the NLL change.
- `compare` fails when any bin's NLL rises by more than 0.002 nats or more than 1% of top-1 choices change.
- A lower NLL at depth is an improvement: the model predicts better from far context.

`prompt_logprobs` bypasses the prompt cache, so no restart is needed, and a repeat on the same image is
byte-identical (checked 2026-11-16).

## Tier 2: long-context tasks

```bash
python3 scripts/bench/longctx_suite.py run --label <label> --source <TensorFold clone> --out docs/speed/baselines/<date>-<label>-tasks.json
python3 scripts/bench/longctx_suite.py compare docs/speed/baselines/<baseline>-tasks.json docs/speed/baselines/<date>-<label>-tasks.json
```

There are 220 items: 6 haystacks at 32K, 3 at 128K and 2 at 256K, each holding ten questions, asked with thinking off
(512-token budget) and on (8,192). The haystacks are real prose with planted facts, and each has these questions:

| Task | Count | What the model must do |
| --- | --- | --- |
| `mk` | 2 | find one key's number among eight keys' numbers |
| `mv` | 2 | find all four codes registered for one key |
| `vt` | 2 | follow a chain of 4 variable copies, beside a decoy chain |
| `hop` | 2 | join two facts far apart (a person's employer, the employer's city), beside decoy companies |
| `cnt` | 1 | count a log line (3-9 times) beside a decoy line |
| `agg` | 1 | list a traveller's five cities beside a decoy traveller's three |

`compare` pairs the items: accuracy by length and task, the items lost and gained, and McNemar's exact p. It fails
when more items are lost than gained at p < 0.05. **Reasoning efficiency** is reported for the items both runs got
right: reasoning tokens before and after. Fewer tokens for the same answers is an improvement, but only while
accuracy holds.

## Speed and exactness

`scripts/bench/attn_baseline.py` measures the served pair at 8K, 32K, 128K and 256K tokens of context. Compare a
change against the latest baseline before it ships:

```bash
# on the head node, from the recipe checkout, the pair serving and otherwise idle
python3 scripts/bench/attn_baseline.py run --label <what changed> --out docs/speed/baselines/<date>-<label>.json
python3 scripts/bench/attn_baseline.py compare docs/speed/baselines/<baseline>.json docs/speed/baselines/<date>-<label>.json
python3 scripts/bench/attn_baseline.py compare ... --exact   # for a change that must not move a single token
```

`compare` exits 1 when prefill or decode speed drops by more than `--tol` percent (default 3) at any context, when
recall drops, or (with `--exact`) when any reply's hash changes.

## What one run measures

For each context the script builds a haystack of random words of that length with four vault codes planted at 10%,
35%, 60% and 90% of its depth, the question at the end. It then sends three greedy requests on one stream:

| Request | What it gives |
| --- | --- |
| `cold`: `max_tokens` 1, a prompt the cache has not seen | TTFT and prefill tok/s at that length |
| `warm`: the same prompt again | the resume cost (the prompt cache restores to the chunk grid and reads the rest) |
| `decode`: 512 tokens, `ignore_eos` | decode tok/s at that context, `(512 - 1) / (its time - warm's time)`; recall of the four codes; the reply's hash |

Then the three greedy identity prompts used since 2026-11-15 (short, ~3K, ~9K tokens: `4f16ce754d1cf860`,
`71c638fe62e9ee76` and `3c75bb14d8ee7d8d` on bf16 dense).

The haystacks depend only on the context and `--seed`, so two runs at the same seed send the same prompts and a
byte-identical image gives byte-identical replies. Recall shows whether the DSA layers' 2,048-token selection still
finds facts deep in the context; the hashes show whether anything moved at all.

## Taking a baseline

1. **Restart the pair first** (`./stop.sh`, `./start.sh`), with `LEARN_GIB=0`. A prompt already in the prompt cache
   would not be cold: the result then carries `"warning"` on that context and its TTFT is not a prefill time.
2. Nothing else may run on the pair; decode is measured at one stream.
3. Run `run`, then commit the JSON here.

**The GPU split (optional).** Restart with the engine's event profiler on, then run with the log:

```bash
./stop.sh && TF_GLM_PROF=1 ./start.sh
python3 scripts/bench/attn_baseline.py run --profiled --log docker:glm53-spark --label <label>-prof \
  --out docs/speed/baselines/<date>-<label>-prof.json
./stop.sh && ./start.sh   # back to normal serving
```

Each context then also holds `profile`:
- `prompt_chunks` is the first, middle and last prompt chunk on rank 0: wall ms and the main stream's GPU ms per
  stage (`kda`, `mla`, `index_score`, `index_select`, `experts`, ...), summed over the layers. Comparing the first and
  last chunk shows what grows with depth. Stages nest (`ffn` holds `experts` and `shared_expert`, for example), so
  their sum exceeds the wall time; side-stream spans are left out.
- `decode` gives each window shape (`verify N rows`) with its section means in ms per call (`kda`, `mla`, `routed`,
  ...).

`TF_GLM_PROF` turns prompt pairing off and adds events, so a profiled run's speeds are lower than a plain run's.
`compare` refuses to compare a profiled run with a plain one. Run it right after the restart, or with a `--seed` not
used since, so its prompts are cold. The profiler lines go to the container's live log only (`docker logs`), not to
`logs/current/rank0.log`, which `start.sh` copies at start.

## Results

| File | What |
| --- | --- |
| `2026-11-16-bf16-2048-speed.json` | speed and exactness, plain |
| `2026-11-16-bf16-2048-prof.json` | the same with `TF_GLM_PROF=1` (`--seed 1`), the GPU split |
| `2026-11-16-bf16-2048-div.json`, `reference/` | tier 1: the reference |
| `2026-11-16-bf16-2048-tasks.json` | tier 2 |

All four are from image `e612c82df940` (recipe `d55ea05`: Mia's main `80be4d6` plus `PROMPT_CHUNK`), with `.env`
`DENSE=bf16 KV=bf16 CONTEXT=500000` and the defaults otherwise (reserve 6.5 GiB, `CACHE_FIRST=1`, a 5.0 GiB prompt
cache, `PARALLEL=4`, DFlash2).

### 2026-11-16 baseline

| Context | Prefill tok/s | TTFT s | Decode tok/s (1 stream) | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 1,561 | 5.3 | 34.1 | 4/4 |
| 32K | 1,670 | 19.7 | 34.0 | 4/4 |
| 128K | 1,670 | 78.8 | 38.9 | 4/4 |
| 256K | 1,613 | 162.9 | 33.6 | 4/4 |

Greedy hashes `4f16ce754d1cf860`, `71c638fe62e9ee76`, `3c75bb14d8ee7d8d`. Decode varies with how many of DFlash2's
drafts the reply accepts, not with depth.

**Tier 1** (NLL in nats a token; top-1 is the share of positions whose top choice is the actual token):

| Depth | Code NLL | Code top-1 | Prose NLL | Prose top-1 |
| --- | ---: | ---: | ---: | ---: |
| 0-8K | 0.871 | 81.3% | 2.378 | 49.2% |
| 8K-32K | 0.701 | 85.3% | 2.337 | 50.5% |
| 32K-128K | 0.574 | 87.5% | 1.758 | 59.4% |
| 128K- | 0.633 | 85.1% | 1.327 | 69.5% |

**Tier 2:** 85/110 with thinking off, 99/110 with thinking on (58,459 reasoning tokens).

| Task | Off | On |
| --- | ---: | ---: |
| `mk` | 22/22 | 22/22 |
| `mv` | 21/22 | 22/22 |
| `vt` | 10/22 | 22/22 |
| `hop` | 21/22 | 22/22 |
| `agg` | 10/11 | 10/11 |
| `cnt` | 1/11 | 1/11 |

By length, off/on: 32K 48/60 and 53/60, 128K 21/30 and 28/30, 256K 16/20 and 18/20. Counting repeats (`cnt`) fails
at every length in both modes. That is the model's own weak spot and the clearest place a change could show a gain.

### Where the time goes (profiled, rank 0)

**A 2,048-row prompt chunk** takes about 1.19 s at the start of a prompt and 1.27 s at 131K-deep. Main-stream GPU
ms, summed over layers (nested stages):

| Stage | Chunk at 0 | Chunk at 131K |
| --- | ---: | ---: |
| `experts` (routed MoE) | 429 | 364 |
| `output_projection` | 230 | 227 |
| `projections` | 172 | 180 |
| `shared_expert` | 133 | 131 |
| `kda` (the 34 linear-attention chains) | 102 | 104 |
| `mla` (the 11 DSA layers' attention) | 96 | 134 |
| `index_score` + `index_select` (DSA indexer) | 0 | 95 |

The attention stages (`kda`, `mla` and the indexer) take about 17% of a chunk at the start and 26% at 131K-deep. The
indexer is the only part that grows with depth, and at 256K it is still under a tenth of a chunk. Prefill speed is
flat from 32K to 256K.

**A decode step** (`verify 2 rows`, the commonest shape) takes 59 ms at 8K and 61 ms at 256K:

| Section | ms |
| --- | ---: |
| `kda` (KDA blocks: their bf16 input projections and the chains) | 22.0 |
| `routed` experts | 18.0-18.5 |
| `mla` (DSA attention and indexer) | 8.2 at 8K, 9.2 at 256K |
| `router`, `head`, `dense`, gathers, `shared` | ~11 |

Long context costs this model's decode about 1 ms a step at 256K. The KDA blocks are the largest section, and the
2026-11 decode notes (`codex-kda-c1-brief.md`) put most of that bucket in the input projections' weight reads, not the
chain.
