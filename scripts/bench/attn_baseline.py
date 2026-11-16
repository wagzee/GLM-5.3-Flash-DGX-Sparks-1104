#!/usr/bin/env python3
"""Long-context baseline: prefill, decode and recall at several context lengths, plus the engine's GPU time split.

  run      measures a running pair and writes one JSON result (docs/speed/baselines/)
  compare  prints two results side by side and exits 1 on a regression past the tolerances
  selftest checks the profiler-log parser on sample lines (scripts/tests/bench-parse.sh)

How to take a result (docs/speed/baselines/README.md has the full procedure):
  1. Restart the pair, so the prompt cache is empty (LEARN_GIB=0), and leave it idle.
  2. python3 scripts/bench/attn_baseline.py run --label <name> --out docs/speed/baselines/<date>-<name>.json
  3. Optional GPU split: restart with TF_GLM_PROF=1 exported, then run again with --log docker:glm53-spark
     --profiled. TF_GLM_PROF turns off prompt pairing and adds events, so a profiled run's times are not
     comparable to a plain run's; compare refuses to mix them.

Per context the script sends three requests on one stream, nothing else running:
  cold      max_tokens 1 on a prompt the cache has not seen: TTFT and prefill tok/s
  warm      max_tokens 1 on the same prompt (resumed from the prompt cache): the resume cost
  decode    max_tokens DECODE with ignore_eos, greedy: (DECODE - 1) / (its wall - warm's wall) tok/s, and the
            reply, which starts with the four planted codes (recall) and is hashed
Then the three greedy identity prompts the recipe's measurements used since 2026-11-15 (short, ~3K, ~9K tokens).
Standard library only; runs on the head node against 127.0.0.1.
"""
import argparse, hashlib, json, os, random, re, subprocess, sys, time, urllib.request

WORDS = ("river stone apple window garden silver motor candle forest pencil harbor ladder violet copper meadow "
         "lantern orbit pepper canyon thunder marble falcon velvet anchor puzzle glacier saddle tunnel cobalt willow").split()
VAULTS = ["Amber", "Basalt", "Cedar", "Dune"]
DEPTHS = [0.10, 0.35, 0.60, 0.90]
TOK_PER_WORD = 1.13  # this tokenizer on WORDS; the result records the real prompt_tokens


def post(url, body, timeout=3600):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.perf_counter() - t


def needle_prompt(ctx, seed):
    """A haystack of ~ctx tokens with four vault codes at fixed depths, the question at the end."""
    rng = random.Random(f"attn-baseline/{ctx}/{seed}")
    n = int(ctx / TOK_PER_WORD) - 120
    words = [rng.choice(WORDS) for _ in range(n)]
    codes = {v: f"{rng.randrange(10**5, 10**6)}" for v in VAULTS}
    for v, d in reversed(list(zip(VAULTS, DEPTHS))):
        words.insert(int(n * d), f"\n(Note: the access code of vault {v} is {codes[v]}.)\n")
    text = (f"[{seed}] Below is a long list of words. Somewhere in it are notes giving access codes of vaults.\n\n"
            + " ".join(words)
            + "\n\nList the access code of each vault, one line each as VAULT=CODE, in the order Amber, Basalt, "
              "Cedar, Dune. Then describe the list of words above in as much detail as you can.")
    return text, codes


def body(url_model, content, max_tokens, ignore_eos=False):
    b = {"model": url_model, "max_tokens": max_tokens, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    if ignore_eos:
        b["ignore_eos"] = True
    return b


def cached(r):
    return (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def reply_text(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


# ---- the profiler log (TF_GLM_PROF=1, rank 0) ----
DECODE_RE = re.compile(r"glm prof (?P<label>[^(]+?) \((?P<calls>\d+) calls\): wall (?P<wall>[\d.]+) ms, gpu (?P<gpu>[\d.]+) ms a call;(?P<rest>.*)")
CHUNK_RE = re.compile(r"glm prompt chunk rank=(?P<rank>\d+) slot=\d+ start=(?P<start>\d+) rows=(?P<rows>\d+) wall_ms=(?P<wall>[\d.]+)")
STAGE_RE = re.compile(r"glm prompt stage rank=(?P<rank>\d+) slot=\d+ start=(?P<start>\d+) rows=\d+ layer=\S+ piece=\S+ site=\S+ "
                      r"stage=(?P<stage>\w+) stream=(?P<stream>\w+) ms=(?P<ms>[\d.]+)")


def parse_log(text):
    """Decode reports by label (calls-weighted section means) and prompt chunks by start (wall, main-stream stage sums)."""
    dec, chunks = {}, {}
    for line in text.splitlines():
        m = DECODE_RE.search(line)
        if m:
            d = dec.setdefault(m["label"].strip(), {"calls": 0, "wall": 0.0, "gpu": 0.0, "sections": {}})
            k = int(m["calls"])
            d["calls"] += k
            d["wall"] += k * float(m["wall"])
            d["gpu"] += k * float(m["gpu"])
            toks = m["rest"].split()
            for name, v in zip(toks[0::2], toks[1::2]):
                d["sections"][name] = d["sections"].get(name, 0.0) + k * float(v)
            continue
        m = CHUNK_RE.search(line)
        if m and m["rank"] == "0":
            chunks.setdefault(int(m["start"]), {"stages": {}})
            chunks[int(m["start"])].update(rows=int(m["rows"]), wall_ms=float(m["wall"]))
            continue
        m = STAGE_RE.search(line)
        if m and m["rank"] == "0" and m["stream"] == "main":
            st = chunks.setdefault(int(m["start"]), {"stages": {}})["stages"]
            st[m["stage"]] = st.get(m["stage"], 0.0) + float(m["ms"])
    out_dec = {}
    for label, d in dec.items():
        k = d["calls"]
        out_dec[label] = {"calls": k, "wall_ms": round(d["wall"] / k, 3), "gpu_ms": round(d["gpu"] / k, 3),
                          "sections_ms": {n: round(v / k, 3) for n, v in sorted(d["sections"].items(), key=lambda x: -x[1])}}
    out_chunks = [{"start": s, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "stages"},
                   "stages_ms": {n: round(v, 3) for n, v in sorted(c["stages"].items(), key=lambda x: -x[1])}}
                  for s, c in sorted(chunks.items())]
    return {"decode": out_dec, "prompt_chunks": out_chunks}


def chunk_summary(chunks):
    """First, middle and last chunk of a prompt: enough to see what grows with depth."""
    if not chunks:
        return []
    pick = sorted({0, len(chunks) // 2, len(chunks) - 1})
    return [chunks[i] for i in pick]


class LogTail:
    """New lines of rank 0's log since the last take: a file, or `docker:NAME` for a container's live log (the
    recipe's logs/current/rank0.log is a copy made at start, so the profiler's lines are only in `docker logs`)."""

    def __init__(self, path):
        self.path = path
        self.docker = path[len("docker:"):] if path and path.startswith("docker:") else None
        self.since = time.time()
        self.pos = os.path.getsize(path) if path and not self.docker else 0

    def take(self):
        if not self.path:
            return ""
        if self.docker:
            now = time.time()
            # the engine logs to stderr, which `docker logs` passes on as stderr: both streams, in order
            out = subprocess.run(["docker", "logs", "--since", f"{self.since:.3f}", "--until", f"{now:.3f}", self.docker],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8", "replace")
            self.since = now
            return out
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        return data.decode("utf-8", "replace")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def run(a):
    url, model = a.url.rstrip("/"), a.model
    log = LogTail(a.log)
    res = {"label": a.label, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "profiled": a.profiled,
           "image": sh("docker ps --filter label=tf.recipe=glm53-spark --format '{{.Image}}' | head -1"),
           "recipe": sh(f"git -C {a.repo} rev-parse --short HEAD"),
           "settings": {k: sh(f"grep -E '^{k}=' {a.repo}/.env | tail -1 | cut -d= -f2-") for k in
                        ("DENSE", "KV", "PROMPT_CHUNK", "PARALLEL", "CONTEXT", "MEMORY_RESERVE_GIB", "CACHE_FIRST", "DRAFTER")},
           "plan": sh(f"grep -a -o 'prompts in passes[^;]*' {a.repo}/logs/current/rank0.log | tail -1"),
           "decode_tokens": a.decode, "contexts": [], "greedy": []}
    for ctx in [int(x) for x in a.contexts.split(",")]:
        text, codes = needle_prompt(ctx, a.seed)
        log.take()
        r, t_cold = post(url, body(model, text, 1))
        prof_prompt = parse_log(log.take())
        r2, t_warm = post(url, body(model, text, 1))
        r3, t_dec = post(url, body(model, text, a.decode, ignore_eos=True))
        prof_dec = parse_log(log.take())
        reply = reply_text(r3)
        found = {v: bool(re.search(rf"{v}\W{{0,6}}{codes[v]}", reply)) for v in VAULTS}
        n = r["usage"]["prompt_tokens"]
        made = r3["usage"]["completion_tokens"]
        c = {"ctx": ctx, "prompt_tokens": n, "cold_cached": cached(r), "warm_cached": cached(r2),
             "ttft_s": round(t_cold, 3), "prefill_tps": round(n / t_cold, 1), "warm_s": round(t_warm, 3),
             "decode_tokens": made, "decode_tps": round((made - 1) / max(t_dec - t_warm, 1e-6), 2),
             "recall": f"{sum(found.values())}/{len(VAULTS)}", "recall_by_vault": found,
             "reply_sha": hashlib.sha256(reply.encode()).hexdigest()[:16], "reply_head": reply[:160]}
        if a.profiled:
            c["profile"] = {"prompt_chunks": chunk_summary(prof_prompt["prompt_chunks"]),
                            "prompt_chunk_count": len(prof_prompt["prompt_chunks"]), "decode": prof_dec["decode"]}
        if c["cold_cached"]:
            c["warning"] = "the cold prompt was partly cached: restart the pair (or pass another --seed) for a cold TTFT"
        print(json.dumps({k: v for k, v in c.items() if k != "profile"}), flush=True)
        res["contexts"].append(c)
    # the recipe's greedy identity prompts (beside.py since 2026-11-15): their hashes follow every image change
    rng = random.Random(1234)
    for name, nw in [("short", 0), ("p3k", 2600), ("p9k", 7800)]:
        words = " ".join(rng.choice(WORDS) for _ in range(nw))
        r, _ = post(url, {"model": model, "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                          "messages": [{"role": "user", "content": f"Here are some words: {words}\nWrite a short poem using five of the words above, then name the most frequent word."}]})
        g = {"name": name, "prompt_tokens": r["usage"]["prompt_tokens"],
             "sha": hashlib.sha256((r["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:16]}
        print(json.dumps(g), flush=True)
        res["greedy"].append(g)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print(f"wrote {a.out}")


def compare(a):
    A, B = (json.load(open(p)) for p in (a.base, a.new))
    if A.get("profiled") != B.get("profiled"):
        sys.exit("one result is profiled (TF_GLM_PROF=1) and the other is not: their times are not comparable")
    bad = []
    print(f"base {A['label']} ({A['image']}, {A['date'][:10]})  new {B['label']} ({B['image']}, {B['date'][:10]})")
    print(f"{'ctx':>7} {'prefill tok/s':>21} {'TTFT s':>17} {'decode tok/s':>19} {'recall':>11}  reply")
    bc = {c["ctx"]: c for c in A["contexts"]}
    for c in B["contexts"]:
        b = bc.get(c["ctx"])
        if not b:
            continue
        dp = 100 * (c["prefill_tps"] / b["prefill_tps"] - 1)
        dd = 100 * (c["decode_tps"] / b["decode_tps"] - 1)
        same = "same" if c["reply_sha"] == b["reply_sha"] else "DIFFERS"
        print(f"{c['ctx']:>7} {b['prefill_tps']:>7} -> {c['prefill_tps']:>7} {dp:+5.1f}% {b['ttft_s']:>7} -> {c['ttft_s']:>7} "
              f"{b['decode_tps']:>6} -> {c['decode_tps']:>6} {dd:+5.1f}% {b['recall']:>4} -> {c['recall']:<4}  {same}")
        if dp < -a.tol:
            bad.append(f"prefill at {c['ctx']} {dp:+.1f}%")
        if dd < -a.tol:
            bad.append(f"decode at {c['ctx']} {dd:+.1f}%")
        if int(c["recall"].split("/")[0]) < int(b["recall"].split("/")[0]):
            bad.append(f"recall at {c['ctx']} {b['recall']} -> {c['recall']}")
        if same != "same" and a.exact:
            bad.append(f"reply at {c['ctx']} differs")
    bg = {g["name"]: g["sha"] for g in A["greedy"]}
    for g in B["greedy"]:
        same = g["sha"] == bg.get(g["name"])
        print(f"greedy {g['name']:<6} {bg.get(g['name'])} -> {g['sha']}  {'same' if same else 'DIFFERS'}")
        if not same and a.exact:
            bad.append(f"greedy {g['name']} differs")
    if bad:
        print("REGRESSION: " + "; ".join(bad))
        sys.exit(1)
    print(f"no regression (speed within {a.tol}%{', replies byte-identical' if a.exact else ''})")


SAMPLE = """info: glm prof verify 2 rows (64 calls): wall 31.20 ms, gpu 30.10 ms a call; gate 0.10 kda 13.00 mla 2.50 routed 9.00
info: glm prof verify 2 rows (64 calls): wall 33.20 ms, gpu 32.10 ms a call; gate 0.10 kda 15.00 mla 2.50 routed 9.00
info: glm prompt chunk rank=0 slot=0 start=0 rows=2048 wall_ms=1300.5000 spans=3
info: glm prompt stage rank=0 slot=0 start=0 rows=2048 layer=0 piece=null site=mixer stage=kda stream=main ms=10.0000
info: glm prompt stage rank=0 slot=0 start=0 rows=2048 layer=3 piece=null site=mixer stage=kda stream=main ms=5.0000
info: glm prompt stage rank=0 slot=0 start=0 rows=2048 layer=3 piece=null site=mixer stage=index_score stream=side ms=7.0000
info: glm prompt stage rank=1 slot=0 start=0 rows=2048 layer=3 piece=null site=mixer stage=mla stream=main ms=99.0000
"""


def selftest(_):
    p = parse_log(SAMPLE)
    d = p["decode"]["verify 2 rows"]
    assert d["calls"] == 128 and d["wall_ms"] == 32.2 and d["sections_ms"]["kda"] == 14.0, d
    (c,) = p["prompt_chunks"]
    assert c["rows"] == 2048 and c["wall_ms"] == 1300.5 and c["stages_ms"] == {"kda": 15.0}, c
    t, codes = needle_prompt(8192, 0)
    assert all(f"vault {v} is {codes[v]}" in t for v in VAULTS) and t == needle_prompt(8192, 0)[0]
    print("attn_baseline selftest: pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8020/v1")
    r.add_argument("--model", default=os.environ.get("SERVED_NAME", "zai/glm-5.3-flash"))
    r.add_argument("--label", required=True)
    r.add_argument("--contexts", default="8192,32768,131072,262144")
    r.add_argument("--decode", type=int, default=512, help="tokens the decode request makes")
    r.add_argument("--seed", type=int, default=0, help="changes every haystack (and so the replies)")
    r.add_argument("--log", help="rank 0's log to read TF_GLM_PROF lines from: docker:glm53-spark (the live container) or a file")
    r.add_argument("--profiled", action="store_true", help="the pair runs with TF_GLM_PROF=1")
    r.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    r.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--tol", type=float, default=3.0, help="percent of prefill or decode speed a change may lose")
    c.add_argument("--exact", action="store_true", help="replies must also be byte-identical")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "compare": compare, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
