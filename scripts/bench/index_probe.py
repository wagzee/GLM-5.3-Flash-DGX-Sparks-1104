#!/usr/bin/env python3
"""Which planted facts each DSA layer's indexer selected, on tier 2's haystacks (needs the engine's TF_GLM_INDEX_PROBE).

  run      asks tier 2 questions (longctx_suite.build) on a pair started with TF_GLM_INDEX_PROBE=1 and reads rank 0's
           probe lines from the container log: for the row that writes the answer's first token (the prompt's last
           row) and the first decode rows, the share of each evidence line's tokens in the 2,048 tokens the layer
           read, the lines fully read, and the same for the decoys
  selftest checks the needle locator and the probe line parser offline

The question this answers: when tier 2 gets an item wrong, did the layers not see the evidence (a selection miss,
which a wider or better selection could fix) or did they see it and the model still answered wrong?

Positions: /tokenize with `messages` renders the served chat template, so its ids are the prompt's tokens and an id's
index is its position (the run checks their count against usage). Tokenizing the user text alone would not do: the
chat route reads a user's "<think>" or "<tool_call>" as text, where a bare /tokenize prompt reads special tokens.
"""
import argparse, json, os, re, subprocess, sys, time
import benchlib
import longctx_suite as suite

PROBE_RE = re.compile(r"glm index probe layer=(\d+) pos=(\d+) rows=(\d+) pools=(\S*)")


def tokenize(url, model, text):
    r, _ = benchlib.post(url, "/tokenize", {"model": model, "prompt": text})
    return r["tokens"]


def chat_tokens(url, model, content):
    """The chat prompt's token ids as the chat route builds them (thinking off)."""
    r, _ = benchlib.post(url, "/tokenize", {"model": model, "messages": [{"role": "user", "content": content}],
                                          "chat_template_kwargs": {"enable_thinking": False}})
    return r["tokens"]


def find_all(seq, sub):
    """Start indexes of sub in seq (non-overlapping)."""
    out, n, i = [], len(sub), 0
    first = sub[0]
    while True:
        try:
            i = seq.index(first, i)
        except ValueError:
            return out
        if seq[i:i + n] == sub:
            out.append(i)
            i += n
        else:
            i += 1


def spans(url, model, ids, sentences, cache):
    """Token spans [s, e) (global positions) of each distinct sentence's occurrences in the user text's ids."""
    out = {}
    for sent in set(sentences):
        if sent not in cache:
            toks = tokenize(url, model, "\n" + sent + "\n")
            core = toks[1:-1] if len(toks) > 2 else toks  # the line without its newlines, which merge with neighbours
            cache[sent] = core
        core = cache[sent]
        out[sent] = [(i, i + len(core)) for i in find_all(ids, core)]
    return out


def parse_pools(text):
    picked = set()
    for part in text.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            picked.update(range(int(a), int(b) + 1))
        else:
            picked.add(int(part))
    return picked


def probe_lines(text):
    """{pos: {layer: pools}} from the log text."""
    out = {}
    for line in text.splitlines():
        m = PROBE_RE.search(line)
        if m:
            out.setdefault(int(m[2]), {})[int(m[1])] = parse_pools(m[4])
    return out


def coverage(picked, occ):
    """Share of the occurrences' tokens whose pool was picked, and the occurrences read whole."""
    toks = [t for s, e in occ for t in range(s, e)]
    if not toks:
        return None, 0
    got = sum((t // 4) in picked for t in toks)
    whole = sum(all((t // 4) in picked for t in range(s, e)) for s, e in occ)
    return got / len(toks), whole


def docker_logs(name, since, until):
    return subprocess.run(["docker", "logs", "--since", f"{since:.3f}", "--until", f"{until:.3f}", name],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8", "replace")


def run(a):
    prose = benchlib.corpus("prose", a.source, 1_700_000)
    tasks = set(a.tasks.split(","))
    res = {**benchlib.identity(a.repo), "label": a.label, "seed": a.seed, "items": []}
    cache = {}
    for spec in a.haystacks.split(","):
        length, index = (int(x) for x in spec.split(":"))
        text, qs = suite.build(length, index, a.seed, prose)
        for q in qs:
            if q["task"] not in tasks:
                continue
            content = text + "\n\n" + q["q"]
            ids = chat_tokens(a.url, a.model, content)
            ev = spans(a.url, a.model, ids, q["evidence"], cache)
            dec = spans(a.url, a.model, ids, q["decoys"], cache)
            t0 = time.time() - 1
            r, _ = benchlib.post(a.url, "/v1/chat/completions", {
                "model": a.model, "max_tokens": a.tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": content}]})
            time.sleep(1)
            lines = probe_lines(docker_logs(a.container, t0, time.time()))
            n = r["usage"]["prompt_tokens"]
            if n != len(ids):
                sys.exit(f"{q['id']}: the prompt is {n} tokens, /tokenize gave {len(ids)}: positions would not match")
            answer = r["choices"][0]["message"].get("content") or ""
            ok, part = suite.score(q["task"], q["want"], q["avoid"], answer)
            last = n - 1
            rows = [p for p in sorted(lines) if p >= last][: a.rows]
            layers = sorted({l for p in rows for l in lines[p]})
            per = {}
            ev_occ = [o for s in set(q["evidence"]) for o in ev[s]]
            dec_occ = [o for s in set(q["decoys"]) for o in dec[s]]
            for l in layers:
                cov = [coverage(lines[p][l], ev_occ) for p in rows if l in lines[p]]
                dcov = [coverage(lines[p][l], dec_occ) for p in rows if l in lines[p]] if dec_occ else []
                per[l] = {"evidence": round(sum(c[0] for c in cov) / len(cov), 3),
                          "evidence_whole": round(sum(c[1] for c in cov) / len(cov), 2),
                          "decoys": round(sum(c[0] for c in dcov) / len(dcov), 3) if dcov else None}
            item = {"id": q["id"], "task": q["task"], "ok": ok, "answer": answer[:120], "want": q["want"],
                    "prompt_tokens": n, "evidence_lines": len(ev_occ), "decoy_lines": len(dec_occ),
                    "evidence_found": len(ev_occ) == len(q["evidence"]), "rows": rows, "layers": per}
            mean = sum(v["evidence"] for v in per.values()) / max(len(per), 1)
            print(json.dumps({"id": q["id"], "ok": ok, "answer": answer[:40], "lines": f"{len(ev_occ)}/{len(q['evidence'])}",
                              "mean_evidence_read": round(mean, 3),
                              "by_layer": {l: v["evidence"] for l, v in per.items()}}), flush=True)
            res["items"].append(item)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print(f"wrote {a.out}")


def selftest(_):
    assert find_all([1, 2, 3, 1, 2, 3, 1, 2], [1, 2, 3]) == [0, 3]
    assert parse_pools("1-3,7,9-10") == {1, 2, 3, 7, 9, 10}
    lines = probe_lines("info: glm index probe layer=3 pos=99 rows=1 pools=0-2,20\nother\n"
                        "info: glm index probe layer=7 pos=99 rows=1 pools=5")
    assert lines == {99: {3: {0, 1, 2, 20}, 7: {5}}}, lines
    # tokens 8..11 are pool 2, 80..83 pool 20: both occurrences read whole at layer 3, none at layer 7
    assert coverage(lines[99][3], [(8, 12), (80, 84)]) == (1.0, 2)
    assert coverage(lines[99][7], [(8, 12)]) == (0.0, 0)
    assert coverage({2}, [(6, 12)]) == (4 / 6, 0)
    print("index_probe selftest: pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8020")
    r.add_argument("--model", default=os.environ.get("SERVED_NAME", "zai/glm-5.3-flash"))
    r.add_argument("--label", required=True)
    r.add_argument("--haystacks", default="32768:0,32768:1,131072:0,262144:0", help="length:index,... (tier 2's)")
    r.add_argument("--tasks", default="cnt,agg,vt,hop", help="which of the ten questions")
    r.add_argument("--tokens", type=int, default=24, help="answer tokens (thinking off)")
    r.add_argument("--rows", type=int, default=4, help="rows averaged: the prompt's last and the first decode rows")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--container", default="glm53-spark")
    r.add_argument("--source", default=os.path.expanduser("~/tmp-qb/tf"))
    r.add_argument("--repo", default=benchlib.default_repo())
    r.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
