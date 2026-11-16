#!/usr/bin/env python3
"""Tier 1 quality: the model's per-token log-probabilities over fixed long texts, against a stored reference.

  run      scores the corpora on the running pair: one /v1/completions call each with prompt_logprobs, every
           position at once (prompt_logprobs bypasses the prompt cache, so every row is computed). Writes a summary
           JSON and, with --save-ref, the per-token reference (.bin.gz) that later runs compare against.
  compare  a run's summary with its reference's, by depth: exits 1 past the tolerances.

Why this measures attention quality: position p's log-probability depends only on tokens 0..p, so one 262K-token
call gives the model's prediction quality at every depth. A change that loses the context shows as the deep bins'
negative log-likelihood (NLL) rising; one that uses it better lowers it. Against the reference, top-1 agreement and
|delta logprob| catch any numeric change token by token, with no sampling noise.

The corpora (benchlib.CORPORA) are TensorFold's own Zig sources (code, ~262K tokens) and Markdown docs (prose,
~131K tokens) at the commit the recipe builds, read from a local clone (--source); they never change.
"""
import argparse, array, gzip, json, math, os, sys
import benchlib

CHARS = {"code": 940_000, "prose": 512_000}
BINS = [(0, 8192), (8192, 32768), (32768, 131072), (131072, 1 << 30)]
REF_FIELDS = 4  # actual id, actual logprob, top-1 id, top-1 logprob


def score(url, model, text, k):
    r, secs = benchlib.post(url, "/v1/completions", {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0,
                                                   "prompt_logprobs": k})
    rows = r["choices"][0]["prompt_logprobs"][1:]  # position 0 has no prediction
    ids, lps, top_ids, top_lps = (array.array("I"), array.array("f"), array.array("I"), array.array("f"))
    for row in rows:
        items = list(row.items())
        tok, e = items[0]  # the server writes the prompt's own token first
        best = min(items, key=lambda kv: kv[1]["rank"])
        ids.append(int(tok)); lps.append(e["logprob"]); top_ids.append(int(best[0])); top_lps.append(best[1]["logprob"])
    return {"ids": ids, "lps": lps, "top_ids": top_ids, "top_lps": top_lps, "secs": secs,
            "prompt_tokens": r["usage"]["prompt_tokens"]}


def save_ref(path, s):
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(array.array("I", [len(s["ids"])]).tobytes())
        for key in ("ids", "lps", "top_ids", "top_lps"):
            f.write(s[key].tobytes())


def load_ref(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    n = array.array("I", data[:4])[0]
    out, off = {}, 4
    for key, code in (("ids", "I"), ("lps", "f"), ("top_ids", "I"), ("top_lps", "f")):
        a = array.array(code)
        a.frombytes(data[off:off + 4 * n])
        out[key] = a
        off += 4 * n
    return out


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(s, ref=None):
    """By depth bin: NLL and how often the top choice is the actual token; against a reference, the differences."""
    out = []
    n = len(s["ids"])
    for lo, hi in BINS:
        hi = min(hi, n)
        if lo >= hi:
            continue
        r = range(lo, hi)
        b = {"from": lo, "to": hi, "tokens": hi - lo,
             "nll": round(-sum(s["lps"][i] for i in r) / (hi - lo), 5),
             "top1_is_actual": round(sum(s["top_ids"][i] == s["ids"][i] for i in r) / (hi - lo), 5)}
        if ref:
            d = [abs(s["lps"][i] - ref["lps"][i]) for i in r]
            b.update(ref_nll=round(-sum(ref["lps"][i] for i in r) / (hi - lo), 5),
                     top1_agree=round(sum(s["top_ids"][i] == ref["top_ids"][i] for i in r) / (hi - lo), 6),
                     dlp_mean=round(sum(d) / len(d), 6), dlp_p99=round(pct(d, 0.99), 5), dlp_max=round(max(d), 5))
            b["dnll"] = round(b["nll"] - b["ref_nll"], 5)
        out.append(b)
    return out


def run(a):
    res = {**benchlib.identity(a.repo), "label": a.label, "tf_commit": benchlib.TF_COMMIT, "corpora": {}}
    for name in a.corpora.split(","):
        text = benchlib.corpus(name, a.source, CHARS[name])
        s = score(a.url, a.model, text, a.k)
        ref_path = os.path.join(a.refdir, f"ref-{name}.bin.gz")
        ref = None
        if not a.save_ref and os.path.exists(ref_path):
            ref = load_ref(ref_path)
            if list(ref["ids"]) != list(s["ids"]):
                sys.exit(f"{name}: the tokens differ from the reference's (tokenizer or corpus changed): not comparable")
        c = {"text_sha": benchlib.sha(text), "tokens_sha": benchlib.sha(s["ids"].tobytes()), "prompt_tokens": s["prompt_tokens"],
             "secs": round(s["secs"], 1), "logprobs_sha": benchlib.sha(s["lps"].tobytes() + s["top_ids"].tobytes()),
             "bins": summarize(s, ref)}
        if ref:
            c["reference"] = ref_path
            c["identical_to_reference"] = s["lps"].tobytes() == ref["lps"].tobytes() and s["top_ids"].tobytes() == ref["top_ids"].tobytes()
        if a.save_ref:
            os.makedirs(a.refdir, exist_ok=True)
            save_ref(ref_path, s)
            c["reference"] = ref_path + " (written by this run)"
        print(json.dumps({"corpus": name, **{k: v for k, v in c.items() if k != "bins"}}), flush=True)
        for b in c["bins"]:
            print("   ", json.dumps(b), flush=True)
        res["corpora"][name] = c
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print(f"wrote {a.out}")


def compare(a):
    """A run against its reference (its bins hold the differences): exits 1 past the tolerances."""
    R = json.load(open(a.result))
    bad = []
    for name, c in R["corpora"].items():
        if c.get("identical_to_reference"):
            print(f"{name}: byte-identical to the reference")
            continue
        print(f"{name}:  {'depth':>17} {'NLL ref -> new':>22} {'top-1 agree':>12} {'|dlp| mean':>11} {'p99':>8} {'max':>8}")
        for b in c["bins"]:
            if "ref_nll" not in b:
                sys.exit(f"{a.result}: {name} was not compared with a reference")
            print(f"   {b['from']:>8}-{b['to']:<8} {b['ref_nll']:>9.4f} -> {b['nll']:<9.4f} ({b['dnll']:+.4f}) {100 * b['top1_agree']:>10.3f}% "
                  f"{b['dlp_mean']:>11.5f} {b['dlp_p99']:>8.4f} {b['dlp_max']:>8.3f}")
            if b["dnll"] > a.nll_tol:
                bad.append(f"{name} NLL at {b['from']}-{b['to']} {b['dnll']:+.4f}")
            if a.agree_tol is not None and b["top1_agree"] < 1 - a.agree_tol:
                bad.append(f"{name} top-1 agreement at {b['from']}-{b['to']} {100 * b['top1_agree']:.3f}%")
    if bad:
        print("REGRESSION: " + "; ".join(bad))
        sys.exit(1)
    print("within tolerance (a negative NLL change is better)")


def selftest(_):
    s = {"ids": array.array("I", [1, 2, 3, 4]), "lps": array.array("f", [-1, -2, -0.5, -0.25]),
         "top_ids": array.array("I", [1, 9, 3, 4]), "top_lps": array.array("f", [-1, -0.1, -0.5, -0.25])}
    (b,) = summarize(s, s)
    assert b["tokens"] == 4 and b["nll"] == 0.9375 and b["top1_is_actual"] == 0.75 and b["top1_agree"] == 1 and b["dlp_max"] == 0, b
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "r.bin.gz")
    save_ref(p, s)
    assert load_ref(p)["top_lps"].tobytes() == s["top_lps"].tobytes()
    print("divergence selftest: pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8020")
    r.add_argument("--model", default=os.environ.get("SERVED_NAME", "zai/glm-5.3-flash"))
    r.add_argument("--label", required=True)
    r.add_argument("--corpora", default="code,prose")
    r.add_argument("--source", default=os.path.expanduser("~/tmp-qb/tf"), help="a TensorFold git clone holding TF_COMMIT")
    r.add_argument("--k", type=int, default=3, help="alternatives a position returns (the top-1 is among them)")
    r.add_argument("--refdir", default=os.path.join(benchlib.default_repo(), "docs/speed/baselines/reference"))
    r.add_argument("--save-ref", action="store_true", help="this run becomes the reference")
    r.add_argument("--repo", default=benchlib.default_repo())
    r.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("result")
    # calibrated 2026-11-16 on TF_GLM_PROMPT_K=shape, a summation-order change alone: NLL moved -0.0004..+0.0018,
    # and 1.4-7.1% of top-1 choices changed. So NLL is the gate, at about twice that noise; top-1 agreement is shown
    # but gates only when asked (any change that is not byte-identical flips a few percent of near-ties).
    c.add_argument("--nll-tol", type=float, default=0.005, help="NLL a depth bin may rise (nats a token)")
    c.add_argument("--agree-tol", type=float, default=None, help="share of top-1 choices that may change (off by default)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "compare": compare, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
