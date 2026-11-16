#!/usr/bin/env python3
"""Tier 2 quality: long-context tasks with exact answers, thinking off and on, compared item by item.

  run      asks every question on the running pair and writes one JSON result
  compare  pairs two results by item: accuracy, the items that flipped, McNemar's exact p, and the reasoning tokens
           both results spent on the items both got right (fewer is better when accuracy holds)
  selftest checks the generator and the scorers offline

Each haystack is real prose (TensorFold's Markdown docs, benchlib "prose") of a set length with the needles of ten
questions planted at random depths; the ten are then asked one by one with the haystack first, so the prompt cache
reads the haystack once a mode. Every haystack, needle and question follows from (length, index, --seed): two runs
at the same seed ask the same questions. Replies are greedy.

The ten questions a haystack:
  mk  x2  one key's number among eight keys' numbers                    (RULER multikey NIAH)
  mv  x2  all four codes registered for one key                         (RULER multivalue NIAH)
  vt  x2  every variable that takes a value through a chain of 4 copies, beside a decoy chain (RULER variable tracking)
  hop x2  a person's employer's headquarters city, the two facts apart, beside decoy companies (two-hop)
  cnt x1  how many times a log line appears (3-9 times), beside a decoy line
  agg x1  every city a traveller visited (5), beside a decoy traveller (aggregation)
"""
import argparse, concurrent.futures as cf, json, math, os, random, re, sys
import benchlib

LENGTHS = {32768: 6, 131072: 3, 262144: 2}  # haystacks a length by default: 110 questions a mode
CHARS_PER_TOKEN = 3.9  # the prose corpus on this tokenizer
SYL = "ka lo mi re su ta ne vo ri pa ze lu ko di ma fe go ha ji ny ob ur es ix".split()
CITIES = ("Lisbon Oslo Quito Hanoi Lagos Perth Cusco Tbilisi Bergen Zurich Nairobi Osaka Porto Krakow Tunis Dakar Lima "
          "Riga Sofia Accra Kyoto Malmo Hobart Fresno Durban Ghent Leeds Tallinn Split Cork").split()
FIRST = "Ivo Mara Tobin Elsa Ravi Noor Felix Ines Otto Zara Linus Petra Hugo Alma Bruno Cleo".split()
LAST = "Marsh Okafor Lindqvist Duarte Brandt Sato Novak Ferreira Kowalski Haddad Moreau Quist".split()


def word(rng, n=3):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


def build(length, index, seed, prose):
    """The haystack text and its ten questions for (length, index, seed)."""
    rng = random.Random(f"longctx/{length}/{index}/{seed}")
    chars = int(length * CHARS_PER_TOKEN) - 2000
    start = rng.randrange(len(prose))
    filler = (prose[start:] + "\n\n" + prose)[:chars]
    needles = []  # (relative depth 0..1, sentence)
    qs = []
    used = set()

    def fresh(make):
        while True:
            v = make()
            if v not in used:
                used.add(v)
                return v

    # mk: eight keys, two asked
    keys = [fresh(lambda: word(rng)) for _ in range(8)]
    nums = {k: str(rng.randrange(10**6, 10**7)) for k in keys}
    for k in keys:
        needles.append((rng.random(), f"The special magic number for {k} is {nums[k]}."))
    for k in rng.sample(keys, 2):
        qs.append({"task": "mk", "q": f"What is the special magic number for {k}? Answer with the number only.",
                   "want": [nums[k]], "avoid": [],
                   "evidence": [f"The special magic number for {k} is {nums[k]}."],
                   "decoys": [f"The special magic number for {o} is {nums[o]}." for o in keys if o != k]})
    # mv: two keys with four codes each
    for _ in range(2):
        k = fresh(lambda: word(rng))
        codes = [str(rng.randrange(10**5, 10**6)) for _ in range(4)]
        for c in codes:
            needles.append((rng.random(), f"One of the registered serial codes for {k} is {c}."))
        qs.append({"task": "mv", "q": f"List all of the registered serial codes for {k}. Answer with the codes only, comma-separated.",
                   "want": codes, "avoid": [],
                   "evidence": [f"One of the registered serial codes for {k} is {c}." for c in codes], "decoys": []})
    # vt: a chain of five variables and a decoy chain, each placed in document order
    for _ in range(2):
        val, decoy = (str(rng.randrange(10**4, 10**5)) for _ in range(2))
        chains = []
        for v in (val, decoy):
            names = [fresh(lambda: "".join(rng.choice("BCDFGHJKLMNPQRSTVWXZ") for _ in range(5))) for _ in range(5)]
            depths = sorted(rng.random() for _ in range(5))
            needles.append((depths[0], f"VAR {names[0]} = {v}"))
            for i in range(1, 5):
                needles.append((depths[i], f"VAR {names[i]} = VAR {names[i - 1]}"))
            chains.append(names)
            sents = [f"VAR {names[0]} = {v}"] + [f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, 5)]
            chains[-1] = (names, sents)
        qs.append({"task": "vt", "q": f"Which variables are assigned the value {val}, directly or through other variables? "
                                      "Answer with the variable names only, comma-separated.",
                   "want": chains[0][0], "avoid": chains[1][0], "evidence": chains[0][1], "decoys": chains[1][1]})
    # hop: two people, their companies and four companies' cities
    comps = [fresh(lambda: word(rng, 2) + rng.choice(["tron", "vex", "lid", "corp", "ware"])) for _ in range(4)]
    cities = rng.sample(CITIES, 4)
    for c, city in zip(comps, cities):
        needles.append((rng.random(), f"{c} has its headquarters in {city}."))
    for i in range(2):
        p = fresh(lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}")
        needles.append((rng.random(), f"{p} works at {comps[i]}."))
        qs.append({"task": "hop", "q": f"In which city are the headquarters of the company {p} works at? Answer with the city only.",
                   "want": [cities[i]], "avoid": [c for c in cities if c != cities[i]],
                   "evidence": [f"{p} works at {comps[i]}.", f"{comps[i]} has its headquarters in {cities[i]}."],
                   "decoys": [f"{c} has its headquarters in {city}." for c, city in zip(comps, cities) if c != comps[i]]})
    # cnt: a log line 3-9 times, a decoy line 2-8 times
    tag, dtag = fresh(lambda: word(rng, 2).upper()), fresh(lambda: word(rng, 2).upper())
    n, dn = rng.randint(3, 9), rng.randint(2, 8)
    needles += [(rng.random(), f"Status log: checkpoint {tag} reached.") for _ in range(n)]
    needles += [(rng.random(), f"Status log: checkpoint {dtag} reached.") for _ in range(dn)]
    qs.append({"task": "cnt", "q": f"How many times does the status log say that checkpoint {tag} was reached? Answer with the number only.",
               "want": [str(n)], "avoid": [], "evidence": [f"Status log: checkpoint {tag} reached."] * n,
               "decoys": [f"Status log: checkpoint {dtag} reached."] * dn})
    # agg: a traveller's five cities, a decoy traveller's three
    pool = rng.sample([c for c in CITIES if c not in cities], 8)
    t, dt = fresh(lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"), fresh(lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    needles += [(rng.random(), f"Travel log: {t} visited {c}.") for c in pool[:5]]
    needles += [(rng.random(), f"Travel log: {dt} visited {c}.") for c in pool[5:]]
    qs.append({"task": "agg", "q": f"List every city that {t} visited according to the travel logs. Answer with the city names only, comma-separated.",
               "want": pool[:5], "avoid": pool[5:], "evidence": [f"Travel log: {t} visited {c}." for c in pool[:5]],
               "decoys": [f"Travel log: {dt} visited {c}." for c in pool[5:]]})
    # plant at line starts, deepest first so earlier offsets hold
    cuts = []
    for depth, s in needles:
        at = filler.rfind("\n", 0, max(1, int(depth * len(filler)))) + 1
        cuts.append((at, s))
    text = filler
    for at, s in sorted(cuts, key=lambda x: -x[0]):
        text = text[:at] + s + "\n" + text[at:]
    for i, q in enumerate(qs):
        q["id"] = f"{length}/{index}/{i:02d}-{q['task']}"
    return text, qs


def score(task, want, avoid, answer):
    """1 if the answer is right, else 0; `part` the share of wanted items found."""
    a = answer.strip()
    if task == "cnt":
        m = re.search(r"\d+", a)
        ok = bool(m) and m.group() == want[0]
        return ok, float(ok)
    if task in ("mk", "hop"):
        ok = want[0] in a and not any(x in a for x in avoid)
        return ok, float(want[0] in a)
    found = sum(w in a for w in want)
    ok = found == len(want) and not any(x in a for x in avoid)
    return ok, found / len(want)


def ask(url, model, text, q, thinking, budget):
    r, secs = benchlib.post(url, "/v1/chat/completions", {
        "model": model, "max_tokens": budget, "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking},
        "messages": [{"role": "user", "content": text + "\n\n" + q["q"]}]})
    m = r["choices"][0]["message"]
    u = r["usage"]
    answer = m.get("content") or ""
    ok, part = score(q["task"], q["want"], q["avoid"], answer)
    return {"id": q["id"], "task": q["task"], "thinking": thinking, "ok": ok, "part": round(part, 3),
            "answer": answer[:300], "want": q["want"], "finish": r["choices"][0].get("finish_reason"),
            "completion_tokens": u["completion_tokens"],
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "prompt_tokens": u["prompt_tokens"], "secs": round(secs, 2)}


def run(a):
    prose = benchlib.corpus("prose", a.source, 1_700_000)
    lengths = {int(k): int(v) for k, v in (x.split(":") for x in a.lengths.split(","))}
    modes = [m == "on" for m in a.thinking.split(",")]
    res = {**benchlib.identity(a.repo), "label": a.label, "seed": a.seed, "lengths": lengths,
           "budgets": {"off": a.budget_off, "on": a.budget_on}, "items": []}
    for length, count in lengths.items():
        for h in range(count):
            text, qs = build(length, h, a.seed, prose)
            for thinking in modes:
                benchlib.post(a.url, "/v1/chat/completions", {  # the haystack into the prompt cache
                    "model": a.model, "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking},
                    "messages": [{"role": "user", "content": text + "\n\nReply OK."}]})
                with cf.ThreadPoolExecutor(a.concurrency) as pool:
                    out = list(pool.map(lambda q: ask(a.url, a.model, text, q, thinking,
                                                      a.budget_on if thinking else a.budget_off), qs))
                for it in out:
                    print(json.dumps({k: it[k] for k in ("id", "thinking", "ok", "part", "reasoning_tokens", "finish", "answer")})[:300], flush=True)
                res["items"] += out
    for thinking in modes:
        its = [i for i in res["items"] if i["thinking"] == thinking]
        print(f"thinking {'on' if thinking else 'off'}: {sum(i['ok'] for i in its)}/{len(its)} right, "
              f"{sum(i['reasoning_tokens'] for i in its)} reasoning tokens")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print(f"wrote {a.out}")


def mcnemar(b, c):
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a):
    A, B = (json.load(open(p)) for p in (a.base, a.new))
    if A["seed"] != B["seed"]:
        sys.exit("the two results used different seeds: their questions differ")
    key = lambda i: (i["id"], i["thinking"])
    ia, ib = {key(i): i for i in A["items"]}, {key(i): i for i in B["items"]}
    common = sorted(set(ia) & set(ib))
    print(f"base {A['label']} ({A['image']})  new {B['label']} ({B['image']}), {len(common)} paired items")
    bad = []
    for thinking in (False, True):
        ks = [k for k in common if k[1] == thinking]
        if not ks:
            continue
        mode = "on" if thinking else "off"
        print(f"\nthinking {mode}:  {'group':<12} {'base':>9} {'new':>9} {'lost':>5} {'gained':>6}")
        groups = {}
        for k in ks:
            g = f"{k[0].split('/')[0]}"
            groups.setdefault(g, []).append(k)
            groups.setdefault(k[0].rsplit("-", 1)[1], []).append(k)
        for g, gk in [("all", ks)] + sorted(groups.items(), key=lambda x: (not x[0].isdigit(), int(x[0]) if x[0].isdigit() else x[0])):
            lost = sum(ia[k]["ok"] and not ib[k]["ok"] for k in gk)
            gained = sum(ib[k]["ok"] and not ia[k]["ok"] for k in gk)
            print(f"   {g:<12} {sum(ia[k]['ok'] for k in gk):>4}/{len(gk):<4} {sum(ib[k]['ok'] for k in gk):>4}/{len(gk):<4} {lost:>5} {gained:>6}")
        lost = [k[0] for k in ks if ia[k]["ok"] and not ib[k]["ok"]]
        gained = [k[0] for k in ks if ib[k]["ok"] and not ia[k]["ok"]]
        p = mcnemar(len(lost), len(gained))
        print(f"   McNemar exact p = {p:.4f} ({len(lost)} lost, {len(gained)} gained)")
        if lost:
            print("   lost: " + ", ".join(lost))
        if gained:
            print("   gained: " + ", ".join(gained))
        both = [k for k in ks if ia[k]["ok"] and ib[k]["ok"]]
        if both:
            ra, rb = (sum(x[k]["reasoning_tokens"] for k in both) for x in (ia, ib))
            ca, cb = (sum(x[k]["completion_tokens"] for k in both) for x in (ia, ib))
            rel = lambda x, y: f"{100 * (y / x - 1):+.1f}%" if x else "n/a"
            print(f"   on the {len(both)} items both got right: reasoning tokens {ra} -> {rb} ({rel(ra, rb)}), "
                  f"all tokens {ca} -> {cb} ({rel(ca, cb)})")
        if len(lost) > len(gained) and p < a.alpha:
            bad.append(f"thinking {mode}: {len(lost)} lost against {len(gained)} gained, p = {p:.4f}")
    if bad:
        print("\nREGRESSION: " + "; ".join(bad))
        sys.exit(1)
    print(f"\nno significant loss (alpha {a.alpha})")


def selftest(_):
    prose = ("Some prose line about kernels.\n" * 20000)
    t1, q1 = build(32768, 0, 0, prose)
    t2, q2 = build(32768, 0, 0, prose)
    assert t1 == t2 and q1 == q2 and len(q1) == 10
    for q in q1:
        for w in q["want"]:
            assert w in t1, (q["id"], w)
    assert t1.count("Status log: checkpoint") >= 5
    for q in q1:  # every evidence and decoy line is in the haystack, as often as listed
        for e in set(q["evidence"]) | set(q["decoys"]):
            assert t1.count(e) >= q["evidence"].count(e) + q["decoys"].count(e) and t1.count(e) >= 1, (q["id"], e)
    assert score("cnt", ["7"], [], "7")[0] and not score("cnt", ["7"], [], "There are 6.")[0]
    assert score("vt", ["A", "B"], ["C"], "A, B")[0] and not score("vt", ["A", "B"], ["C"], "A, B, C")[0]
    assert score("mv", ["1", "2", "3", "4"], [], "1, 2, 3") == (False, 0.75)
    assert abs(mcnemar(0, 6) - 0.03125) < 1e-9 and mcnemar(3, 3) == 1.0
    print("longctx_suite selftest: pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8020")
    r.add_argument("--model", default=os.environ.get("SERVED_NAME", "zai/glm-5.3-flash"))
    r.add_argument("--label", required=True)
    r.add_argument("--lengths", default=",".join(f"{k}:{v}" for k, v in LENGTHS.items()), help="tokens:haystacks,...")
    r.add_argument("--thinking", default="off,on")
    r.add_argument("--budget-off", type=int, default=512)
    r.add_argument("--budget-on", type=int, default=8192)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--source", default=os.path.expanduser("~/tmp-qb/tf"), help="a TensorFold git clone holding TF_COMMIT")
    r.add_argument("--repo", default=benchlib.default_repo())
    r.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--alpha", type=float, default=0.05)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "compare": compare, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
