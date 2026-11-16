"""Shared by scripts/bench/*.py: requests to the served pair, the fixed text corpora, the run's identity."""
import hashlib, json, os, subprocess, time, urllib.request

# The corpora: files of TensorFold at the commit this recipe builds (README: "Engine"), read from a local clone. The
# same commit and the same rules give the same bytes, so a corpus never changes under a stored reference.
TF_COMMIT = "f350cd3ea253a042e205746928271f9d8f71a82d"
CORPORA = {
    # code an agent reads: the engine's Zig sources
    "code": {"suffix": ".zig", "header": "// ===== {path} =====\n"},
    # prose: its Markdown documentation
    "prose": {"suffix": ".md", "header": "\n\n# ===== {path} =====\n\n"},
}


def corpus(name, source, chars):
    """The first `chars` characters of corpus `name` from the git clone at `source` (files in path order)."""
    spec = CORPORA[name]
    files = subprocess.run(["git", "-C", source, "ls-tree", "-r", "--name-only", TF_COMMIT], check=True,
                           capture_output=True, text=True).stdout.split("\n")
    out, n = [], 0
    for path in sorted(f for f in files if f.endswith(spec["suffix"])):
        body = subprocess.run(["git", "-C", source, "show", f"{TF_COMMIT}:{path}"], check=True,
                              capture_output=True).stdout.decode("utf-8", "replace")
        piece = spec["header"].format(path=path) + body
        out.append(piece)
        n += len(piece)
        if n >= chars:
            break
    text = "".join(out)[:chars]
    if len(text) < chars:
        raise SystemExit(f"corpus {name} holds only {len(text)} characters, {chars} asked")
    return text


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()[:16]


def post(url, route, body, timeout=7200):
    req = urllib.request.Request(url.rstrip("/") + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.perf_counter() - t


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def identity(repo):
    """What served the run: the image, the recipe commit, the .env settings that change numbers, the plan line."""
    return {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "image": sh("docker ps --filter label=tf.recipe=glm53-spark --format '{{.Image}}' | head -1"),
            "recipe": sh(f"git -C {repo} rev-parse --short HEAD"),
            "settings": {k: sh(f"grep -E '^{k}=' {repo}/.env | tail -1 | cut -d= -f2-") or "(default)" for k in
                         ("DENSE", "KV", "HEAD", "PROMPT_CHUNK", "PARALLEL", "CONTEXT", "DRAFTER", "ABSORB_MMA", "W8A8")},
            "plan": sh(f"grep -a -o 'prompts in passes[^;]*' {repo}/logs/current/rank0.log | tail -1")}


def default_repo():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
