#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-GPU queue in front of Ollama, with an optional llama.cpp backend. Stdlib only.

Speaks the Ollama API on the front, so clients only change their base URL.
- Models listed in LLAMA_ROUTES are served by a llama.cpp server over its OpenAI API instead of
  Ollama; the request and response are translated here so clients keep speaking Ollama.
- GPU endpoints (chat/generate/embed/v1) run one at a time.
- Order: interactive before batch; within a priority, same model as the last job first
  (fewer model swaps); a batch job waiting > BATCH_MAX_WAIT is treated as interactive.
- Priority comes from the X-Priority header (interactive|batch); default interactive.
- If upstream fails (5xx / connection error) before any byte reached the client, wait
  for Ollama to answer /api/version again, then retry (RETRIES times).
- Everything else (tags, ps, show, version, pull...) is passed through unqueued.
- llama.cpp's own timings (prompt evaluated vs cached, generation) land in the run db; a job whose
  model the router has not loaded is a cold start, shown as "loading model" in the queue.
- GET / is the status page, GET /_status.json its data, GET /_queue the queue state.
"""
import collections, datetime, http.client, json, os, select, socket, sqlite3, threading, time, itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

UP = urlparse(os.environ.get("UPSTREAM", ""))              # Ollama; empty = no Ollama in the path
LLAMA = urlparse(os.environ.get("LLAMA_UPSTREAM", ""))      # llama.cpp server, OpenAI API; empty = disabled
ROUTES = dict(x.split("=", 1) for x in os.environ.get("LLAMA_ROUTES", "").split(",") if "=" in x)
EMBED = urlparse(os.environ.get("EMBED_UPSTREAM", ""))         # a llama.cpp server in embedding mode
EMBED_ROUTES = dict(x.split("=", 1) for x in os.environ.get("EMBED_ROUTES", "").split(",") if "=" in x)
# Ollama model name -> llama.cpp model id. The Ollama names stay authoritative for clients.
PORT = int(os.environ.get("PORT", "11435"))
# What /api/version reports. Clients use it to decide whether the server is "Ollama enough"; the API
# here is the subset they actually call, so a mid-2025 Ollama version is the honest claim.
OLLAMA_COMPAT = os.environ.get("OLLAMA_COMPAT", "0.12.0")
RETRIES = int(os.environ.get("RETRIES", "2"))
RECOVER_WAIT = int(os.environ.get("RECOVER_WAIT", "180"))
BATCH_MAX_WAIT = int(os.environ.get("BATCH_MAX_WAIT", "600"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))
QUEUE_TIMEOUT = int(os.environ.get("QUEUE_TIMEOUT", "3600"))
AFFINITY_MAX_WAIT = int(os.environ.get("AFFINITY_MAX_WAIT", "120"))   # after this wait a job outranks same-model affinity
LOAD_POLL = float(os.environ.get("LOAD_POLL", "5"))   # seconds between asks while llama.cpp says "Loading model"
RUN_DB = os.environ.get("RUN_DB", "")                     # sqlite file for run data; empty = disabled
RUN_PROMPTS = os.environ.get("RUN_PROMPTS", "1") != "0"    # keep prompt and response text next to each row
RUN_PROMPT_MAX = int(os.environ.get("RUN_PROMPT_MAX", "32768"))   # characters kept per field
RUN_KEEP_DAYS = int(os.environ.get("RUN_KEEP_DAYS", "90"))
# Only inference endpoints take the GPU lane. /v1/models and everything else pass straight through.
GPU = ("/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
       "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses")

# ---------- run data ----------
# One row per GPU job that reached an upstream: what ran, for how long, and what came back. sqlite
# because it needs no server and the volume is already mounted; DuckDB would only pay off for
# columnar scans over far more rows than this box produces.
_db = None
DB_LOCK = threading.Lock()
# tokens_in counts the whole prompt; cache_n of it came from llama.cpp's prompt cache, the rest was evaluated
# in prompt_ms. cold = the router did not have the model loaded when the job started.
COLS = ("ts", "model", "backend", "path", "prio", "wait_s", "run_s", "status", "result",
        "retries", "tokens_in", "tokens_out", "gen_tps", "prompt_tps", "note",
        "cache_n", "prompt_ms", "gen_ms", "cold")
COL_TYPES = {"note": "TEXT", "cache_n": "INTEGER", "prompt_ms": "REAL", "gen_ms": "REAL", "cold": "INTEGER"}


def run_db():
    global _db
    if not RUN_DB:
        return None
    with DB_LOCK:
        if _db is None:
            try:
                _db = sqlite3.connect(RUN_DB, check_same_thread=False)
                _db.execute("PRAGMA journal_mode=WAL")
                _db.execute("""CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, ts REAL, model TEXT,
                               backend TEXT, path TEXT, prio TEXT, wait_s REAL, run_s REAL, status INTEGER,
                               result TEXT, retries INTEGER, tokens_in INTEGER, tokens_out INTEGER,
                               gen_tps REAL, prompt_tps REAL, note TEXT)""")
                have = {r[1] for r in _db.execute("PRAGMA table_info(runs)")}
                for col, typ in COL_TYPES.items():         # files written before these columns existed
                    if col not in have:
                        _db.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
                _db.execute("CREATE INDEX IF NOT EXISTS runs_ts ON runs(ts)")
                # Prompt and response text live beside the row, not in it: most queries only want the numbers.
                _db.execute("""CREATE TABLE IF NOT EXISTS calls (run_id INTEGER PRIMARY KEY, prompt TEXT,
                               response TEXT, meta TEXT)""")
                _db.execute("""CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT,
                               detail TEXT)""")
                _db.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")
                _db.commit()
            except Exception:
                _db = None
        return _db


def record(row):
    """Never raises: logging must not fail a request."""
    db = run_db()
    if db is None:
        return
    try:
        with DB_LOCK:
            cur = db.execute(f"INSERT INTO runs ({','.join(COLS)}) VALUES ({','.join(':' + c for c in COLS)})",
                             {c: row.get(c) for c in COLS})
            if db.total_changes % 500 == 0:
                cut = time.time() - RUN_KEEP_DAYS * 86400
                db.execute("DELETE FROM runs WHERE ts < ?", (cut,))
                db.execute("DELETE FROM calls WHERE run_id NOT IN (SELECT id FROM runs)")
                db.execute("DELETE FROM events WHERE ts < ?", (cut,))
            db.commit()
            return cur.lastrowid
    except Exception:
        pass


def keep_text(s):
    """Truncate to the configured size; keep the head, which is what identifies the call."""
    s = s or ""
    return s if len(s) <= RUN_PROMPT_MAX else s[:RUN_PROMPT_MAX] + f"\n[truncated {len(s) - RUN_PROMPT_MAX} chars]"


def extract_prompt(body):
    """The text a human would call 'the prompt': the messages for a chat call, the prompt for a completion."""
    try:
        d = json.loads(body or b"{}")
        if d.get("messages"):
            return "\n".join(f"{m.get('role', '?')}: {m.get('content') or ''}" for m in d["messages"])
        return str(d.get("prompt") or d.get("input") or "")
    except Exception:                                  # whatever the client sent, it is not ours to fail on
        return ""


def save_call(run_id, prompt, response=None, meta=None):
    db = run_db()
    if db is None or run_id is None or not RUN_PROMPTS:
        return
    try:
        with DB_LOCK:
            db.execute("INSERT OR REPLACE INTO calls (run_id, prompt, response, meta) VALUES (?,?,?,?)",
                       (run_id, keep_text(prompt), keep_text(response),
                        json.dumps(meta)[:4096] if meta else None))
            db.commit()
    except Exception:
        pass


def event(kind, **detail):
    """Session log: what the gateway itself did, as opposed to what it served."""
    db = run_db()
    if db is None:
        return
    try:
        with DB_LOCK:
            db.execute("INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
                       (time.time(), kind, json.dumps(detail, default=str)[:4096]))
            db.commit()
    except Exception:
        pass


STARTED = time.time()
cv = threading.Condition()
waiting = []          # tickets
running = None
last_model = None
seq = itertools.count()
history = collections.deque(maxlen=50)
stats = {"done": 0, "failed": 0, "retried": 0, "swaps": 0, "aborted": 0}


def client_gone(sock):
    """True if the client closed its end (EOF). A pipelined next request leaves data to read, which is not 'gone'."""
    try:
        if select.select([sock], [], [], 0)[0]:
            return sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True
    return False


class Ticket:
    """One GPU job, from queue entry to its run row. state: waiting -> running, or loading while the backend
    reads the model; m is the job's slice of the run row, filled in as the job learns it."""
    def __init__(self, model, prio, path):
        self.model, self.prio, self.path = model, prio, path
        self.t, self.n = time.time(), next(seq)
        self.state = "waiting"
        self.llama = ROUTES.get(model) if routed(model) else None
        self.m = {"retries": 0}

    def rank(self):
        # lowest wins: interactive before batch; then jobs that waited past AFFINITY_MAX_WAIT (oldest first) before the
        # rest; among the rest, the loaded model first (fewer swaps), then FIFO.
        age = time.time() - self.t
        batch = self.prio == "batch" and age < BATCH_MAX_WAIT
        aged = age >= AFFINITY_MAX_WAIT
        return (batch, 0 if aged else 1, self.n if aged else 0, 0 if self.model == last_model else 1, self.n)


def acquire(tk, sock):
    """'ok' | 'timeout' | 'gone'. A client that disconnects while queued is dropped before it ever reaches the GPU."""
    global running
    with cv:
        waiting.append(tk)
        deadline = time.time() + QUEUE_TIMEOUT
        while True:
            if running is None and min(waiting, key=Ticket.rank) is tk:
                waiting.remove(tk); running = tk; tk.state = "running"
                return "ok"
            if client_gone(sock):
                waiting.remove(tk); cv.notify_all(); stats["aborted"] += 1
                event("queued_client_gone", model=tk.model)
                return "gone"
            if time.time() > deadline:
                waiting.remove(tk); return "timeout"
            cv.wait(0.5)


def release(tk):
    global running, last_model
    with cv:
        if last_model is not None and tk.model != last_model:
            stats["swaps"] += 1
            event("swap", was=last_model, now=tk.model)
        last_model = tk.model
        running = None
        cv.notify_all()


def upstream(method, path, body, headers, timeout):
    c = http.client.HTTPConnection(UP.hostname, UP.port or 80, timeout=timeout)
    c.request(method, path, body, headers)
    return c, c.getresponse()


def ollama_conn(timeout):
    return http.client.HTTPConnection(UP.hostname, UP.port or 80, timeout=timeout)


def llama_conn(timeout):
    return http.client.HTTPConnection(LLAMA.hostname, LLAMA.port or 80, timeout=timeout)


def llama_upstream(method, path, body, headers, timeout):
    c = http.client.HTTPConnection(LLAMA.hostname, LLAMA.port or 80, timeout=timeout)
    c.request(method, path, body, headers)
    return c, c.getresponse()


def routed(model):
    return bool(LLAMA.hostname) and model in ROUTES


def iso_now():
    """RFC 3339 with microseconds: time.strftime has no %f, datetime does."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def generate_as_chat(body):
    """Ollama /api/generate body -> the equivalent /api/chat body, so one translation serves both."""
    msgs = [{"role": "system", "content": body["system"]}] if body.get("system") else []
    out = {k: v for k, v in body.items() if k not in ("prompt", "system", "suffix", "raw", "context")}
    out["messages"] = msgs + [{"role": "user", "content": body.get("prompt") or "",
                               **({"images": body["images"]} if body.get("images") else {})}]
    return out


def wants_stream(path, body):
    """Ollama's /api/* endpoints stream unless told not to; the OpenAI /v1/* ones do the opposite."""
    default = path.startswith("/api/")
    return bool(body.get("stream", default)) if isinstance(body, dict) else False


def to_openai(body, stream):
    """Ollama /api/chat body -> llama.cpp /v1/chat/completions body."""
    o = body.get("options") or {}
    out = {"model": ROUTES[body.get("model")], "messages": body.get("messages") or [], "stream": stream}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                     ("min_p", "min_p"), ("seed", "seed"), ("num_predict", "max_tokens"),
                     ("stop", "stop"), ("repeat_penalty", "repeat_penalty"), ("presence_penalty", "presence_penalty"),
                     ("frequency_penalty", "frequency_penalty")):
        if src in o:
            out[dst] = o[src]
    fmt = body.get("format")
    if isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"schema": fmt}}
    elif fmt == "json":
        out["response_format"] = {"type": "json_object"}
    kw = dict(body.get("chat_template_kwargs") or {})
    if body.get("think") is False:
        kw["enable_thinking"] = False          # Ollama's think:false maps to the template switch
    if kw:
        out["chat_template_kwargs"] = kw
    return out


def read_llama_chat(r):
    """Read one llama.cpp completion. Returns (body, is_empty, note); an empty body with no tokens is
    reported rather than passed on, because a client cannot tell it apart from a valid empty answer."""
    raw = r.read() or b"{}"
    try:
        d = json.loads(raw)
    except Exception:
        return {}, True, f"unparsable body: {raw[:200]!r}"
    ch = (d.get("choices") or [{}])[0]
    if not ((ch.get("message") or {}).get("content") or "").strip():
        return d, True, f"finish={ch.get('finish_reason')} usage={d.get('usage')} timings={d.get('timings')}"
    return d, False, None


def upstream_error(status, raw):
    """'upstream 400: <message>' from an error body, using the server's own message when it sent JSON."""
    try:
        e = json.loads(raw).get("error")
        msg = e.get("message") if isinstance(e, dict) else e
    except Exception:
        msg = None
    return f"upstream {status}: {msg or raw[:300].decode('utf-8', 'replace')}"


def timing_fields(d):
    """llama.cpp's own accounting of a completion, as run-row fields. prompt_n tokens were evaluated in
    prompt_ms; cache_n more were reused from the prompt cache; tokens_in is their sum."""
    t, u = d.get("timings") or {}, d.get("usage") or {}
    pn, cn = t.get("prompt_n"), t.get("cache_n")
    tps = lambda v: round(v, 2) if v else None
    return {"tokens_in": u.get("prompt_tokens", (pn or 0) + (cn or 0) if pn is not None else None),
            "tokens_out": u.get("completion_tokens", t.get("predicted_n")), "cache_n": cn,
            "prompt_ms": t.get("prompt_ms"), "gen_ms": t.get("predicted_ms"),
            "prompt_tps": tps(t.get("prompt_per_second")), "gen_tps": tps(t.get("predicted_per_second"))}


def ollama_reply(kind, model, text, done, f=None, finish=None, wall=None):
    """One Ollama-shaped message: a /api/chat or /api/generate chunk, or the final one with the counters."""
    out = {"model": model, "created_at": iso_now(), "done": done}
    if kind == "chat":
        out["message"] = {"role": "assistant", "content": text}
    else:
        out["response"] = text
    if done:
        f = f or {}
        out["done_reason"] = finish or "stop"
        out["eval_count"] = f.get("tokens_out") or 0
        out["prompt_eval_count"] = f.get("tokens_in") or 0
        if wall is not None:
            out["total_duration"] = int(wall * 1e9)
        if f.get("prompt_ms"):
            out["prompt_eval_duration"] = int(f["prompt_ms"] * 1e6)
        if f.get("gen_ms"):
            out["eval_duration"] = int(f["gen_ms"] * 1e6)
    return out


def router_models():
    """The router's model list, or [] when it cannot be asked. One fetch, shared by everything below."""
    if not LLAMA.hostname:
        return []
    try:
        c, r = llama_upstream("GET", "/v1/models", None, {}, 10)
        d = json.loads(r.read() or b"{}"); c.close()
        return d.get("data") or []
    except Exception:
        return []


def llama_status(llama_id):
    """The router's view of one model ('loaded', 'loading', 'unloaded', ...), or None if it does not say."""
    m = next((m for m in router_models() if m.get("id") == llama_id), None)
    return ((m or {}).get("status") or {}).get("value")


def wait_alive():
    end = time.time() + RECOVER_WAIT
    while time.time() < end:
        try:
            c, r = upstream("GET", "/api/version", None, {}, 5)
            ok = r.status == 200; r.read(); c.close()
            if ok:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ---------- status page ----------
# One read-only page: backends, the queue, the last 24 hours from the run db, and recent jobs.
STATUS_HTML = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ollama gateway</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#777;--line:#e3e3e3;--ok:#0a6;--bad:#c33;--warn:#b80}
@media(prefers-color-scheme:dark){:root{--bg:#111;--fg:#e8e8e8;--mut:#999;--line:#2a2a2a;--ok:#4c9;--bad:#e66;--warn:#db4}}
body{font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0 auto;max-width:1100px;padding:18px}
h1{font-size:17px;margin:0 0 2px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:22px 0 6px}
.sub{color:var(--mut);font-size:12px;margin-bottom:14px}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:12px}
td.n,th.n{text-align:right}
.cards{display:flex;gap:10px;flex-wrap:wrap}
.card{flex:1 1 220px;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:13px;margin-bottom:4px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--mut)}
.ok{background:var(--ok)}.bad{background:var(--bad)}
.mut{color:var(--mut)}.err{color:var(--bad)}.warn{color:var(--warn)}.run{color:var(--ok);font-weight:600}
.note{color:var(--mut);font-size:12px;white-space:normal;max-width:420px}
button{font:inherit;font-size:12px}
</style>
<h1>ollama gateway</h1>
<div class=sub id=sub>loading…</div>
<div class=cards id=cards></div>
<h2>Queue</h2><div class=wrap><table id=queue></table></div>
<h2>Last 24 hours</h2><div class=wrap><table id=day></table></div>
<h2>Recent jobs</h2><div class=wrap><table id=recent></table></div>
<script>
const e=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=(v,d=1,u="")=>v==null?"<span class=mut>—</span>":(+v).toFixed(d)+u;
const N=new Set(["calls","failed","wait","run","prompt t/s","gen t/s","cache","cold","waited","ran","in","out","age"]);
function table(id,head,rows){document.getElementById(id).innerHTML=
 "<tr>"+head.map(h=>"<th"+(N.has(h)?" class=n":"")+">"+h+"</th>").join("")+"</tr>"+
 (rows.length?rows.map(r=>"<tr>"+r.map((c,i)=>"<td"+(N.has(head[i])?" class=n":"")+">"+c+"</td>").join("")+"</tr>").join("")
  :"<tr><td class=mut colspan="+head.length+">nothing yet</td></tr>")}
const state=s=>s=="running"?"<span class=run>running</span>":s=="loading"?"<span class=warn>loading model</span>":e(s);
function result(j){
 const ok=j.result=="ok", r=ok?"ok":"<span class=err>"+e(j.result)+"</span>";
 return r+(j.note?"<div class=note>"+e(j.note)+"</div>":"")}
function load(){
 fetch("/_status.json").then(r=>r.json()).then(d=>{
  const t=d.day&&d.day.totals, s=d.since_start;
  document.getElementById("sub").textContent="up "+d.uptime+" · "+(t
   ?"last 24 h: "+t.calls+" calls · "+t.failed+" failed · "+t.cold+" cold loads"+(t.cache!=null?" · "+Math.round(t.cache*100)+"% prompt cached":"")
   :"since start: "+s.done+" done · "+s.failed+" failed (set RUN_DB for history)");
  document.getElementById("cards").innerHTML=d.backends.map(b=>
   "<div class=card><b><span class='dot "+(b.ok?"ok":"bad")+"'></span>"+e(b.name)+"</b>"+
   (b.models.length?b.models.map(m=>"<div>"+e(m.name)+" <span class=mut>"+e(m.state)+"</span>"+
     (b.unload?" <button data-m='"+e(m.name)+"'>unload</button>":"")+"</div>").join(""):"<div class=mut>no model loaded</div>")+
   "<div class=mut style='margin-top:6px'>"+e(b.detail)+"</div></div>").join("");
  const q=d.queue;
  table("queue",["state","model","priority","age"],
   (q.running?[[state(q.running.state),e(q.running.model),e(q.running.prio),q.running.age+"s"]]:[])
    .concat(q.waiting.map((w,i)=>["waiting #"+(i+1),e(w.model),e(w.prio),w.age+"s"])));
  table("day",["model","backend","calls","failed","wait","run","prompt t/s","gen t/s","cache","cold"],
   d.day?d.day.models.map(r=>[e(r.model),e(r.backend),r.calls,r.failed||"<span class=mut>0</span>",num(r.avg_wait,1,"s"),num(r.avg_run,1,"s"),
     num(r.prompt_tps),num(r.gen_tps),r.cache==null?"<span class=mut>—</span>":Math.round(r.cache*100)+"%",r.cold]):[]);
  table("recent",["at","model","prio","waited","ran","in","out","prompt t/s","gen t/s","result"],
   d.recent.map(j=>[new Date(j.ts*1000).toLocaleTimeString(),e(j.model),e(j.prio),num(j.wait_s,1,"s"),num(j.run_s,1,"s"),
     j.tokens_in==null?"<span class=mut>—</span>":j.tokens_in+(j.cache_n?" <span class=mut>("+j.cache_n+" cached)</span>":""),
     j.tokens_out==null?"<span class=mut>—</span>":j.tokens_out,num(j.prompt_tps),num(j.gen_tps),result(j)]));
 }).catch(err=>{document.getElementById("sub").textContent="status unavailable: "+err});
}
document.addEventListener("click",ev=>{const m=ev.target.getAttribute&&ev.target.getAttribute("data-m");
 if(m)fetch("/_unload?model="+encodeURIComponent(m),{method:"POST"}).then(load)});
load();setInterval(load,5000);
</script>"""


def _get_json(url, timeout=3):
    try:
        c = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
        c.request("GET", url.path or "/")
        r = c.getresponse()
        body = r.read()
        c.close()
        return json.loads(body or b"{}")
    except Exception:
        return None


def job_view(t, now):
    return {"model": t.model, "prio": t.prio, "path": t.path, "state": t.state, "age": round(now - t.t)}


def backend_cards():
    cards = []
    if LLAMA.hostname:
        d = _get_json(urlparse(f"http://{LLAMA.hostname}:{LLAMA.port or 80}/v1/models"))
        models = [{"name": m.get("id"), "state": (m.get("status") or {}).get("value", "")} for m in (d or {}).get("data", [])]
        cards.append({"name": "llama.cpp", "ok": d is not None, "models": models,
                      "detail": f"{LLAMA.hostname}:{LLAMA.port or 80}" + ("" if d is not None else " unreachable")})
    if UP.hostname:
        d = _get_json(urlparse(f"http://{UP.hostname}:{UP.port or 80}/api/ps"))
        models = [{"name": m.get("name"), "state": f"{(m.get('size_vram') or 0) / 1e9:.2f} GB in VRAM"}
                  for m in (d or {}).get("models", [])]
        cards.append({"name": "ollama", "ok": d is not None, "models": models, "unload": True,
                      "detail": f"{UP.hostname}:{UP.port or 80}" + ("" if d is not None else " unreachable")})
    if EMBED.hostname:
        models = [{"name": k, "state": f"-> {v}"} for k, v in EMBED_ROUTES.items()]
        cards.append({"name": "llama.cpp embeddings", "ok": True, "models": models,
                      "detail": f"{EMBED.hostname}:{EMBED.port or 80}"})
    return cards


def day_summary(hours=24):
    """Per model over the window, from the run db. Rates are token-weighted: total tokens over total time,
    using only rows that carry llama.cpp's timings; the cache share is reused prompt tokens over all prompt tokens."""
    db = run_db()
    if db is None:
        return None
    try:
        with DB_LOCK:
            rows = db.execute("""
                select model, backend, count(*), sum(result != 'ok'), avg(wait_s),
                       avg(case when result = 'ok' then run_s end),
                       sum(case when prompt_ms > 0 then tokens_in - coalesce(cache_n, 0) end) * 1000.0
                           / nullif(sum(case when prompt_ms > 0 then prompt_ms end), 0),
                       sum(case when gen_ms > 0 then tokens_out end) * 1000.0
                           / nullif(sum(case when gen_ms > 0 then gen_ms end), 0),
                       sum(cache_n), sum(case when cache_n is not null then tokens_in end),
                       sum(coalesce(cold, 0))
                from runs where ts > ? group by 1, 2 order by 3 desc""", (time.time() - hours * 3600,)).fetchall()
    except Exception:
        return None
    models = [{"model": m, "backend": b, "calls": n, "failed": f, "avg_wait": w, "avg_run": r,
               "prompt_tps": pt, "gen_tps": gt, "cache": (cn / ct) if ct else None, "cold": cold,
               "_cn": cn or 0, "_ct": ct or 0}
              for m, b, n, f, w, r, pt, gt, cn, ct, cold in rows]
    cn, ct = sum(x.pop("_cn") for x in models), sum(x.pop("_ct") for x in models)
    totals = {"calls": sum(x["calls"] for x in models), "failed": sum(x["failed"] for x in models),
              "cold": sum(x["cold"] for x in models), "cache": (cn / ct) if ct else None}
    return {"hours": hours, "totals": totals, "models": models}


def recent_jobs(limit=20):
    """The last jobs, from the db when there is one (it survives restarts), else from memory. Same row shape either way."""
    db = run_db()
    if db is not None:
        try:
            with DB_LOCK:
                rows = db.execute(f"select {','.join(COLS)} from runs order by id desc limit ?", (limit,)).fetchall()
            return [dict(zip(COLS, r)) for r in rows]
        except Exception:
            pass
    with cv:
        return list(history)[:limit]


def status_json():
    """Everything the page shows, in one object. Read-only: no upstream call can block it for long."""
    now = time.time()
    with cv:
        q = {"running": running and job_view(running, now), "waiting": [job_view(t, now) for t in waiting]}
        since = dict(stats)
    return {"uptime": _uptime(), "db": bool(RUN_DB), "queue": q, "backends": backend_cards(),
            "day": day_summary(), "recent": recent_jobs(), "since_start": since}


def _uptime():
    s = int(time.time() - STARTED)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, f, *a): pass

    def _fwd_headers(self):
        return {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "connection", "x-priority", "content-length", "transfer-encoding")}

    def _send_json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _pipe(self, r):
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            d = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not d: break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def _chunk(self, b):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b)); self.wfile.flush()

    def _pipe_llama(self, r, model, kind, t_run):
        """OpenAI SSE stream -> Ollama NDJSON stream. Returns (timing fields, finish reason) from the last
        chunk, where llama.cpp puts its timings."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        last, finish = {}, None
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"): continue
            try: d = json.loads(line[5:].strip())
            except Exception: continue
            if d.get("timings") or d.get("usage"): last = d
            ch = (d.get("choices") or [{}])[0]
            finish = ch.get("finish_reason") or finish
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece:
                self._chunk((json.dumps(ollama_reply(kind, model, piece, False)) + "\n").encode())
        f = timing_fields(last)
        self._chunk((json.dumps(ollama_reply(kind, model, "", True, f, finish, time.time() - t_run)) + "\n").encode())
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        return f, finish

    def _embed(self, body):
        """Ollama /api/embed -> llama.cpp /v1/embeddings -> Ollama shape. Runs outside the GPU lane:
        the embedding server is its own process, so it does not queue behind a long chat job."""
        try:
            d = json.loads(body or b"{}")
        except Exception:
            return self._send_json(400, {"error": "gateway: unparsable embed body"})
        model = d.get("model", "")
        target = EMBED_ROUTES.get(model)
        if target is None:
            return self._send_json(404, {"error": f"gateway: no embedding route for {model!r}"})
        out = json.dumps({"model": target, "input": d.get("input")}).encode()
        t0 = time.time()
        try:
            c = http.client.HTTPConnection(EMBED.hostname, EMBED.port or 80, timeout=120)
            c.request("POST", "/v1/embeddings", out, {"Content-Type": "application/json"})
            r = c.getresponse()
            raw = r.read()
            c.close()
        except Exception as e:
            record({"ts": t0, "model": model, "backend": "llama-embed", "path": "/api/embed", "prio": "batch",
                    "wait_s": 0, "run_s": round(time.time() - t0, 2), "result": "failed", "retries": 0,
                    "note": repr(e)[:200]})
            return self._send_json(502, {"error": f"gateway: embed upstream failed: {e}"})
        try:
            j = json.loads(raw or b"{}")
            vecs = [row.get("embedding") for row in j.get("data") or []]
            if not vecs:
                raise ValueError(f"empty data: {raw[:120]!r}")
        except Exception as e:
            return self._send_json(502, {"error": f"gateway: bad embed reply: {e}"})
        record({"ts": t0, "model": model, "backend": "llama-embed", "path": "/api/embed", "prio": "batch",
                "wait_s": 0, "run_s": round(time.time() - t0, 2), "result": "ok", "retries": 0,
                "status": r.status, "tokens_in": j.get("usage", {}).get("prompt_tokens")})
        return self._send_json(200, {"model": model, "embeddings": vecs})

    def _models(self):
        """Model list for OpenAI-style clients, with ids renamed to the names clients ask for: the
        router calls its model qwen36-35b, everything else here knows it as qwen3.6:35b."""
        try:
            c, r = llama_upstream("GET", "/v1/models", None, {}, 10)
            d = json.loads(r.read() or b"{}"); c.close()
        except Exception as e:
            return self._send_json(502, {"error": f"gateway: model list unavailable: {e}"})
        back = {v: k for k, v in ROUTES.items()}
        for m in d.get("data") or []:
            m["id"] = back.get(m.get("id"), m.get("id"))
        return self._send_json(200, d)

    def _llama_models(self):
        return router_models()

    def _merged_tags(self):
        """Ollama tags (when there is an upstream) plus the llama.cpp models, under the names clients ask for."""
        d = {"models": []}
        if UP.hostname:
            c, r = upstream("GET", "/api/tags", None, {}, 30)
            d = json.loads(r.read() or b"{}"); c.close()
        names = {m.get("model") or m.get("name") for m in d.get("models") or []}
        for m in self._llama_models():
            name = next((k for k, v in ROUTES.items() if v == m.get("id")), m.get("id"))
            if name not in names:
                d.setdefault("models", []).append({"name": name, "model": name, "size": 0, "digest": "",
                    "details": {"format": "gguf", "family": "", "families": [],
                                "parameter_size": "", "quantization_level": ""}, "modified_at": ""})
        return d

    def _merged_ps(self):
        d = {"models": []}
        if UP.hostname:
            c, r = upstream("GET", "/api/ps", None, {}, 30)
            d = json.loads(r.read() or b"{}"); c.close()
        loaded = {m.get("name") for m in d.get("models") or []}
        for m in self._llama_models():
            name = next((k for k, v in ROUTES.items() if v == m.get("id")), m.get("id"))
            if (m.get("status") or {}).get("value") == "loaded" and name not in loaded:
                d.setdefault("models", []).append({"name": name, "model": name, "size": 0,
                                                   "size_vram": 0, "expires_at": ""})
        return d
    def handle_any(self):
        if self.path in ("/", "/_status"):
            b = STATUS_HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); return self.wfile.write(b)
        if self.path == "/_status.json":
            return self._send_json(200, status_json())
        if self.path == "/_ui":                             # the old second page; everything is on / now
            self.send_response(301); self.send_header("Location", "/"); self.send_header("Content-Length", "0")
            return self.end_headers()
        if self.path.startswith("/_unload") and self.command == "POST":
            if not UP.hostname:
                return self._send_json(404, {"error": "gateway: unload needs an Ollama upstream"})
            m = unquote(self.path.split("model=", 1)[-1])
            try:
                c, r = upstream("POST", "/api/generate", json.dumps({"model": m, "keep_alive": 0}).encode(), {"Content-Type": "application/json"}, 60)
                return self._send_json(r.status, {"unloaded": m})
            except Exception as e:
                return self._send_json(502, {"error": str(e)})
        if self.path == "/api/version":
            return self._send_json(200, {"version": OLLAMA_COMPAT})
        if self.path == "/api/show" and self.command == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                want = json.loads(raw or b"{}").get("model") or json.loads(raw or b"{}").get("name")
            except Exception:
                want = None
            target = ROUTES.get(want, want)
            m = next((m for m in router_models() if m.get("id") == target), None)
            if m is None:
                return self._send_json(404, {"error": f"model {want!r} not found"})
            return self._send_json(200, {"modelfile": "", "parameters": "", "template": "",
                                         "details": {"parent_model": "", "format": "gguf", "family": "",
                                                     "families": [], "parameter_size": "",
                                                     "quantization_level": ""},
                                         "model_info": {}, "capabilities": ["completion"]})
        if self.path == "/_queue":
            now = time.time()
            with cv:
                return self._send_json(200, {
                    "running": running and job_view(running, now), "waiting": [job_view(t, now) for t in waiting],
                    "last_model": last_model, "recent": list(history), **stats})
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        if self.path == "/api/embed" and EMBED.hostname and EMBED_ROUTES:
            return self._embed(body)
        hdr = self._fwd_headers()
        if body is not None: hdr["Content-Length"] = str(len(body))
        if self.path in ("/api/tags", "/api/ps") and ROUTES:
            try:
                return self._send_json(200, self._merged_tags() if self.path == "/api/tags" else self._merged_ps())
            except Exception as e:
                return self._send_json(502, {"error": f"gateway merge: {e}"})
        if self.path in ("/models", "/v1/models") and LLAMA.hostname:
            return self._models()
        if not self.path.startswith(GPU):
            if not UP.hostname:
                return self._send_json(503, {"error": "gateway: no Ollama upstream configured"})
            try:
                c, r = upstream(self.command, self.path, body, hdr, 600); return self._pipe(r)
            except Exception as e:
                return self._send_json(502, {"error": f"gateway upstream: {e}"})
        try: req = json.loads(body or b"{}")
        except Exception: req = {}
        if not isinstance(req, dict): req = {}
        tk = Ticket(req.get("model", "?"), (self.headers.get("X-Priority") or "interactive").lower(), self.path)
        got = acquire(tk, self.connection)
        if got == "gone": return
        if got == "timeout": return self._send_json(503, {"error": "gateway: queue wait timed out"})
        t_run, m, result, resp_text = time.time(), tk.m, "ok", None
        kind = {"/api/chat": "chat", "/api/generate": "generate"}.get(self.path)
        stream = wants_stream(self.path, req)
        if tk.llama:
            st = llama_status(tk.llama)
            m["cold"] = int(st is not None and st != "loaded")
            if m["cold"]: tk.state = "loading"
        try:
            if tk.llama and kind:
                result, resp_text = self._llama_ollama(tk, kind, req, stream, t_run)
            else:
                result = self._passthrough(tk, body, hdr)
        except OSError:                                      # client went away mid-response: abort upstream, do NOT retry
            result = "client gone"
            m["note"] = m.get("note") or "client disconnected during the response"
        except Exception as e:                               # a gateway bug: record it rather than call the job ok
            result = "failed"
            m["note"] = f"gateway error: {e!r}"[:300]
        finally:
            try:
                self._finish(tk, t_run, result, body, resp_text, stream)
            finally:
                release(tk)                                  # the lane must reopen whatever bookkeeping did

    def _finish(self, tk, t_run, result, body, resp_text, stream):
        """Close out a job: load event, counters, run row, prompt text, in-memory history."""
        m = tk.m
        run_s = time.time() - t_run
        if m.get("cold"):
            load_s = run_s - ((m.get("prompt_ms") or 0) + (m.get("gen_ms") or 0)) / 1000
            event("load", model=tk.model, seconds=round(load_s, 1))
            m["note"] = m.get("note") or f"cold start: ~{load_s:.0f}s loading the model"
        stats["done" if result == "ok" else "aborted" if result == "client gone" else "failed"] += 1
        row = {"ts": tk.t, "model": tk.model, "backend": "llama" if tk.llama else "ollama", "path": tk.path,
               "prio": tk.prio, "wait_s": round(t_run - tk.t, 2), "run_s": round(run_s, 2), "result": result, **m}
        run_id = record(row)
        save_call(run_id, extract_prompt(body), resp_text, {"path": tk.path, "model": tk.model, "stream": stream})
        with cv:
            history.appendleft(row)

    def _llama_ollama(self, tk, kind, req, stream, t_run):
        """An Ollama /api/chat or /api/generate job served by llama.cpp. Returns (result, response text)."""
        m = tk.m
        if kind == "generate" and not req.get("prompt"):     # Ollama's "load this model" call: nothing to generate
            self._send_json(200, ollama_reply(kind, tk.model, "", True, finish="load"))
            return "ok", None
        oai = json.dumps(to_openai(req if kind == "chat" else generate_as_chat(req), stream)).encode()
        for attempt in range(RETRIES + 1):
            resp = self._call(tk, "/v1/chat/completions", oai, {"Content-Type": "application/json"}, llama_conn, None)
            if resp is None:
                return "client gone", None
            c, r, err = resp
            if r is None:
                m["note"] = err
                self._send_json(502, {"error": f"gateway: upstream failed after retries: {err}"})
                return "failed", None
            m["status"] = r.status
            try:
                if r.status != 200:                          # a 4xx from llama.cpp: the request itself is wrong
                    raw = r.read()
                    m["note"] = upstream_error(r.status, raw)
                    self._send_raw(r.status, raw)
                    return "failed", None
                if stream:
                    f, _ = self._pipe_llama(r, tk.model, kind, t_run)
                    m.update(f)
                    return "ok", None
                d, empty, note = read_llama_chat(r)
            finally:
                c.close()                                    # closing the upstream socket makes llama.cpp stop generating
            if not empty:
                f = timing_fields(d)
                m.update(f)
                ch = (d.get("choices") or [{}])[0]
                text = (ch.get("message") or {}).get("content") or ""
                self._send_json(200, ollama_reply(kind, tk.model, text, True, f, ch.get("finish_reason"), time.time() - t_run))
                return "ok", text
            m["note"] = note                                 # a 200 with nothing in it is not an answer
            if attempt < RETRIES:
                m["retries"] += 1; stats["retried"] += 1
                time.sleep(2)
        event("empty_reply", model=tk.model, detail=m["note"])
        self._send_json(502, {"error": f"gateway: empty reply from llama.cpp after {RETRIES + 1} attempts: {m['note']}"})
        return "empty", None

    def _passthrough(self, tk, body, hdr):
        """Any other GPU job: forwarded as is (to llama.cpp with the model id rewritten, or to Ollama). Returns the result."""
        m = tk.m
        if tk.llama and self.path.startswith("/v1/") and body:
            try:
                j = json.loads(body); j["model"] = tk.llama
                body = json.dumps(j).encode(); hdr["Content-Length"] = str(len(body))
            except Exception:
                pass
        resp = self._call(tk, self.path, body, hdr, llama_conn if tk.llama else ollama_conn, JOB_TIMEOUT)
        if resp is None:
            return "client gone"
        c, r, err = resp
        if r is None:
            m["note"] = err
            self._send_json(502, {"error": f"gateway: upstream failed after retries: {err}"})
            return "failed"
        m["status"] = r.status
        try:
            self._pipe(r)
        finally:
            c.close()
        return "ok"

    def _send_raw(self, code, raw):
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def _call(self, tk, path, body, hdr, factory, timeout=None):
        """Send the request, retrying 5xx/connection errors while the upstream recovers. Returns (conn, resp, None),
        (None, None, err) once retries are spent, or None if the client disconnected while we waited."""
        timeout = timeout or JOB_TIMEOUT
        deadline = time.time() + timeout
        err = None
        attempt = 0
        while True:
            holder = {}
            def work():
                try:
                    c = factory(timeout); holder["c"] = c
                    c.request(self.command, path, body, hdr); holder["r"] = c.getresponse()
                except Exception as e:
                    holder["e"] = e
            th = threading.Thread(target=work, daemon=True); th.start()
            probed = time.time()
            while th.is_alive():                              # non-streaming calls only answer when finished: watch the client meanwhile
                th.join(0.5)
                if th.is_alive() and client_gone(self.connection):
                    if holder.get("c"): holder["c"].close()
                    return None
                if tk.state == "loading" and tk.llama and time.time() - probed > 2:
                    probed = time.time()
                    if llama_status(tk.llama) == "loaded":
                        tk.state = "running"
            r = holder.get("r")
            if r is not None and r.status < 500:
                tk.state = "running"
                return holder["c"], r, None
            if r is not None:
                body_txt = r.read()
                holder["c"].close()
                err = upstream_error(r.status, body_txt)
                # A loading model is not a failure: llama.cpp answers 503 while it reads a large GGUF, which can
                # take minutes. Waiting does not spend a retry: the deadline bounds it, and the client can still leave.
                if b"Loading model" in body_txt and time.time() < deadline:
                    tk.state, tk.m["cold"] = "loading", 1
                    time.sleep(LOAD_POLL)
                    if client_gone(self.connection):
                        return None
                    continue
            else:
                err = repr(holder.get("e"))
            if attempt == RETRIES:
                return None, None, err
            attempt += 1
            tk.m["retries"] += 1; stats["retried"] += 1
            if factory is ollama_conn:
                wait_alive()
            else:
                time.sleep(2)

    do_GET = do_POST = do_DELETE = do_HEAD = do_PUT = handle_any


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    print(f"gateway :{PORT} -> {UP.geturl()}", flush=True)
    event("start", upstream=UP.geturl(), llama=LLAMA.geturl() or None, routes=ROUTES,
          retries=RETRIES, affinity_max_wait=AFFINITY_MAX_WAIT, run_db=RUN_DB or None)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
