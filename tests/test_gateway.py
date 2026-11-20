# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gateway tests against fake upstreams. Run from the repo root: python3 -W ignore -m unittest discover -s tests -v"""
import http.client, json, os, socket, sqlite3, subprocess, sys, tempfile, threading, time, unittest, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = threading.Lock()
EXEC = []                      # tags in the order the fake upstream started them
FAIL_ONCE = {"crash": True}


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/api/version":
            return self._json(200, {"version": "fake"})
        if self.path == "/api/ps":
            return self._json(200, {"models": [{"name": "fake:1b", "size_vram": 1_500_000_000}]})
        self._json(200, {"models": []})
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tag = d.get("tag", d["model"])
        if tag == "crash" and FAIL_ONCE.pop("crash", False):
            return self._json(500, {"error": "boom"})
        with LOCK: EXEC.append(tag)
        n, gap = d.get("chunks", 2), d.get("gap", 0.05)
        self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        try:
            for i in range(n):
                p = b'{"p":%d}\n' % i
                self.wfile.write(b"%x\r\n%s\r\n" % (len(p), p)); self.wfile.flush(); time.sleep(gap)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class GatewayCase(unittest.TestCase):
    ENV = {}
    @classmethod
    def setUpClass(cls):
        cls.up_port, cls.gw_port = free_port(), free_port()
        cls.up = ThreadingHTTPServer(("127.0.0.1", cls.up_port), Fake); cls.up.daemon_threads = True
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.gw = subprocess.Popen([sys.executable, os.path.join(ROOT, "gateway", "gateway.py")],
                                  env={**os.environ, "UPSTREAM": f"http://127.0.0.1:{cls.up_port}", "PORT": str(cls.gw_port), "RECOVER_WAIT": "10", **cls.ENV},
                                  stdout=subprocess.DEVNULL)
        for _ in range(50):
            try: urllib.request.urlopen(f"http://127.0.0.1:{cls.gw_port}/_queue", timeout=1); break
            except Exception: time.sleep(0.1)
    @classmethod
    def tearDownClass(cls):
        cls.gw.terminate(); cls.gw.wait(); cls.up.shutdown()
    def setUp(self):
        with LOCK: EXEC.clear()
        self.wait_idle()
    def url(self, p): return f"http://127.0.0.1:{self.gw_port}{p}"
    def wait_idle(self, t=20):
        end = time.time() + t
        while time.time() < end:
            q = json.load(urllib.request.urlopen(self.url("/_queue")))
            if not q["running"] and not q["waiting"]: return
            time.sleep(0.1)
        self.fail("gateway never went idle")
    def post(self, model, prio="interactive", **extra):
        body = json.dumps({"model": model, **extra}).encode()
        r = urllib.request.Request(self.url("/api/chat"), body, {"X-Priority": prio})
        return urllib.request.urlopen(r, timeout=30).read()
    def bg(self, *a, **k):
        t = threading.Thread(target=self.post, args=a, kwargs=k); t.start(); return t
    def raw_post(self, model, **extra):
        s = socket.create_connection(("127.0.0.1", self.gw_port))
        body = json.dumps({"model": model, **extra}).encode()
        s.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        return s


class Ordering(GatewayCase):
    ENV = {"AFFINITY_MAX_WAIT": "1"}
    def test_interactive_first_and_same_model_grouped(self):
        ts = [self.bg("A", "batch", tag="A-first", chunks=8)]; time.sleep(0.15)
        for m, p, tag in [("B", "batch", "B-batch"), ("A", "batch", "A-batch"), ("C", "interactive", "C-chat"), ("A", "batch", "A-batch2")]:
            ts.append(self.bg(m, p, tag=tag, chunks=2)); time.sleep(0.05)
        [t.join() for t in ts]
        self.assertEqual(EXEC[:2], ["A-first", "C-chat"])
        self.assertEqual(EXEC[-2:], ["A-batch", "A-batch2"])      # grouped with the loaded model A

    def test_streaming_passthrough_and_crash_retry(self):
        self.assertEqual(self.post("S", tag="s", chunks=3).decode().count('"p"'), 3)
        self.assertIn(b'"p"', self.post("X", tag="crash"))

    def test_model_affinity_is_bounded(self):
        """A same-priority job for another model must not wait behind an unbounded stream of loaded-model jobs."""
        ts = [self.bg("A", tag="A0", chunks=20, gap=0.1)]; time.sleep(0.2)      # holds the lane ~2 s
        ts.append(self.bg("B", tag="B1", chunks=2)); time.sleep(0.05)
        for i in range(3): ts.append(self.bg("A", tag=f"A{i+1}", chunks=2)); time.sleep(0.05)
        [t.join() for t in ts]
        self.assertLess(EXEC.index("B1"), EXEC.index("A1"), EXEC)    # B waited > AFFINITY_MAX_WAIT, so it outranks affinity


class Disconnect(GatewayCase):
    def test_disconnect_while_streaming_does_not_rerun(self):
        s = self.raw_post("A", tag="dc", chunks=10, gap=0.2)
        s.recv(200); s.close()                      # got the first bytes, then the client goes away
        time.sleep(4)
        self.assertEqual(EXEC.count("dc"), 1, EXEC)

    def test_disconnect_while_queued_drops_the_job(self):
        blocker = self.bg("A", tag="block", chunks=10, gap=0.2); time.sleep(0.3)
        s = self.raw_post("A", tag="ghost", chunks=2); time.sleep(0.3); s.close()
        blocker.join(); time.sleep(1.5)
        self.assertNotIn("ghost", EXEC, EXEC)


class Passthrough(GatewayCase):
    def test_v1_models_and_tags_are_not_queued(self):
        blocker = self.bg("A", tag="block", chunks=12, gap=0.2); time.sleep(0.3)
        for p in ("/v1/models", "/api/tags"):
            t = time.time(); urllib.request.urlopen(self.url(p), timeout=5).read()
            self.assertLess(time.time() - t, 1.0, p)
        blocker.join()


if __name__ == "__main__":
    unittest.main()


# ---------- llama.cpp backend: the gateway must keep the Ollama API in front of it ----------
SEEN = {}


LOADING = {"left": 0}


class FakeLlama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        data = [{"id": "qwen36-35b", "status": {"value": "loaded"}}, {"id": "qwen35-4b", "status": {"value": "unloaded"}}]
        b = json.dumps({"data": data}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        SEEN["body"] = body
        SEEN["path"] = self.path
        if self.path.endswith("/embeddings"):
            return self._json(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}], "usage": {"prompt_tokens": 4}})
        if (body.get("messages") or [{}])[0].get("content") == "loading" and LOADING["left"] > 0:
            LOADING["left"] -= 1                    # llama-server's answer while it reads a GGUF from disk
            return self._json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        first = (body.get("messages") or [{}])[-1].get("content")
        if first == "boom":
            return self._json(500, {"error": {"code": 500, "message": "boom"}})
        if first == "bad":
            return self._json(400, {"error": {"code": 400, "message": "bad request"}})
        if (body.get("messages") or [{}])[0].get("content") == "empty":
            return self._json(200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
        if not body.get("stream"):
            out = {"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                   "timings": {"cache_n": 4, "prompt_n": 7, "prompt_ms": 100.0, "prompt_per_second": 70.0,
                               "predicted_n": 7, "predicted_ms": 700.0, "predicted_per_second": 10.0}}
            return self._json(200, out)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        for piece in ("he", "llo"):
            ev = b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev)); self.wfile.flush()
        end = (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":3,"prompt_n":2,'
               b'"prompt_ms":40.0,"prompt_per_second":50.0,"predicted_n":2,"predicted_ms":80.0,"predicted_per_second":25.0}}\n\n'
               b'data: [DONE]\n\n')
        self.wfile.write(b"%x\r\n%s\r\n" % (len(end), end))
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


class LlamaRouteCase(unittest.TestCase):
    """Clients speak Ollama; the routed models are served by a llama.cpp OpenAI server."""
    ROUTES = "qwen3.6:35b=qwen36-35b,qwen3.5:4b=qwen35-4b"

    @classmethod
    def setUpClass(cls):
        cls.ol_port, cls.ll_port, cls.gw_port = free_port(), free_port(), free_port()
        cls.ol = ThreadingHTTPServer(("127.0.0.1", cls.ol_port), Fake); cls.ol.daemon_threads = True
        cls.ll = ThreadingHTTPServer(("127.0.0.1", cls.ll_port), FakeLlama); cls.ll.daemon_threads = True
        for srv in (cls.ol, cls.ll):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        cls.run_db = os.path.join(tempfile.mkdtemp(), "runs.db")
        env = {**os.environ, "UPSTREAM": f"http://127.0.0.1:{cls.ol_port}", "RUN_DB": cls.run_db,
               "EMBED_UPSTREAM": f"http://127.0.0.1:{cls.ll_port}", "EMBED_ROUTES": "embeddinggemma=embeddinggemma",
               "LLAMA_UPSTREAM": f"http://127.0.0.1:{cls.ll_port}", "LLAMA_ROUTES": cls.ROUTES,
               "PORT": str(cls.gw_port), "RECOVER_WAIT": "5", "LOAD_POLL": "0.2"}
        cls.gw = subprocess.Popen([sys.executable, os.path.join(ROOT, "gateway", "gateway.py")],
                                  env=env, stdout=subprocess.DEVNULL)
        for _ in range(50):
            try: urllib.request.urlopen(f"http://127.0.0.1:{cls.gw_port}/_queue", timeout=1); break
            except Exception: time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.gw.terminate(); cls.gw.wait()
        cls.ol.shutdown(); cls.ll.shutdown()

    def chat(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.gw_port}/api/chat",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def test_a_loading_model_is_waited_out_not_counted_as_a_failure(self):
        """More 'Loading model' answers than RETRIES allows must still end in the real reply."""
        LOADING["left"] = 6
        out = json.loads(self.chat({"model": "qwen3.6:35b", "stream": False,
                                    "messages": [{"role": "user", "content": "loading"}]}))
        self.assertEqual(LOADING["left"], 0)
        self.assertEqual(out["message"]["content"], '{"ok": true}')

    def test_routed_chat_is_translated(self):
        out = self.chat({"model": "qwen3.6:35b", "stream": False, "think": False,
                         "format": {"type": "object", "properties": {"a": {"type": "integer"}}},
                         "messages": [{"role": "user", "content": "hi"}],
                         "options": {"num_predict": 64, "temperature": 0, "num_ctx": 4096}})
        out = json.loads(out)
        sent = SEEN["body"]
        self.assertEqual(SEEN["path"], "/v1/chat/completions")
        self.assertEqual(sent["model"], "qwen36-35b")
        self.assertEqual(sent["max_tokens"], 64)
        self.assertEqual(sent["temperature"], 0)
        self.assertEqual(sent["response_format"]["json_schema"]["schema"]["type"], "object")
        self.assertIs(sent["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("num_ctx", sent)
        self.assertEqual(out["message"]["content"], '{"ok": true}')
        self.assertEqual(out["eval_count"], 7)
        self.assertEqual(out["prompt_eval_count"], 11)
        self.assertTrue(out["done"])

    def test_unrouted_model_still_goes_to_ollama(self):
        out = self.chat({"model": "qwen3:8b", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn('{"p":0}', out)      # only the fake Ollama streams this shape

    def test_streaming_routed_chat_becomes_ndjson(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.gw_port}/api/chat",
                                     data=json.dumps({"model": "qwen3.5:4b", "stream": True,
                                                      "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
        self.assertEqual("".join(l["message"]["content"] for l in lines if not l["done"]), "hello")
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(lines[-1]["eval_count"], 2)
        self.assertEqual(lines[-1]["prompt_eval_count"], 5)
        self.assertEqual(lines[-1]["done_reason"], "stop")
        self.assertEqual(lines[-1]["eval_duration"], 80_000_000)

    def test_every_job_is_recorded_in_sqlite(self):
        self.chat({"model": "qwen3.6:35b", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        # the row is written after the response is sent; filter to a successful one because the
        # empty-reply test in this class also writes rows, asynchronously, to the same database
        # filter by backend as well: the embed test in this class writes llama-embed rows to the same database
        rows = self.wait_rows("SELECT model, backend, path, prio, status, result, tokens_in, tokens_out "
                              "FROM runs WHERE result = 'ok' AND backend = 'llama' AND path = '/api/chat' "
                              "ORDER BY id DESC LIMIT 1")
        self.assertTrue(rows, "no run row written")
        model, backend, path, prio, status, result, tin, tout = rows[-1]
        self.assertEqual((model, backend, path, result), ("qwen3.6:35b", "llama", "/api/chat", "ok"))
        self.assertEqual((status, tin, tout), (200, 11, 7))

    def wait_rows(self, sql, tries=40):
        rows = []
        for _ in range(tries):
            db = sqlite3.connect(self.run_db)
            try:
                rows = db.execute(sql).fetchall()
            except sqlite3.OperationalError:
                rows = []
            db.close()
            if rows:
                return rows
            time.sleep(0.1)
        return rows

    def test_prompt_and_response_text_are_kept(self):
        self.chat({"model": "qwen3.6:35b", "stream": False,
                   "messages": [{"role": "user", "content": "marker-alpha"}]})
        rows = self.wait_rows("SELECT prompt, response FROM calls WHERE prompt LIKE '%marker-alpha%'")
        self.assertTrue(rows, "no prompt stored")
        self.assertIn("user: marker-alpha", rows[0][0])
        self.assertEqual(rows[0][1], '{"ok": true}')

    def test_session_events_are_logged(self):
        self.chat({"model": "qwen3.6:35b", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        kinds = {r[0] for r in self.wait_rows("SELECT kind FROM events")}
        self.assertIn("start", kinds)

    def test_an_empty_upstream_reply_is_an_error_not_an_answer(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.chat({"model": "qwen3.6:35b", "stream": False,
                       "messages": [{"role": "user", "content": "empty"}]})
        self.assertEqual(ctx.exception.code, 502)
        self.assertIn("empty reply", ctx.exception.read().decode())
        rows = []
        for _ in range(30):
            db = sqlite3.connect(self.run_db)
            rows = db.execute("SELECT result, note FROM runs WHERE result='empty'").fetchall()
            db.close()
            if rows: break
            time.sleep(0.1)
        self.assertTrue(rows and rows[-1][1], "no empty run row with a note")

    def test_embed_is_translated_to_openai_shape(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.gw_port}/api/embed",
                                     data=json.dumps({"model": "embeddinggemma", "input": "hello"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        self.assertEqual(d["model"], "embeddinggemma")
        self.assertEqual(d["embeddings"], [[0.1, 0.2, 0.3]])

    def test_status_page_and_json(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.gw_port}/", timeout=10) as r:
            html = r.read().decode()
        self.assertIn("ollama gateway", html)
        self.assertIn("/_status.json", html)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.gw_port}/_status.json", timeout=10) as r:
            d = json.load(r)
        self.assertEqual(set(d) >= {"uptime", "queue", "backends", "day", "recent", "db", "since_start"}, True)
        names = [b["name"] for b in d["backends"]]
        self.assertEqual(names[:2], ["llama.cpp", "ollama"])
        self.assertIn("llama.cpp embeddings", names)
        self.assertTrue(all("ok" in b for b in d["backends"]))

    def row_for(self, marker, cols):
        rows = self.wait_rows(f"SELECT {cols} FROM runs r JOIN calls c ON c.run_id = r.id WHERE c.prompt LIKE '%{marker}%'")
        self.assertTrue(rows, f"no run row for {marker}")
        return rows[-1]

    def stream_lines(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.gw_port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]

    def test_llama_timings_split_cached_and_evaluated_prompt_tokens(self):
        self.chat({"model": "qwen3.6:35b", "stream": False, "messages": [{"role": "user", "content": "mk-timings"}]})
        row = self.row_for("mk-timings", "tokens_in, cache_n, prompt_ms, gen_ms, prompt_tps, gen_tps, retries, cold")
        self.assertEqual(row, (11, 4, 100.0, 700.0, 70.0, 10.0, 0, 0))

    def test_streamed_chat_records_llama_timings(self):
        self.stream_lines("/api/chat", {"model": "qwen3.6:35b", "messages": [{"role": "user", "content": "mk-stream"}]})
        row = self.row_for("mk-stream", "tokens_in, tokens_out, cache_n, gen_ms, result")
        self.assertEqual(row, (5, 2, 3, 80.0, "ok"))

    def test_generate_is_served_by_llama_in_ollama_shape(self):
        out = json.loads(self.post_json("/api/generate", {"model": "qwen3.6:35b", "stream": False,
                                                         "system": "be brief", "prompt": "mk-generate"}))
        self.assertEqual(out["response"], '{"ok": true}')
        self.assertNotIn("message", out)
        self.assertEqual(SEEN["body"]["messages"], [{"role": "system", "content": "be brief"},
                                                    {"role": "user", "content": "mk-generate"}])
        lines = self.stream_lines("/api/generate", {"model": "qwen3.6:35b", "prompt": "hi"})   # Ollama streams by default
        self.assertEqual("".join(l["response"] for l in lines), "hello")
        self.assertTrue(lines[-1]["done"])

    def post_json(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.gw_port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def test_a_failed_job_records_why(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.chat({"model": "qwen3.6:35b", "stream": False, "messages": [{"role": "user", "content": "boom"}]})
        self.assertEqual(ctx.exception.code, 502)
        result, note, retries = self.row_for("boom", "result, note, retries")
        self.assertEqual((result, retries), ("failed", 2))
        self.assertIn("upstream 500", note)

    def test_a_llama_4xx_reaches_the_client_and_is_recorded(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.chat({"model": "qwen3.6:35b", "stream": False, "messages": [{"role": "user", "content": "bad"}]})
        self.assertEqual(ctx.exception.code, 400)
        self.assertIn("bad request", ctx.exception.read().decode())
        result, note = self.row_for("bad", "result, note")
        self.assertEqual(result, "failed")
        self.assertIn("upstream 400", note)

    def test_a_job_for_an_unloaded_model_is_a_cold_load(self):
        self.chat({"model": "qwen3.5:4b", "stream": False, "messages": [{"role": "user", "content": "mk-cold"}]})
        cold, note = self.row_for("mk-cold", "cold, note")
        self.assertEqual(cold, 1)
        self.assertIn("cold start", note)
        self.assertIn("load", {r[0] for r in self.wait_rows("SELECT kind FROM events")})
        with urllib.request.urlopen(f"http://127.0.0.1:{self.gw_port}/_status.json", timeout=10) as r:
            day = json.load(r)["day"]
        self.assertGreaterEqual(day["totals"]["cold"], 1)

    def test_queue_shows_a_loading_model(self):
        LOADING["left"] = 15                                 # ~3 s of "Loading model" at LOAD_POLL=0.2
        t = threading.Thread(target=self.chat, args=({"model": "qwen3.6:35b", "stream": False,
                                                      "messages": [{"role": "user", "content": "loading"}]},))
        t.start()
        seen = set()
        for _ in range(40):
            with urllib.request.urlopen(f"http://127.0.0.1:{self.gw_port}/_status.json", timeout=10) as r:
                run = json.load(r)["queue"]["running"]
            if run: seen.add(run["state"])
            if "loading" in seen: break
            time.sleep(0.1)
        t.join()
        self.assertIn("loading", seen)

    def test_a_malformed_request_does_not_lock_the_gpu_lane(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.chat({"model": "qwen3.6:35b", "stream": False, "messages": "bad"})
        out = json.loads(self.chat({"model": "qwen3.6:35b", "stream": False,
                                    "messages": [{"role": "user", "content": "after"}]}))
        self.assertEqual(out["message"]["content"], '{"ok": true}')

    def test_old_ui_path_redirects_to_the_status_page(self):
        c = http.client.HTTPConnection("127.0.0.1", self.gw_port, timeout=10)
        c.request("GET", "/_ui")
        r = c.getresponse()
        self.assertEqual((r.status, r.getheader("Location")), (301, "/"))

    def test_tags_list_routed_models_under_ollama_names(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.gw_port}/api/tags", timeout=10) as r:
            names = {m["name"] for m in json.load(r)["models"]}
        self.assertIn("qwen3.6:35b", names)
        self.assertIn("qwen3.5:4b", names)
