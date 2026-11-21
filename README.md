<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.svg">
    <img src="docs/banner.svg" alt="A border collie herding three llamas in single file" width="100%">
  </picture>
</p>

# ollama-gateway

A small queue and translator that sits between your apps and your GPU. Apps keep talking to it in
the Ollama API. Behind it, the models can run on [llama.cpp]'s `llama-server`, on [Ollama], or on
both.

It is one Python file with no dependencies, and it was built for a single home server with a single
GPU.

[Ollama]: https://github.com/ollama/ollama
[llama.cpp]: https://github.com/ggml-org/llama.cpp

## Why it exists

The setup is a familiar homelab one: a NAS with one consumer GPU and a handful of apps that want a
model. A feed reader scores articles in the background all day, hundreds of calls. Open WebUI is
there for chatting. Something else wants embeddings. They all speak Ollama, because almost
everything self-hosted does.

Ollama made the first steps easy, and then a few things started to hurt:

- **Your chat waits behind the batch backlog.** Ollama serves requests in arrival order. If a
  background job has queued forty calls, your question in Open WebUI is number forty-one.
- **Models swap without you seeing it.** Two apps asking for two models take turns evicting each
  other, and every swap reloads gigabytes from disk.
- **The model you want doesn't fit, and Ollama won't let you fix that.** A large mixture-of-experts
  model can run on a modest card if the expert weights stay in system RAM. llama.cpp does
  this with `--n-cpu-moe`. Ollama doesn't expose that flag, or most of llama.cpp's newer options,
  and its bundled llama.cpp tends to trail upstream.
- **You can't see what happened.** The base URL says "Ollama is running". It doesn't tell you how
  long the queue was, which call failed or why, or how fast prompts are actually being processed.

Moving the models to `llama-server` solved the third problem. But every client still spoke the
Ollama API, and rewriting them was out of the question. So this gateway keeps the Ollama API at the
front and translates to llama.cpp at the back. While it was in the middle anyway, it took on the
queue and the record keeping too.

## What it does

- **One GPU job at a time, in a sensible order.** Interactive requests go before batch ones, and
  jobs for the model that's already loaded go before ones that would cause a swap. Waiting time
  wins in the end, so nothing starves.
- **Ollama API in, llama.cpp out.** `/api/chat`, `/api/generate` and `/api/embed` are translated in
  both directions, including streaming, JSON-schema output and `think: false`. Models you don't route
  pass through to Ollama unchanged.
- **The Ollama surface keeps working without Ollama.** `GET /models` and `GET /v1/models` list the
  routed llama.cpp models under the Ollama names clients ask for, `/api/version` reports
  `OLLAMA_COMPAT`, and `/api/show` answers for a routed model.
- **It understands llama.cpp's model loading.** In router mode, `llama-server` answers "Loading
  model" for up to a few minutes while it reads a large GGUF. The gateway waits that out instead of
  failing the call, and the queue shows the job as `loading model`.
- **It keeps a record.** With `RUN_DB` set, every job lands in a sqlite file: wait, run time,
  tokens, prompt and generation speed, how much of the prompt came from llama.cpp's cache, cold
  starts, and the upstream's own error message when something failed.
- **It has a status page.** `/` shows the backends, the queue, the last 24 hours per model and the
  recent jobs. It's plain HTML that refreshes itself.

```
 apps (Ollama API)                          ┌── llama-server (router)  models in LLAMA_ROUTES
 ─────────────────────▶  ollama-gateway  ───┼── llama-server           embeddings, EMBED_ROUTES
  X-Priority: batch      queue · translate  └── Ollama (optional)      everything else
                         · record
```

## Quick start

The shortest path is a llama.cpp router and the gateway in one compose file. Put your GGUF files in
`./models`, next to a `preset.ini`:

```yaml
services:
  gateway:
    build: https://github.com/samousli/ollama-gateway.git
    ports: ["11435:11435"]
    environment:
      LLAMA_UPSTREAM: http://llama:8080
      LLAMA_ROUTES: qwen3:30b=qwen3-30b-a3b   # Ollama name your apps use = llama.cpp model id
      RUN_DB: /data/runs.db
    volumes: ["./data:/data"]

  llama:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda   # pin a build tag, e.g. server-cuda-b6xxx
    command: ["--models-dir", "/models", "--models-preset", "/models/preset.ini",
              "--models-max", "1", "--host", "0.0.0.0", "--port", "8080"]
    volumes: ["./models:/models:ro"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
```

Then point your apps at `http://<host>:11435` where they had `:11434`, and open the same address in
a browser for the status page.

To try it without Docker, against an Ollama you already run:

```sh
UPSTREAM=http://localhost:11434 python3 gateway/gateway.py
curl localhost:11435/api/chat -d '{"model":"llama3.2","messages":[{"role":"user","content":"hi"}]}'
```

Python 3.11 or newer is enough. There's nothing to install.

## Setting up llama.cpp

This is the part that takes the most fiddling, so here is what worked, as a starting point rather
than a recipe.

**The image.** The llama.cpp project publishes images at `ghcr.io/ggml-org/llama.cpp`. The
`server-*` tags contain only `llama-server`. Pick the one for your hardware: `server-cuda`,
`server-rocm`, `server-vulkan`, `server-intel`, or plain `server` for CPU. Then pin a build number,
because flags and endpoints do change between builds. The gateway was last tested against b12429.

**Router mode.** Started with `--models-dir`, one `llama-server` serves every GGUF in that folder,
loads a model on its first request, and with `--models-max 1` unloads the previous one to make room.
That's the behaviour Ollama users expect. `GET /models` on the router shows each model's state and
the exact flags it was started with, which is handy when a setting doesn't seem to take.

**Presets.** `preset.ini` holds the flags, either for every model or for one. A section name is the
GGUF's file name without `.gguf`, and that's also the model id you put in `LLAMA_ROUTES`. Any
`llama-server` flag works, written without the leading dashes:

```ini
[*]
ctx-size = 16384
n-gpu-layers = 99
jinja = 1
reasoning-budget = 0

[qwen3-30b-a3b]
n-cpu-moe = 30
```

A few flags matter more than the rest:

| flag | what it's for |
|---|---|
| `ctx-size` | Context per request. It costs VRAM, so set what your apps actually send, not the model's maximum. Ollama's `num_ctx` can't change it per request; the gateway drops that option. |
| `n-gpu-layers` | `99` means everything on the GPU. Lower it only if nothing else fits. |
| `n-cpu-moe` | For mixture-of-experts models: keeps the expert weights of the first N layers in system RAM. This is what lets a large MoE model run on a card far smaller than the model. Raise it until the model loads with some VRAM to spare, then stop. |
| `jinja` | Use the chat template embedded in the GGUF. Needed for tool calls and most recent models. |
| `reasoning-budget` | `0` turns off thinking by default, for models that support it. |
| `cache-reuse` | Lets llama.cpp reuse cached prompt chunks even when the start of the prompt changed. Worth trying if your app sends long prompts that share most of their text. |

**How to tell whether a change helped.** That's what the run data is for. After a day of traffic,
the status page shows prompt and generation speed per model, the share of prompt tokens served from
cache, and how many jobs had to wait for a cold load. With `n-cpu-moe`, expect generation speed to
hold up well and prompt processing to suffer, since the CPU now does part of that work. If your app
sends long prompts and short answers, prompt speed is the number to watch.

**Already have models in Ollama?** Ollama stores them as plain GGUF files under `blobs/`, named by
hash. `ollama show --modelfile <name>` prints which blob a model uses. You can symlink that blob into
the models folder under a readable name instead of downloading it again. Mount the blob store into
the llama container as well, or the link will point at nothing.

**Embeddings.** Run a second `llama-server` with `--embedding` and the embedding model. Set
`EMBED_UPSTREAM` and `EMBED_ROUTES` on the gateway. Embedding calls skip the GPU queue, because they
are quick and run in their own process. A small embedding model next to a big chat model is usually
worth the few hundred MB of VRAM: it's the difference between hundredths of a second and several
seconds per call on CPU.

## Adapting it to your setup

The whole gateway is `gateway/gateway.py`, about 900 lines. It was written to be read and changed,
by you or by a coding agent you point at it. Some places you're likely to touch:

- **Queue order** is `Ticket.rank()`. It returns a tuple, and the smallest one runs next. Adding a
  third priority level, or a per-app quota, starts there.
- **Request translation** is `to_openai()`, which maps Ollama options onto llama.cpp's request
  fields. `ollama_reply()` builds the response in Ollama's shape. A new option is usually a
  one-line change in `to_openai()`.
- **Which paths use the GPU lane**, and how each one is dispatched, is in `H.handle_any()`.
- **Retries and model loading** live in `H._call()`.
- **The status page** is the `STATUS_HTML` string plus `status_json()`. The page only reads
  `/_status.json`, so you can replace it entirely, or ignore it and read the JSON from your own
  dashboard.
- **Recorded columns** are the `COLS` list. New columns are added to an existing database when the
  gateway starts.

The tests in `tests/test_gateway.py` start the real gateway against fake Ollama and llama.cpp
servers, so a change can be checked end to end in about thirty seconds. Asking an agent to "add a
test that fails first" works well with this suite.

## Reference

### Queue order

Only inference endpoints take the GPU lane: `/api/chat`, `/api/generate`, `/api/embed`,
`/api/embeddings` and the `/v1/` chat, completion, embedding and responses endpoints. Everything else
passes straight through. The first rule that tells two jobs apart decides:

1. Interactive before batch. Clients send `X-Priority: batch`; the default is `interactive`. A batch
   job that has waited `BATCH_MAX_WAIT` seconds counts as interactive.
2. A job that has waited `AFFINITY_MAX_WAIT` seconds goes ahead of those that haven't, oldest first.
3. A job for the model that ran last goes first.
4. Otherwise, first in, first out.

### Failures

- An upstream 5xx or dropped connection is retried up to `RETRIES` times, if nothing has reached the
  client yet. For Ollama, the gateway first waits for `/api/version` to answer again.
- llama.cpp's "Loading model" answer is waited out every `LOAD_POLL` seconds and doesn't count as a
  retry. The job deadline still applies.
- A llama.cpp 4xx goes back to the client as it is, with llama.cpp's message.
- An empty llama.cpp reply is retried and then answered with a 502 that says what the upstream
  reported.
- A client that leaves while queued is dropped from the queue. A client that leaves mid-job closes
  the upstream connection, which stops generation.
- Whatever a request contains, the lane is released when the job ends.

### Translation to llama.cpp

| Ollama | llama.cpp |
|---|---|
| `options.num_predict` | `max_tokens` |
| `options.temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `repeat_penalty`, `presence_penalty`, `frequency_penalty` | same names |
| `options.num_ctx` | dropped: set `ctx-size` in the preset |
| `format: {…schema…}` | `response_format: {type: json_schema, …}` |
| `format: "json"` | `response_format: {type: json_object}` |
| `think: false` | `chat_template_kwargs.enable_thinking: false` |
| `/api/generate` `system` + `prompt` | a chat with those two messages |

Responses use Ollama's fields (`message.content` or `response`, `done_reason`, `eval_count`,
`prompt_eval_count`, the durations), filled from llama.cpp's own timings. As with Ollama, `/api/*`
streams unless the request says `"stream": false`. An empty `/api/generate` prompt, which Ollama
clients send to preload a model, is answered without generating anything. `/api/tags` and `/api/ps`
merge both backends. `/v1/*` requests for a routed model are forwarded with the model id rewritten,
and `GET /models` and `GET /v1/models` list the routed models under the names clients ask for.

Without an Ollama upstream, tags and ps list only llama.cpp models, version reports `OLLAMA_COMPAT`,
and show returns a minimal entry for a routed model; other Ollama paths answer 503.

### Configuration

| variable | default | meaning |
|---|---|---|
| `UPSTREAM` | empty | Ollama base URL; empty means no Ollama |
| `LLAMA_UPSTREAM` | empty | llama.cpp server base URL; empty disables routing |
| `LLAMA_ROUTES` | empty | `ollama-name=llama-id` pairs, comma separated |
| `EMBED_UPSTREAM` | empty | llama.cpp server started with `--embedding` |
| `EMBED_ROUTES` | empty | `ollama-name=llama-id` pairs for `/api/embed` |
| `PORT` | `11435` | listen port |
| `OLLAMA_COMPAT` | `0.12.0` | the version string `/api/version` reports |
| `RETRIES` | `2` | retries after an upstream 5xx or connection error |
| `RECOVER_WAIT` | `180` | seconds to wait for Ollama to come back before a retry |
| `LOAD_POLL` | `5` | seconds between checks while llama.cpp is loading a model |
| `BATCH_MAX_WAIT` | `600` | seconds before a batch job counts as interactive |
| `AFFINITY_MAX_WAIT` | `120` | seconds before waiting beats same-model grouping |
| `JOB_TIMEOUT` | `900` | upstream deadline per job, in seconds |
| `QUEUE_TIMEOUT` | `3600` | longest a job may wait in the queue before a 503 |
| `RUN_DB` | empty | sqlite file for run data; empty turns recording off |
| `RUN_PROMPTS` | `1` | store prompt and response text; `0` keeps only the numbers |
| `RUN_PROMPT_MAX` | `32768` | characters kept per text field |
| `RUN_KEEP_DAYS` | `90` | rows older than this are pruned |

### Endpoints of its own

| path | what |
|---|---|
| `GET /` | the status page |
| `GET /_status.json` | the same data as JSON |
| `GET /_queue` | queue state and counters |
| `POST /_unload?model=…` | ask Ollama to unload a model |
| `GET /models`, `GET /v1/models` | routed models under the names clients ask for |
| `GET /api/version` | `OLLAMA_COMPAT` |
| `POST /api/show` | a minimal model entry for a routed model |

### Run data

`RUN_DB` holds three tables. `runs` has one row per GPU job: `ts, model, backend, path, prio,
wait_s, run_s, status, result, retries, tokens_in, tokens_out, gen_tps, prompt_tps, note, cache_n,
prompt_ms, gen_ms, cold`. `calls` keeps the prompt and, for non-streamed calls, the response.
`events` is the gateway's own log: starts, model loads, swaps, empty replies and clients that gave
up while queued. For Ollama backends only wait and run time are recorded, since their responses pass
through untouched.

```sql
-- prompt speed and cache share per model and day
SELECT date(ts, 'unixepoch') d, model,
       round(sum(tokens_in - cache_n) * 1000.0 / sum(prompt_ms), 1) AS prompt_tps,
       round(100.0 * sum(cache_n) / sum(tokens_in)) AS cached_pct
FROM runs WHERE prompt_ms > 0 GROUP BY d, model;
```

Writing to the database never raises, so recording can't make a request fail.

## Security

There's no authentication, and Ollama and `llama-server` don't have any by default either. Keep the
gateway on localhost or your LAN, or put an authenticating reverse proxy in front of it.

With `RUN_DB` set, prompts and responses are stored in plain text. Set `RUN_PROMPTS=0` if that's not
something you want on disk.

## Other projects

If your setup looks different, one of these may suit it better:

| You want | Look at |
|---|---|
| OpenAI-API clients and llama.cpp model swapping | [llama-swap](https://github.com/mostlygeek/llama-swap) |
| Load balancing across several Ollama, LM Studio or llama.cpp hosts | [Olla](https://github.com/thushan/olla) |
| Auth and queueing over several Ollama servers | [ollama_proxy_server](https://github.com/ParisNeo/ollama_proxy_server) |
| Many providers, API keys, budgets | [LiteLLM proxy](https://github.com/BerriAI/litellm) |
| Just a concurrency limit on Ollama | `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_QUEUE` |

## Tests

```sh
python3 -W ignore -m unittest discover -s tests -v
```

No test dependencies. The suite runs the real gateway as a subprocess against fake upstreams.

## License

[AGPL-3.0-or-later](LICENSE).
