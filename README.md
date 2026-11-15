# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/dekakironz/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/dekakironz/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, both client wires, and a real free OpenRouter model behind every answer. For the make.fun "The best live free-model aggregator" bounty.

- Serves `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic), JSON and streaming (SSE).
- Answers `GET /v1/models` in the shape its auth header names.
- Dials the strongest free OpenRouter model that meets a prompt's requirements (quality-rated routing), fails over to the next on a refusal, and reports the failover trail.
- Registers with the env and polls `/work` until done, then exits 0.

## Run it

```sh
node main.ts --list
```

Registers the free model catalog and exits before starting the server.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>