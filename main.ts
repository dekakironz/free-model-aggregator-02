/**
 * free-model-aggregator-02
 *
 * A single-file aggregator server. It reads a catalog of free OpenRouter
 * models plus a prompt manifest, registers itself with the Makefun env,
 * then serves OpenAI Chat Completions and Anthropic Messages wires,
 * routing every "auto" request to the best eligible free model and
 * dialling that model through the graded gateway.
 *
 * Node 22 stdlib only. No external dependencies.
 */

import * as http from "node:http";
import * as fs from "node:fs";
import * as crypto from "node:crypto";

/* ------------------------------------------------------------------ *
 * Configuration from the environment.
 * ------------------------------------------------------------------ */

const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE || "";
const ENV_URL = process.env.MAKEFUN_ENV_URL || "";
const GATEWAY_BASE = process.env.OPENAI_BASE_URL || "";
const GATEWAY_KEY = process.env.OPENAI_API_KEY || "";
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED || "0";

interface SealedModel {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface PromptSpec {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
}

/** A catalog in the shape the env wants: keyed by provider, only free ids. */
interface ProviderCatalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number }>;
  }>;
}

/* ------------------------------------------------------------------ *
 * Catalog loading.
 * ------------------------------------------------------------------ */

/** Locate each top-level JSON document in the prompt file by its schema
 *  marker, parse the object span it opens, and split the docs in two. */
function readSealedCatalog(): { models: SealedModel[]; prompts: PromptSpec[] } {
  const raw = fs.readFileSync(PROMPT_FILE, "utf8");
  const schemaRe = /"schema"\s*:\s*"([^"]+)"/g;
  const models: SealedModel[] = [];
  const prompts: PromptSpec[] = [];
  let m: RegExpExecArray | null;
  while ((m = schemaRe.exec(raw)) !== null) {
    const schema = m[1];
    const open = raw.lastIndexOf("{", m.index);
    const close = scanJsonEnd(raw, open);
    const parsed = JSON.parse(raw.slice(open, close + 1)) as Record<string, unknown>;
    if (schema.startsWith("free-model-aggregator-02/models/v1")) {
      models.push(...((parsed.models as SealedModel[]) || []));
    } else if (schema.startsWith("free-model-aggregator-02/prompts/v1")) {
      prompts.push(...((parsed.prompts as PromptSpec[]) || []));
    }
  }
  return { models, prompts };
}

/** Given the index of a `{` opening a JSON value, return the index of its
 *  matching closing `}`, giving weight to strings and nesting. */
function scanJsonEnd(text: string, open: number): number {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return text.length - 1;
}

function buildCatalog(models: SealedModel[]): ProviderCatalog {
  const free = models.filter((m) => m.free);
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: free.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.context_length,
        })),
      },
    ],
  };
}

/* ------------------------------------------------------------------ *
 * Model routing.
 * ------------------------------------------------------------------ */

interface Capability {
  tools: boolean;
  vision: boolean;
  json: boolean;
  min_context: number;
}

type Entry = SealedModel;

/** Derive capability needs from an incoming request body. */
function deriveCapabilities(body: Record<string, unknown>): Capability {
  const caps: Capability = { tools: false, vision: false, json: false, min_context: 0 };
  if (Array.isArray(body.tools) && (body.tools as unknown[]).length > 0) caps.tools = true;
  const rf = body.response_format as Record<string, unknown> | undefined;
  if (rf && rf.type === "json_object") caps.json = true;
  const messages = body.messages as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      const content = msg.content;
      if (typeof content === "string") continue;
      if (Array.isArray(content) && content.some((p) => p && p.type === "image_url")) {
        caps.vision = true;
      }
    }
  }
  return caps;
}

function eligible(models: SealedModel[], caps: Capability): SealedModel[] {
  return models.filter((m) => {
    if (!m.free) return false;
    if (caps.tools && !m.tools) return false;
    if (caps.vision && !m.vision) return false;
    if (caps.json && !m.json) return false;
    if (m.context_length < caps.min_context) return false;
    return true;
  });
}

/** Deterministic ranking: higher rating first, then lower id on ties. */
function rank(list: SealedModel[]): SealedModel[] {
  const copy = list.slice();
  copy.sort(
    (a, b) =>
      b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  );
  return copy;
}

interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

/* ------------------------------------------------------------------ *
 * Gateway client.
 * ------------------------------------------------------------------ */

interface UpstreamResult {
  ok: boolean;
  status: number;
  bodyText: string;
}

async function gatewayCall(
  modelId: string,
  payload: Record<string, unknown>,
): Promise<UpstreamResult & { json: Record<string, unknown> | null }> {
  const res = await fetch(`${GATEWAY_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      Authorization: `Bearer ${GATEWAY_KEY}`,
    },
    body: JSON.stringify({ ...payload, model: modelId }),
  });
  const text = await res.text();
  let json: Record<string, unknown> | null = null;
  try {
    json = JSON.parse(text) as Record<string, unknown>;
  } catch {
    /* non-JSON body */
  }
  return { ok: res.status < 400, status: res.status, bodyText: text, json };
}

/* ------------------------------------------------------------------ *
 * Anthropic <-> OpenAI translation.
 * ------------------------------------------------------------------ */

function anthropicToOpenAi(
  body: Record<string, unknown>,
): { payload: Record<string, unknown>; declaredTools: string[] } {
  const messages = Array.isArray(body.messages)
    ? (body.messages as Array<Record<string, unknown>>)
    : [];
  const out: Array<Record<string, unknown>> = [];
  const declaredTools: string[] = [];

  const txt = (v: unknown): string => {
    if (typeof v === "string") return v;
    if (Array.isArray(v)) {
      return (v as Array<Record<string, unknown>>)
        .filter((p) => p && p.type === "text" && typeof p.text === "string")
        .map((p) => p.text as string)
        .join("");
    }
    return "";
  };

  const systemText = txt(body.system);
  if (systemText) out.push({ role: "system", content: systemText });

  for (const msg of messages) {
    const role = msg.role as string;
    const content = msg.content;
    if (role === "assistant" && Array.isArray(content)) {
      const text = txt(content);
      const toolUses = (content as Array<Record<string, unknown>>).filter(
        (b) => b && b.type === "tool_use",
      );
      const oa: Record<string, unknown> = { role: "assistant", content: text || "" };
      if (toolUses.length > 0) {
        oa.tool_calls = toolUses.map((tu) => ({
          id: tu.id,
          type: "function",
          function: {
            name: tu.name,
            arguments: JSON.stringify(tu.input ?? {}),
          },
        }));
      }
      out.push(oa);
    } else if (role === "user") {
      if (typeof content === "string") {
        out.push({ role: "user", content });
      } else if (Array.isArray(content)) {
        const text = txt(content);
        const toolResult = (content as Array<Record<string, unknown>>).find(
          (b) => b && b.type === "tool_result",
        );
        if (toolResult) {
          out.push({
            role: "tool",
            tool_call_id: toolResult.tool_use_id,
            content: txt(toolResult.content),
          });
        } else if (text) {
          out.push({ role: "user", content: text });
        }
      }
    }
  }

  const tools = Array.isArray(body.tools) ? (body.tools as unknown[]) : [];
  for (const t of tools) {
    const tn = (t as Record<string, unknown>).name;
    if (typeof tn === "string") declaredTools.push(tn);
  }

  const payload: Record<string, unknown> = { model: body.model, messages: out };
  if (tools.length > 0) payload.tools = tools;
  if (typeof body.max_tokens === "number") payload.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") payload.temperature = body.temperature;
  if (typeof body.top_p === "number") payload.top_p = body.top_p;
  if (body.stream === true) payload.stream = true;
  if (Array.isArray(body.stop_sequences)) payload.stop = body.stop_sequences;
  return { payload, declaredTools };
}

function openAiToAnthropic(
  dialedId: string,
  oa: Record<string, unknown>,
): Record<string, unknown> {
  const choice = ((oa.choices as Array<Record<string, unknown>>) || [])[0] || {};
  const msg = (choice.message as Record<string, unknown>) || {};
  const raw = msg.content;
  const text =
    typeof raw === "string"
      ? raw
      : Array.isArray(raw)
        ? (raw as Array<Record<string, unknown>>)
            .map((p) => (typeof p.text === "string" ? p.text : ""))
            .join("")
        : "";
  const blocks: Array<Record<string, unknown>> = [];
  if (text) blocks.push({ type: "text", text });
  const toolCalls = msg.tool_calls as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      const fn = (tc.function as Record<string, unknown>) || {};
      blocks.push({
        type: "tool_use",
        id: tc.id,
        name: fn.name,
        input: jsonOr(fn.arguments, {}),
      });
    }
  }
  const usage = (oa.usage as Record<string, unknown>) || {};
  const finish = (choice.finish_reason as string) || null;
  let stop = finish === "tool_calls" ? "tool_use" : finish;
  if (!stop) stop = "end_turn";
  return {
    id: `msg_${hexId()}`,
    type: "message",
    role: "assistant",
    model: dialedId,
    content: blocks,
    stop_reason: stop,
    stop_sequence: null,
    usage: {
      input_tokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      output_tokens:
        typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
    },
  };
}

/* ------------------------------------------------------------------ *
 * SSE streaming translation (OpenAI gateway -> Anthropic wire).
 * ------------------------------------------------------------------ */

interface AnthState {
  started: boolean;
  blockIndex: number;
  toolCalls: Array<{ id: string; name: string; args: string }>;
}

function makeId(): string {
  return crypto.randomBytes(6).toString("hex");
}

function hexId(): string {
  return makeId();
}

function jsonOr(s: unknown, fallback: unknown): unknown {
  if (typeof s !== "string") return fallback;
  try {
    return JSON.parse(s);
  } catch {
    return fallback;
  }
}

/** Convert one OpenAI SSE data line into a batch of Anthropic SSE events. */
function openAiLineToAnthEvents(line: string, state: AnthState): string[] {
  const body = line.replace(/^data:\s*/, "");
  if (!body || body === "[DONE]") return [];
  let chunk: Record<string, unknown>;
  try {
    chunk = JSON.parse(body) as Record<string, unknown>;
  } catch {
    return [];
  }
  const choice = ((chunk.choices as Array<Record<string, unknown>>) || [])[0] || {};
  const delta = (choice.delta as Record<string, unknown>) || {};
  const finish = (choice.finish_reason as string) || null;
  const text = typeof delta.content === "string" ? delta.content : "";
  const out: string[] = [];

  if (!state.started) {
    state.started = true;
    state.blockIndex = 0;
    out.push("event: message_start");
    out.push(
      JSON.stringify({
        type: "message_start",
        message: {
          id: `msg_${makeId()}`,
          type: "message",
          role: "assistant",
          content: [],
          model: chunk.model ?? "",
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    );
    out.push("event: content_block_start");
    out.push(
      JSON.stringify({
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      }),
    );
  }

  if (text) {
    out.push("event: content_block_delta");
    out.push(
      JSON.stringify({
        type: "content_block_delta",
        index: state.blockIndex,
        delta: { type: "text_delta", text },
      }),
    );
  }

  const toolCalls = delta.tool_calls as
    | Array<Record<string, unknown>>
    | undefined;
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      const fn = (tc.function as Record<string, unknown>) || {};
      if (typeof tc.id === "string" && tc.id) {
        state.toolCalls.push({
          id: tc.id,
          name: typeof fn.name === "string" ? fn.name : "",
          args: typeof fn.arguments === "string" ? fn.arguments : "",
        });
      } else {
        // continuation chunk: merge arguments into the last known call
        const last = state.toolCalls[state.toolCalls.length - 1];
        if (last && typeof fn.arguments === "string") last.args += fn.arguments;
      }
    }
  }

  if (finish) {
    out.push("event: content_block_stop");
    out.push(JSON.stringify({ type: "content_block_stop", index: state.blockIndex }));
    for (const tc of state.toolCalls) {
      out.push("event: content_block_start");
      out.push(
        JSON.stringify({
          type: "content_block_start",
          index: state.blockIndex + 1,
          content_block: {
            type: "tool_use",
            id: tc.id,
            name: tc.name,
            input: jsonOr(tc.args, {}),
          },
        }),
      );
      out.push("event: content_block_stop");
      out.push(
        JSON.stringify({ type: "content_block_stop", index: state.blockIndex + 1 }),
      );
    }
    out.push("event: message_delta");
    out.push(
      JSON.stringify({
        type: "message_delta",
        delta: {
          stop_reason: finish === "tool_calls" ? "tool_use" : finish,
          stop_sequence: null,
        },
        usage: { output_tokens: 0 },
      }),
    );
    out.push("event: message_stop");
    out.push(JSON.stringify({ type: "message_stop" }));
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * SSE event-boundary helpers shared by both streaming relays.
 * ------------------------------------------------------------------ */

/** Find the start of the next blank-line boundary (`\n\n` or `\r\n\r\n`)
 *  that terminates an SSE event, or -1 if none is buffered yet. */
function findEventSep(buf: string): number {
  const idx = buf.indexOf("\n\n");
  if (idx >= 0) return idx;
  const crlf = buf.indexOf("\r\n\r\n");
  return crlf;
}

/** Length of the separator consumed at `sep` (2 for `\n\n`, 4 for
 *  `\r\n\r\n`). */
function sepLen(buf: string, sep: number): number {
  return buf.startsWith("\r\n\r\n", sep) ? 4 : 2;
}

/** Pull the value of the `data:` field(s) out of one SSE event block.
 *  Returns null when the block carries no data field. */
function extractDataPayload(event: string): string | null {
  const lines = event.split(/\r?\n/);
  const parts: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      parts.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (parts.length === 0) return null;
  return parts.join("\n");
}

/* ------------------------------------------------------------------ *
 * HTTP server.
 * ------------------------------------------------------------------ */

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function writeJson(res: http.ServerResponse, status: number, payload: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
}

class Aggregator {
  private models: SealedModel[];
  private catalog: ProviderCatalog;

  constructor(models: SealedModel[]) {
    this.models = models;
    this.catalog = buildCatalog(models);
  }

  /** GET /v1/models: OpenRouter shape for Bearer auth, Anthropic for x-api-key. */
  modelsResponse(headers: http.IncomingHttpHeaders): unknown {
    const ids = this.models
      .filter((m) => m.free)
      .map((m) => ({ id: m.id, display: m.display_name }));
    if (headers["x-api-key"]) {
      return {
        data: ids.map((r) => ({
          type: "model",
          id: r.id,
          display_name: r.display,
          created_at: "2026-01-01T00:00:00Z",
        })),
      };
    }
    return {
      object: "list",
      data: ids.map((r) => ({
        id: r.id,
        object: "model",
        created: 1767225600,
        owned_by: r.id.split("/")[1] || "openrouter",
      })),
    };
  }

  private refusalHeader(refusals: Refusal[]): Record<string, string> {
    if (refusals.length === 0) return {};
    return { "x-makefun-upstream-failures": JSON.stringify(refusals.slice(0, 8)) };
  }

  /** Full non-streaming path with deterministic failover across the ranked
   *  eligible pool. Returns the successful dial, or throws with a refusal. */
  private async dialWithFailover(
    targets: SealedModel[],
    payload: Record<string, unknown>,
  ): Promise<{ model: SealedModel; json: Record<string, unknown>; refusals: Refusal[] }> {
    const refusals: Refusal[] = [];
    for (const model of targets) {
      const r = await gatewayCall(model.id, payload);
      if (r.ok && r.json) {
        return { model, json: r.json, refusals };
      }
      const code = (() => {
        const e = r.json?.error as Record<string, unknown> | undefined;
        return e && typeof e.code === "string" ? e.code : "upstream_error";
      })();
      if (refusals.length < 8) refusals.push({ status: r.status, code, model: model.id });
    }
    throw { status: refusals[refusals.length - 1]?.status ?? 599, refusals };
  }

  private async handleOpenAi(
    body: Record<string, unknown>,
    res: http.ServerResponse,
  ): Promise<void> {
    const caps = deriveCapabilities(body);
    const ranked = rank(eligible(this.models, caps));
    if (ranked.length === 0) {
      writeJson(res, 400, { error: { message: "no eligible free model", type: "invalid_request_error" } });
      return;
    }
    const stream = body.stream === true;
    const first = ranked[0];

    if (!stream) {
      try {
        const { model, json, refusals } = await this.dialWithFailover(ranked, body);
        json.model = model.id;
        const headers = { "content-type": "application/json", ...this.refusalHeader(refusals) };
        res.writeHead(200, headers);
        res.end(JSON.stringify(json));
      } catch (e) {
        const ex = e as { status: number; refusals: Refusal[] };
        writeJson(res, ex.status, {
          error: { message: "upstream refusal", type: "upstream_error" },
        });
      }
      return;
    }

    // Streaming: relay gateway SSE, pin model id on every chunk.
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    // Flush the response head so the client sees headers immediately.
    res.flushHeaders();
    try {
      const up = await fetch(`${GATEWAY_BASE}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: JSON.stringify({ ...body, model: first.id }),
      });
      if (up.status >= 400 || !up.body) {
        res.write(`data: ${JSON.stringify({ error: { message: "upstream", type: "upstream_error" } })}\n\n`);
        res.end();
        return;
      }
      const seen = new Set<string>();
      const forced = first.id;
      await this.pumpSse(up, res, (json) => {
        const idKey = String(json.id ?? "");
        if (idKey) seen.add(idKey);
        json.model = forced;
        res.write(`data: ${JSON.stringify(json)}\n\n`);
        return false;
      }, () => {
        res.write("data: [DONE]\n\n");
        return true;
      });
      void seen;
    } catch {
      // never leave the client without a terminal event
      if (!res.writableEnded) res.end();
    }
  }

  /** Read the gateway's SSE response and hand each complete `data:` payload
   *  to `onData`. `onData` returns true when the caller wants to stop (e.g.
   *  after the [DONE] sentinel); upstream EOF also stops us. Returns the
   *  number of events seen. Assumes events are separated by blank lines. */
  private async pumpSse(
    up: Awaited<ReturnType<typeof fetch>>,
    res: http.ServerResponse,
    onData: (json: Record<string, unknown>) => boolean,
    onDone: () => boolean,
  ): Promise<number> {
    const reader = up.body!.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    let count = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (value && value.byteLength > 0) {
          buf += decoder.decode(value, { stream: true });
        }
        // Emit every complete event currently buffered. Events are
        // separated by a blank line (`\n\n` or `\r\n\r\n`).
        let sep: number;
        while ((sep = findEventSep(buf)) >= 0) {
          const event = buf.slice(0, sep);
          buf = buf.slice(sep + sepLen(buf, sep));
          const data = extractDataPayload(event);
          if (data === null) continue;
          if (data === "[DONE]") {
            count++;
            if (onDone()) {
              await reader.cancel();
              res.end();
              return count;
            }
            continue;
          }
          let json: Record<string, unknown>;
          try {
            json = JSON.parse(data) as Record<string, unknown>;
          } catch {
            continue;
          }
          count++;
          if (onData(json)) {
            await reader.cancel();
            res.end();
            return count;
          }
        }
        if (done) {
          if (!res.writableEnded) res.end();
          return count;
        }
        if (buf.length > 0 && buf.trim().length === 0) {
          // trailing whitespace only; keep reading, nothing to parse
          buf = "";
        }
      }
    } finally {
      try {
        await reader.cancel();
      } catch {
        /* already closed */
      }
      if (!res.writableEnded) res.end();
    }
  }

  private async handleAnthropic(
    body: Record<string, unknown>,
    res: http.ServerResponse,
  ): Promise<void> {
    const caps = deriveCapabilities(body);
    const ranked = rank(eligible(this.models, caps));
    if (ranked.length === 0) {
      writeJson(res, 400, {
        type: "error",
        error: { type: "invalid_request_error", message: "no eligible free model" },
      });
      return;
    }
    const { payload, declaredTools } = anthropicToOpenAi(body);
    const first = ranked[0];
    const anthHeaders = { "content-type": "application/json", "anthropic-version": "2023-06-01" };

    if (!body.stream) {
      try {
        const { model, json, refusals } = await this.dialWithFailover(ranked, payload);
        const answer = openAiToAnthropic(model.id, json);
        // wire check: only pass back the tool names the client declared
        const blocks = answer.content as Array<Record<string, unknown>>;
        const decl = new Set(declaredTools);
        answer.content = blocks.filter(
          (b) => b.type !== "tool_use" || decl.has(String(b.name)),
        );
        const headers = { ...anthHeaders, ...this.refusalHeader(refusals) };
        res.writeHead(200, headers);
        res.end(JSON.stringify(answer));
      } catch (e) {
        const ex = e as { status: number };
        writeJson(res, ex.status, { type: "error", error: { type: "api_error", message: "upstream refusal" } });
      }
      return;
    }

    // Streaming: translate gateway OpenAI SSE to Anthropic SSE.
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "anthropic-version": "2023-06-01",
    });
    res.flushHeaders();
    const state: AnthState = { started: false, blockIndex: 0, toolCalls: [] };
    const forced = first.id;
    const writeAnth = (fragment: string) => res.write(fragment + "\n");
    try {
      const up = await fetch(`${GATEWAY_BASE}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${GATEWAY_KEY}`,
        },
        body: JSON.stringify({ ...payload, model: first.id }),
      });
      if (up.status >= 400 || !up.body) {
        res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: { type: "api_error", message: "upstream" } })}\n\n`);
        res.end();
        return;
      }
      await this.pumpSse(up, res, (json) => {
        json.model = forced;
        const events = openAiLineToAnthEvents(`data: ${JSON.stringify(json)}`, state);
        if (events.length > 0) {
          writeAnth(events.map((e) => e + "\n").join(""));
        }
        return false;
      }, () => true);
      // OpenRouter models have no image output tokens; fine to leave usage
      // in the message_close. Ensure the Anthropic message is terminated.
      if (!state.started) {
        res.end();
        return;
      }
      return;
    } catch {
      if (!state.started && !res.writableEnded) {
        // Announce an error so the client never waits forever.
        writeAnth(
          `event: error\ndata: ${JSON.stringify({
            type: "error",
            error: { type: "api_error", message: "upstream" },
          })}`,
        );
      }
      if (!res.writableEnded) res.end();
    }
  }

  async route(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url || "/", "http://127.0.0.1");

    if (req.method === "GET" && url.pathname === "/v1/models") {
      writeJson(res, 200, this.modelsResponse(req.headers));
      return;
    }

    const isBear = !!req.headers.authorization;
    const isKey = !!req.headers["x-api-key"];
    if (!isBear && !isKey) {
      writeJson(res, 401, { error: { message: "missing auth header", type: "authentication_error" } });
      return;
    }

    if (req.method !== "POST") {
      writeJson(res, 405, { error: { message: "method not allowed", type: "not_allowed" } });
      return;
    }

    const body = await readBody(req);
    let parsed: Record<string, unknown> | null = null;
    try {
      parsed = JSON.parse(body) as Record<string, unknown>;
    } catch {
      /* bad json */
    }
    if (!parsed) {
      writeJson(res, 400, { error: { message: "invalid json body", type: "invalid_request_error" } });
      return;
    }

    if (url.pathname === "/v1/chat/completions" && isBear) {
      await this.handleOpenAi(parsed, res);
      return;
    }
    if (url.pathname === "/v1/messages" && isKey) {
      await this.handleAnthropic(parsed, res);
      return;
    }

    writeJson(res, 404, { error: { message: "not found", type: "not_found" } });
  }
}

/* ------------------------------------------------------------------ *
 * Env registration and work loop.
 * ------------------------------------------------------------------ */

async function register(port: number, catalog: ProviderCatalog): Promise<void> {
  await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
  });
}

async function workDone(): Promise<boolean> {
  try {
    const res = await fetch(`${ENV_URL}/work`);
    const text = await res.text();
    const j = JSON.parse(text) as Record<string, unknown>;
    return j.done === true;
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------------ *
 * Entry point.
 * ------------------------------------------------------------------ */

function printCatalog(catalog: ProviderCatalog): void {
  process.stdout.write(JSON.stringify(catalog, null, 2) + "\n");
}

async function main(): Promise<void> {
  const { models } = readSealedCatalog();

  if (process.argv.includes("--list")) {
    printCatalog(buildCatalog(models));
    return;
  }

  const aggregator = new Aggregator(models);
  const server = http.createServer((req, res) => {
    aggregator
      .route(req, res)
      .catch(() => {
        if (!res.writableEnded) writeJson(res, 500, { error: { message: "internal error", type: "server_error" } });
      });
  });

  const port = Number(process.env.PORT || 8093);
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));

  await register(port, aggregator.catalog);

  // Poll /work until the env reports done, then exit cleanly.
  while (true) {
    if (await workDone()) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});