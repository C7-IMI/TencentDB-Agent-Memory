/**
 * Files API passthrough (`/v1/files`).
 *
 * The Anthropic Files API is how a client uploads an attachment (an image, for
 * example) and then references it from a message by `file_id`. The proxy serves
 * `/v1/messages` itself, so the upload must reach the same upstream account or
 * the `file_id` in the forwarded message means nothing there.
 *
 * Without these routes an upload falls through to the catch-all
 * `POST /*` → OpenAI chat handler, which JSON-parses the multipart body and
 * answers `400 Invalid JSON body`. A client that uploads an attachment then
 * retries the whole model request and reports a transport failure, so the
 * visible symptom is "the model request failed" rather than "the upload failed".
 *
 * This is a **raw passthrough**, deliberately:
 *   - the body is taken as `arrayBuffer()`, never decoded as text, so a
 *     multipart boundary and binary payload survive byte-for-byte;
 *   - the `content-type` header (which carries the multipart boundary) is
 *     preserved, because the hop-by-hop skip list does not include it;
 *   - the response is streamed back with its own content-type, because an
 *     upload answer is JSON while a file download is binary.
 *
 * It is not a conversation turn: no session init, no injection, no L0 record.
 * An attachment upload is not something the model said.
 */
import type { Context } from "hono";
import type { ProxyConfig } from "../types.js";
import { stageAuth } from "../pipeline/stages/auth.js";
import {
  SKIP_REQUEST_HEADERS_WITH_INTERNAL,
  filterResponseHeaders,
} from "../common/constants.js";
import { log } from "../report/log.js";

/** The part of the request path starting at `/v1/files`, plus the query string. */
function filesSuffix(path: string, search: string): string {
  const idx = path.indexOf("/v1/files");
  if (idx < 0) return "/v1/files" + search;
  // Keep a trailing `/files/<id>` but never the agent or space prefix.
  return path.slice(idx) + search;
}

/**
 * Resolve the upstream base + key for a Files API call.
 *
 * `/v1/messages` is routed through the per-agent Anthropic base
 * (`upstream.agents.<agent>.url`, e.g. https://api.deepseek.com/anthropic), not
 * the OpenAI-compatible default (`upstream.url`, e.g. .../v1). A Files API call
 * must land on the same base as the message that references its `file_id`, so
 * the agent segment is resolved the same way here.
 */
function resolveUpstream(c: Context, config: ProxyConfig): { base: string; apiKey: string } {
  const segments = c.req.path.split("/").filter(Boolean);
  const agentName = segments.length > 0 && segments[0] !== "v1" ? segments[0] : undefined;
  const agentEntry = agentName ? config.upstream?.agents?.[agentName] : undefined;
  const base =
    agentEntry?.url ??
    (config.upstream as { anthropicUpstream?: { url: string } })?.anthropicUpstream?.url ??
    config.upstream?.url ??
    "";
  const apiKey = agentEntry?.apiKey || config.upstream?.apiKey || "";
  return { base, apiKey };
}

/** Join without doubling `/v1`, because the default base already ends in it. */
function joinFilesUrl(base: string, suffix: string): string {
  const b = base.replace(/\/+$/, "");
  const s = suffix.startsWith("/") ? suffix : "/" + suffix;
  if (/\/v1$/.test(b) && s.startsWith("/v1/")) return b + s.slice(3);
  return b + s;
}

function buildHeaders(c: Context, apiKey: string): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const [k, v] of c.req.raw.headers.entries()) {
    if (!SKIP_REQUEST_HEADERS_WITH_INTERNAL.has(k.toLowerCase())) {
      headers[k] = v;
    }
  }
  if (apiKey) {
    headers["x-api-key"] = apiKey;
    delete headers["authorization"];
  }
  return headers;
}

export async function handleFilesPassthrough(
  c: Context,
  config: ProxyConfig,
): Promise<Response> {
  const auth = await stageAuth(c, "xapi-first");
  if (auth.rejected) {
    return c.json(
      { error: `Authentication failed: ${auth.rejectReason ?? "unknown"}` },
      401,
    );
  }

  const suffix = filesSuffix(c.req.path, new URL(c.req.url).search);
  const { base, apiKey } = resolveUpstream(c, config);
  const upstreamUrl = joinFilesUrl(base, suffix);
  const method = c.req.method.toUpperCase();

  const headers = buildHeaders(c, apiKey);
  let body: ArrayBuffer | undefined;
  if (method !== "GET" && method !== "HEAD" && method !== "DELETE") {
    body = await c.req.arrayBuffer();
  }

  log.info("files.passthrough", {
    method,
    path: c.req.path,
    upstreamUrl,
    bytes: body ? body.byteLength : 0,
    spaceId: auth.spaceId,
  });

  let upstreamResp: Response;
  try {
    upstreamResp = await fetch(upstreamUrl, { method, headers, body });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    log.warn("files.passthrough.error", { path: c.req.path, message });
    return c.json({ error: "Upstream files request failed", detail: message }, 502);
  }

  return new Response(upstreamResp.body, {
    status: upstreamResp.status,
    headers: filterResponseHeaders(upstreamResp.headers),
  });
}
