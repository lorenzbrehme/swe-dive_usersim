/**
 * Pi coding-agent extension: Azure OpenAI Chat Completions provider.
 *
 * Loaded by pi's jiti extension loader from ~/.pi/agent/extensions/.
 * Registers a custom "azure-foundry" provider that calls the
 * deployment-specific Chat Completions endpoint with ?api-version as a
 * query parameter.
 *
 * Pi's built-in openai-responses provider omits the query param (→ 404),
 * the azure-openai-responses provider adds /v1/ to the path (wrong URL),
 * and the Responses API returns 401 for some Azure resources.
 * Chat Completions at /openai/deployments/{dep}/chat/completions works
 * universally and matches what LiteLLM uses for the user simulator.
 */

import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  calculateCost,
  createAssistantMessageEventStream,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "fs";

const LOG_FILE = "/logs/agent/azure-foundry.log";

function log(msg: string): void {
  const ts = new Date().toISOString();
  try {
    appendFileSync(LOG_FILE, `${ts} ${msg}\n`);
  } catch {
    /* ignore — log dir may not exist yet */
  }
}

function convertMessages(
  messages: Message[],
  systemPrompt?: string,
): Array<{ role: string; content: string }> {
  const result: Array<{ role: string; content: string }> = [];
  if (systemPrompt) {
    result.push({ role: "system", content: systemPrompt });
  }
  for (const msg of messages) {
    if (msg.role === "user") {
      const content =
        typeof msg.content === "string"
          ? msg.content
          : msg.content.map((c) => (c as TextContent).text).join("");
      if (content.trim()) result.push({ role: "user", content });
    } else if (msg.role === "assistant") {
      const content = msg.content
        .filter((c) => c.type === "text")
        .map((c) => (c as TextContent).text)
        .join("");
      if (content.trim()) result.push({ role: "assistant", content });
    }
  }
  return result;
}

function streamAzureFoundry(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const azureBase = (process.env["AZURE_API_BASE"] ?? "").replace(/\/+$/, "");
    const azureVersion =
      process.env["AZURE_API_VERSION"] ?? "2025-04-01-preview";
    // Prefer env vars directly — pi passes the literal "$AZURE_API_KEY" string
    // to streamSimple via options.apiKey rather than the resolved value.
    const apiKey =
      process.env["AZURE_API_KEY"] ??
      process.env["OPENAI_API_KEY"] ??
      options?.apiKey ??
      "";

    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    try {
      const url = `${azureBase}/openai/deployments/${model.id}/chat/completions?api-version=${azureVersion}`;
      log(`fetch POST ${url} keyLen=${apiKey.length} keyPrefix=${apiKey.substring(0, 4)}`);
      const body: Record<string, unknown> = {
        messages: convertMessages(context.messages, context.systemPrompt),
        stream: true,
        stream_options: { include_usage: true },
      };

      const resp = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "api-key": apiKey,
        },
        body: JSON.stringify(body),
        signal: options?.signal,
      });

      if (!resp.ok) {
        const errBody = await resp.text();
        log(`HTTP ${resp.status} error: ${errBody.slice(0, 500)}`);
        throw new Error(`HTTP ${resp.status}: ${errBody}`);
      }
      log(`HTTP ${resp.status} OK, streaming...`);

      stream.push({ type: "start", partial: output });
      output.content.push({ type: "text", text: "" });
      stream.push({ type: "text_start", contentIndex: 0, partial: output });

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (!data || data === "[DONE]") continue;

          let event: Record<string, unknown>;
          try {
            event = JSON.parse(data) as Record<string, unknown>;
          } catch {
            continue;
          }

          const choices = event["choices"] as
            | Array<Record<string, unknown>>
            | undefined;
          if (choices?.length) {
            const delta = choices[0]["delta"] as
              | Record<string, unknown>
              | undefined;
            const content = delta?.["content"];
            if (typeof content === "string" && content) {
              (output.content[0] as TextContent).text += content;
              stream.push({
                type: "text_delta",
                contentIndex: 0,
                delta: content,
                partial: output,
              });
            }
            if (choices[0]["finish_reason"] === "length") {
              output.stopReason = "length";
            }
          }

          const usage = event["usage"] as Record<string, number> | undefined;
          if (usage) {
            output.usage.input = usage["prompt_tokens"] ?? 0;
            output.usage.output = usage["completion_tokens"] ?? 0;
            output.usage.totalTokens = usage["total_tokens"] ?? 0;
            try {
              calculateCost(model, output.usage);
            } catch {
              /* model may not have cost info; ignore */
            }
          }
        }
      }

      const finalText = (output.content[0] as TextContent).text;
      log(`done, text length=${finalText.length}`);
      stream.push({
        type: "text_end",
        contentIndex: 0,
        content: finalText,
        partial: output,
      });
      stream.push({
        type: "done",
        reason: output.stopReason as "stop" | "length",
        message: output,
      });
      stream.end();
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      log(`catch: ${errMsg}`);
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = errMsg;
      stream.push({
        type: "error",
        reason: output.stopReason as "aborted" | "error",
        error: output,
      });
      stream.end();
    }
  })();

  return stream;
}

export default function (pi: ExtensionAPI) {
  // Derive the deployment name from LLM_MODEL ("azure/<deployment>") which is
  // always forwarded, or fall back to AZURE_MODEL, then to a safe default.
  const lmm = process.env["LLM_MODEL"] ?? "";
  const deployment =
    (lmm.includes("/") ? lmm.split("/").slice(1).join("/") : lmm) ||
    process.env["AZURE_MODEL"] ||
    "gpt-4o";
  const azureBase = (process.env["AZURE_API_BASE"] ?? "").replace(/\/+$/, "");
  log(
    `registering provider: deployment=${deployment} base=${azureBase || "(unset)"}`,
  );
  pi.registerProvider("azure-foundry", {
    name: "Azure OpenAI (Foundry)",
    baseUrl: azureBase,
    apiKey: "$AZURE_API_KEY",
    api: "azure-foundry-chat",
    streamSimple: streamAzureFoundry,
    models: [
      {
        id: deployment,
        name: deployment,
        reasoning: false,
        input: ["text", "image"],
        contextWindow: 128000,
        maxTokens: 16384,
      },
    ],
  });
}
