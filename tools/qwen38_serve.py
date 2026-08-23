#!/usr/bin/env python3
"""OpenAI-compatible server for the resident Qwen3.8-27B runtime.

Wraps the resident C chat binary (machine protocol mode) behind two APIs
over one engine:

  POST /v1/chat/completions  - SSE streaming for Chatbox, Cherry Studio,
                               Open WebUI, Raycast, Continue, Cline
  POST /v1/responses         - the OpenAI Responses API, which is the only
                               wire protocol Codex CLI still speaks

Standard library only; the shim renders the official chat template
(thinking and tool calling optional) and owns the HTTP protocol, while
all model execution stays in the C/Metal runtime.

Usage:
  tools/qwen38_serve.py                     # serve on 127.0.0.1:8199
  tools/qwen38_serve.py --port 9000
  curl http://127.0.0.1:8199/v1/chat/completions -d '{
    "model": "qwen3.8-27b", "stream": true,
    "messages": [{"role": "user", "content": "hello"}]}'
  tools/codex-qwen                          # Codex CLI on this server

Then point a client at base URL http://127.0.0.1:8199/v1 with any API
key. Requests are served one at a time; the model loads and wires once
at startup, so every request runs at ready-state latency.
"""

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "qwen3.8-27b"


REASONING_EFFORT_TEXT = {
    "xhigh": ("Reasoning effort is set to xhigh. Please think carefully "
              "through the task, validate key assumptions, consider "
              "plausible alternatives, and prioritize correctness, "
              "consistency, and clarity in the final answer."),
    "low": ("Reasoning effort is set to low. Keep your thinking brief "
            "and focused, moving directly to the conclusion without "
            "unnecessary elaboration."),
    "medium": None,
}


def message_text(message):
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict))
    return content


def render_template(messages, thinking=False, effort="xhigh"):
    """The official Qwen3.8 template. Default is the pinned
    enable_thinking=false rendering; thinking mode opens the reply at
    '<think>\n' and injects the Qwen3.8 reasoning-effort instructions
    (verbatim template strings) at the top of the system turn, creating
    one when the effort is not medium and no system message exists.

    Prior assistant turns replay the think block the model actually saw
    (empty in no-think mode, or the turn's reasoning_content) — the
    Qwen3.8 preserve_thinking semantics. This also makes each follow-up
    request an exact token extension of the resident conversation state,
    so the engine prefills only the new turn instead of the whole
    history."""
    parts = []
    remaining = list(messages)
    instructions = REASONING_EFFORT_TEXT.get(effort) if thinking else None
    if remaining and remaining[0].get("role") == "system":
        content = message_text(remaining[0]).strip()
        remaining = remaining[1:]
        head = (instructions + "\n\n" if instructions else "") + content
        if head:
            parts.append(f"<|im_start|>system\n{head}<|im_end|>\n")
    elif instructions:
        parts.append(f"<|im_start|>system\n{instructions}<|im_end|>\n")
    for message in remaining:
        role = message.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = message_text(message)
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            parts.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n"
                         f"</think>\n\n{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if thinking:
        parts.append("<|im_start|>assistant\n<think>\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


class ThinkSplitter:
    """Split a streamed reply into reasoning and answer. In thinking
    mode the generation begins inside the think block; everything before
    '</think>' is reasoning_content, everything after (minus the
    separating blank line) is content. Holds back a possible partial tag
    at a delta boundary."""

    def __init__(self, active):
        self.active = active
        self.buffer = ""
        self.done = not active

    def feed(self, text):
        if self.done:
            return ("", text) if self.active else (None, text)
        self.buffer += text
        marker = self.buffer.find("</think>")
        if marker >= 0:
            reasoning = self.buffer[:marker]
            rest = self.buffer[marker + len("</think>"):]
            rest = rest.lstrip("\n")
            self.buffer = ""
            self.done = True
            return (reasoning, rest)
        # hold back a suffix that could start the closing tag
        keep = 0
        for probe in range(min(len("</think>") - 1, len(self.buffer)), 0, -1):
            if "</think>".startswith(self.buffer[-probe:]):
                keep = probe
                break
        emit = self.buffer[:len(self.buffer) - keep]
        self.buffer = self.buffer[len(self.buffer) - keep:]
        return (emit, "")

    def flush(self):
        rest = self.buffer
        self.buffer = ""
        return rest


# --- Responses API (Codex CLI) ------------------------------------------
#
# Codex CLI 0.149 dropped `wire_api = "chat"`; it speaks only the OpenAI
# Responses API. The shim below renders a Responses request into the same
# Qwen3.8 chat template used above - including the official tool-calling
# block - and turns the reply back into the Responses SSE events that
# Codex actually consumes (codex-rs/codex-api/src/sse/responses.rs):
# response.created, response.output_text.delta, response.reasoning_text.delta,
# response.output_item.added/done and response.completed. Codex ignores
# response.function_call_arguments.delta, so a tool call is delivered whole
# in one output_item.done carrying a `function_call` ResponseItem.

TOOL_CALL_FORMAT = (
    "\n\nIf you choose to call a function ONLY reply in the following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>"
    "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
    "the specified format: an inner <function=...></function> block must be "
    "nested within <tool_call></tool_call> XML tags\n- Required parameters "
    "MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT "
    "after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>"
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FUNCTION_RE = re.compile(r"<function=([^>\s]*)>\s*(.*?)\s*</function>", re.S)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]*)>\n?(.*?)\n?</parameter>",
                          re.S)


def flatten_tools(tools):
    """Responses tool list -> ordered {qualified name: (namespace, schema)}.

    Codex sends flat `function` tools plus `namespace` tools that group
    sub-functions; a namespaced call must come back with both `name` and
    `namespace`, so the model sees `namespace.function` and the parser
    splits it again. Provider-executed types (web_search) have no local
    implementation and are dropped."""
    table = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        if kind == "function":
            table[tool.get("name", "")] = (None, tool)
        elif kind == "namespace":
            space = tool.get("name", "")
            for inner in tool.get("tools") or []:
                if isinstance(inner, dict) and inner.get("type") == "function":
                    table[f"{space}.{inner.get('name', '')}"] = (space, inner)
    table.pop("", None)
    return table


def tool_manifest(table):
    """The <tools> block entries, in the OpenAI function-schema shape the
    Qwen3.8 template dumps verbatim."""
    manifest = []
    for name, (_, schema) in table.items():
        manifest.append({"type": "function", "function": {
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters",
                                     {"type": "object", "properties": {}})}})
    return manifest


def content_text(content):
    """Text of a Responses content array (input_text/output_text/...)."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") in ("input_text", "output_text", "text",
                                    "summary_text", "reasoning_text"):
                parts.append(part.get("text", ""))
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


def call_output_text(output):
    """function_call_output.output is a bare string or structured items."""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if "content" in output:
            return content_text(output["content"])
        return json.dumps(output)
    if isinstance(output, list):
        return content_text(output)
    return "" if output is None else str(output)


def render_call_block(name, arguments):
    """One assistant-side <tool_call> in the template's own XML shape."""
    try:
        fields = json.loads(arguments) if isinstance(arguments, str) \
            else (arguments or {})
    except ValueError:
        fields = {}
    if not isinstance(fields, dict):
        fields = {}
    body = [f"<tool_call>\n<function={name}>\n"]
    for key, value in fields.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        body.append(f"<parameter={key}>\n{rendered}\n</parameter>\n")
    body.append("</function>\n</tool_call>")
    return "".join(body)


def render_responses_template(request, table, thinking, effort):
    """Render a Responses request into the Qwen3.8 chat template.

    `instructions` and any leading developer messages become the system
    turn, ahead of the tool block. Assistant/function_call items replay as
    assistant turns and function_call_output items as <tool_response>
    blocks, so each Codex turn is an exact token extension of the previous
    one and the resident engine prefills only the new suffix."""
    reasoning = REASONING_EFFORT_TEXT.get(effort) if thinking else None
    items = list(request.get("input") or [])

    head = []
    if reasoning:
        head.append(reasoning)
    instructions = (request.get("instructions") or "").strip()

    # Leading developer messages are instructions, not conversation; Qwen
    # has no developer role, so they extend the system turn.
    developer = []
    while items:
        first = items[0]
        if isinstance(first, dict) and first.get("type", "message") == \
                "message" and first.get("role") == "developer":
            developer.append(content_text(first.get("content")).strip())
            items.pop(0)
        else:
            break

    manifest = tool_manifest(table)
    if manifest:
        block = ["# Tools\n\nYou have access to the following functions:"
                 "\n\n<tools>"]
        for entry in manifest:
            block.append("\n" + json.dumps(entry))
        block.append("\n</tools>")
        block.append(TOOL_CALL_FORMAT)
        head.append("".join(block))
    if instructions:
        head.append(instructions)
    head.extend(part for part in developer if part)

    parts = []
    if head:
        parts.append("<|im_start|>system\n" + "\n\n".join(head) +
                     "<|im_end|>\n")

    # Consecutive function_call items share one assistant turn and
    # consecutive outputs share one user turn, matching the template.
    index = 0
    while index < len(items):
        item = items[index]
        if not isinstance(item, dict):
            index += 1
            continue
        kind = item.get("type", "message")
        if kind == "function_call":
            calls = []
            while index < len(items) and isinstance(items[index], dict) and \
                    items[index].get("type") == "function_call":
                call = items[index]
                calls.append(render_call_block(call.get("name", ""),
                                               call.get("arguments", "{}")))
                index += 1
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n" +
                         "\n".join(calls) + "<|im_end|>\n")
            continue
        if kind in ("function_call_output", "custom_tool_call_output"):
            blocks = []
            while index < len(items) and isinstance(items[index], dict) and \
                    items[index].get("type") in ("function_call_output",
                                                 "custom_tool_call_output"):
                text = call_output_text(items[index].get("output"))
                blocks.append(f"\n<tool_response>\n{text}\n</tool_response>")
                index += 1
            parts.append("<|im_start|>user" + "".join(blocks) + "<|im_end|>\n")
            continue
        index += 1
        if kind == "reasoning":
            # Reasoning is never replayed: the shim does not emit reasoning
            # items, so replaying one would break the prefix extension.
            continue
        if kind != "message":
            continue
        role = item.get("role", "user")
        text = content_text(item.get("content"))
        if role == "assistant":
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n" +
                         text + "<|im_end|>\n")
        else:
            if role not in ("user", "developer", "system"):
                role = "user"
            parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")

    if thinking:
        parts.append("<|im_start|>assistant\n<think>\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def coerce_argument(text, schema):
    """A parameter arrives as raw text; the tool's JSON schema decides
    whether it stays a string or is parsed back into JSON."""
    kind = (schema or {}).get("type")
    if kind == "string":
        return text
    if kind == "boolean":
        lowered = text.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_tool_calls(text, table):
    """Extract <tool_call> blocks, returning (leading text, calls)."""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        for name, body in FUNCTION_RE.findall(block):
            namespace, schema = table.get(name, (None, {}))
            properties = (schema.get("parameters") or {}).get("properties") \
                or {}
            arguments = {}
            for key, value in PARAMETER_RE.findall(body):
                arguments[key] = coerce_argument(value, properties.get(key))
            bare = name.split(".", 1)[1] if namespace and "." in name else name
            calls.append({"name": bare, "namespace": namespace,
                          "arguments": json.dumps(arguments)})
    marker = text.find(TOOL_CALL_OPEN)
    leading = text if marker < 0 else text[:marker]
    return leading.strip(), calls


class ToolCallSplitter:
    """Stream text until a tool call starts. The template forbids any
    suffix after a call, so once <tool_call> appears nothing more is
    surfaced as assistant text; a partial opening tag is held back at a
    delta boundary."""

    def __init__(self):
        self.buffer = ""
        self.stopped = False

    def feed(self, text):
        if self.stopped:
            return ""
        self.buffer += text
        marker = self.buffer.find(TOOL_CALL_OPEN)
        if marker >= 0:
            emit = self.buffer[:marker]
            self.buffer = ""
            self.stopped = True
            return emit
        keep = 0
        for probe in range(min(len(TOOL_CALL_OPEN) - 1, len(self.buffer)),
                           0, -1):
            if TOOL_CALL_OPEN.startswith(self.buffer[-probe:]):
                keep = probe
                break
        emit = self.buffer[:len(self.buffer) - keep]
        self.buffer = self.buffer[len(self.buffer) - keep:]
        return emit

    def flush(self):
        rest = "" if self.stopped else self.buffer
        self.buffer = ""
        return rest


class Engine:
    """One resident chat process, one request at a time."""

    def __init__(self, arguments):
        self.arguments = arguments
        self.lock = threading.Lock()
        self.process = None
        self.ready = {}

    def start(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        binary = os.path.join(repo, "build", "qwen38-m3-chat")
        model = self.arguments.model_dir or os.path.join(
            repo, "tmp", "qwen38-27b-runtime")
        command = [
            binary, model,
            self.arguments.metallib or os.path.join(
                repo, "build", "qwen38-m3-q4.metallib"),
            self.arguments.tokenizer or os.path.join(
                model, "tokenizer.q38tok"),
            str(self.arguments.context), str(self.arguments.max_tokens),
            str(self.arguments.temperature), str(self.arguments.top_k),
            str(self.arguments.seed),
        ]
        environment = dict(os.environ, QWEN38_MACHINE="1")
        print(f"loading model (one-time weight wiring)...", flush=True)
        self.process = subprocess.Popen(
            command, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
            bufsize=1)
        line = self.process.stdout.readline()
        if not line.startswith("R "):
            raise RuntimeError(f"engine did not become ready: {line!r}")
        self.ready = json.loads(line[2:])
        print(f"model ready: context {self.ready.get('context')}, "
              f"max reply {self.ready.get('max_new')} tokens", flush=True)

    def generate(self, rendered, on_delta, sampling=None):
        """Run one request; call on_delta(text) per chunk; return stats."""
        with self.lock:
            if self.process.poll() is not None:
                self.start()
            if sampling:
                request = dict(sampling)
                request["prompt"] = rendered
                self.process.stdin.write(json.dumps(request) + "\n")
            else:
                self.process.stdin.write(json.dumps(rendered) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("engine exited mid-request")
                if line.startswith("D "):
                    on_delta(json.loads(line[2:]))
                elif line.startswith("E "):
                    return json.loads(line[2:])
                elif line.startswith("X "):
                    raise RuntimeError(json.loads(line[2:]))


ENGINE = None
THINKING_DEFAULT = False
EFFORT_DEFAULT = "xhigh"
RESPONSES_HONOR_EFFORT = os.environ.get(
    "QWEN38_RESPONSES_EFFORT", "") not in ("", "0")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *values):
        print(f"{self.address_string()} {format % values}", flush=True)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        # Codex probes /v1/models?client_version=..., so match on the
        # path alone rather than the raw request target.
        route = self.path.split("?", 1)[0]
        if route in ("/v1/models", "/models"):
            self.send_json({"object": "list", "data": [{
                "id": MODEL_ID, "object": "model",
                "created": int(time.time()), "owned_by": "local"}]})
        elif route == "/health":
            # The launcher checks `context` here: a server started for a
            # small chat context cannot serve a Codex-sized prompt.
            ready = ENGINE.ready if ENGINE else {}
            self.send_json({"status": "ok", "model": MODEL_ID,
                            "context": ready.get("context"),
                            "max_new": ready.get("max_new"),
                            "thinking": THINKING_DEFAULT})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path in ("/v1/responses", "/responses"):
            self.handle_responses()
            return
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
            messages = request["messages"]
        except (ValueError, KeyError):
            self.send_json({"error": {
                "message": "body must be JSON with a messages array",
                "type": "invalid_request_error"}}, 400)
            return
        # Thinking mode: the server default (--thinking /
        # QWEN38_THINKING) can be overridden per request with the
        # OpenAI-style reasoning_effort field ("none" disables thinking,
        # low/medium/xhigh select the Qwen3.8 effort levels; "high" maps
        # to xhigh).
        thinking = THINKING_DEFAULT
        effort = EFFORT_DEFAULT
        effort_field = request.get("reasoning_effort")
        if isinstance(effort_field, str):
            level = effort_field.lower()
            if level == "none":
                thinking = False
            elif level in ("low", "medium", "xhigh", "high"):
                thinking = True
                effort = "xhigh" if level == "high" else level
        rendered = render_template(messages, thinking, effort)
        # Per-request sampling passes straight through to the engine.
        # Absent fields keep the engine defaults (greedy, which also
        # enables lossless speculative decoding); a request that sets
        # temperature > 0 gets true sampled decoding with top_k, top_p,
        # min_p and presence_penalty honored.
        sampling = {}
        for source, target in (("temperature", "temperature"),
                               ("top_k", "top_k"),
                               ("top_p", "top_p"),
                               ("min_p", "min_p"),
                               ("presence_penalty", "presence_penalty"),
                               ("max_tokens", "max_new"),
                               ("max_completion_tokens", "max_new")):
            value = request.get(source)
            if isinstance(value, (int, float)) and not isinstance(
                    value, bool):
                sampling[target] = value
        identifier = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if request.get("stream"):
            include_usage = bool(
                (request.get("stream_options") or {}).get("include_usage"))
            self.stream_completion(rendered, identifier, created,
                                   include_usage, sampling, thinking)
        else:
            self.plain_completion(rendered, identifier, created, sampling,
                                  thinking)

    # --- Responses API ---------------------------------------------

    def sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def sse_event(self, name, payload):
        payload = dict(payload, type=name)
        data = f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def sse_close(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def handle_responses(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
        except ValueError:
            self.send_json({"error": {"message": "body must be JSON",
                                      "type": "invalid_request_error"}}, 400)
            return
        thinking = THINKING_DEFAULT
        effort = EFFORT_DEFAULT
        asked = (request.get("reasoning") or {}).get("effort")
        if RESPONSES_HONOR_EFFORT and isinstance(asked, str):
            level = asked.lower()
            if level in ("none", "minimal"):
                thinking = False
            elif level in ("low", "medium", "xhigh", "high", "max", "ultra"):
                thinking = True
                effort = "low" if level == "low" else \
                    ("medium" if level == "medium" else "xhigh")
        table = flatten_tools(request.get("tools"))
        rendered = render_responses_template(request, table, thinking, effort)
        sampling = {}
        budget = request.get("max_output_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            sampling["max_new"] = budget
        identifier = f"resp_{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if request.get("stream", True):
            self.stream_responses(rendered, table, identifier, created,
                                  sampling, thinking)
        else:
            self.plain_responses(rendered, table, identifier, created,
                                 sampling, thinking)

    def responses_envelope(self, identifier, created, status, output=None,
                           stats=None):
        envelope = {"id": identifier, "object": "response",
                    "created_at": created, "status": status,
                    "model": MODEL_ID, "output": output or []}
        if stats is not None:
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("tokens", 0)
            envelope["usage"] = {
                "input_tokens": prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_tokens + completion_tokens}
        return envelope

    def responses_output(self, text, table, identifier):
        """The finished output array: assistant text first, then one
        function_call item per parsed <tool_call>."""
        leading, calls = parse_tool_calls(text, table)
        output = []
        if leading:
            output.append({"type": "message", "id": f"msg_{identifier[5:]}",
                           "status": "completed", "role": "assistant",
                           "content": [{"type": "output_text",
                                        "text": leading}]})
        for index, call in enumerate(calls):
            item = {"type": "function_call",
                    "id": f"fc_{identifier[5:]}_{index}",
                    "call_id": f"call_{identifier[5:]}_{index}",
                    "name": call["name"], "arguments": call["arguments"]}
            if call["namespace"]:
                item["namespace"] = call["namespace"]
            output.append(item)
        return output

    def plain_responses(self, rendered, table, identifier, created,
                        sampling, thinking):
        chunks = []
        try:
            stats = ENGINE.generate(rendered, chunks.append, sampling)
        except RuntimeError as failure:
            self.send_json({"error": {"message": str(failure),
                                      "type": "server_error"}}, 500)
            return
        text = "".join(chunks)
        if thinking:
            marker = text.find("</think>")
            if marker >= 0:
                text = text[marker + len("</think>"):].lstrip("\n")
        output = self.responses_output(text, table, identifier)
        self.send_json(self.responses_envelope(identifier, created,
                                               "completed", output, stats))

    def stream_responses(self, rendered, table, identifier, created,
                         sampling, thinking):
        self.sse_open()
        message_id = f"msg_{identifier[5:]}"
        try:
            self.sse_event("response.created", {"response":
                self.responses_envelope(identifier, created, "in_progress")})
            think = ThinkSplitter(thinking)
            splitter = ToolCallSplitter()
            answer = []
            opened = [False]

            def deliver(chunk):
                reasoning, content = think.feed(chunk)
                if reasoning:
                    self.sse_event("response.reasoning_text.delta",
                                   {"item_id": f"rs_{identifier[5:]}",
                                    "output_index": 0, "content_index": 0,
                                    "delta": reasoning})
                if not content:
                    return
                answer.append(content)
                visible = splitter.feed(content)
                if not visible:
                    return
                if not opened[0]:
                    opened[0] = True
                    self.sse_event("response.output_item.added", {
                        "output_index": 0,
                        "item": {"type": "message", "id": message_id,
                                 "status": "in_progress", "role": "assistant",
                                 "content": []}})
                self.sse_event("response.output_text.delta", {
                    "item_id": message_id, "output_index": 0,
                    "content_index": 0, "delta": visible})

            stats = ENGINE.generate(rendered, deliver, sampling)
            tail = think.flush()
            if tail and think.done:
                answer.append(tail)
                visible = splitter.feed(tail)
                if visible:
                    if not opened[0]:
                        opened[0] = True
                        self.sse_event("response.output_item.added", {
                            "output_index": 0,
                            "item": {"type": "message", "id": message_id,
                                     "status": "in_progress",
                                     "role": "assistant", "content": []}})
                    self.sse_event("response.output_text.delta", {
                        "item_id": message_id, "output_index": 0,
                        "content_index": 0, "delta": visible})
            output = self.responses_output("".join(answer), table, identifier)
            calls = [item["name"] for item in output
                     if item["type"] == "function_call"]
            prompt_tokens = stats.get("prompt_tokens", 0)
            reused = stats.get("prefilled_from", 0)
            generated = stats.get("tokens", 0)
            total = stats.get("total_s", 0.0) or 0.0
            first = stats.get("first_token_s", 0.0) or 0.0
            rate = generated / (total - first) if total > first else 0.0
            if not reused:
                print(f"responses: cold turn exposes {len(table)} tools: "
                      f"{', '.join(table)}", flush=True)
            print(f"responses: {len(table)} tools, {prompt_tokens} prompt "
                  f"tokens ({reused} reused), {first:.1f} s to first token, "
                  f"{generated} generated at {rate:.1f} tok/s, "
                  f"calls={calls or 'none'}", flush=True)
            for index, item in enumerate(output):
                self.sse_event("response.output_item.done",
                               {"output_index": index, "item": item})
            self.sse_event("response.completed", {"response":
                self.responses_envelope(identifier, created, "completed",
                                        output, stats)})
            self.sse_close()
        except (RuntimeError, BrokenPipeError, ConnectionResetError) \
                as failure:
            print(f"responses stream aborted: {failure}", flush=True)
            self.close_connection = True

    def plain_completion(self, rendered, identifier, created,
                         sampling=None, thinking=False):
        chunks = []
        try:
            stats = ENGINE.generate(rendered, chunks.append, sampling)
        except RuntimeError as failure:
            self.send_json({"error": {"message": str(failure),
                                      "type": "server_error"}}, 500)
            return
        text = "".join(chunks)
        message = {"role": "assistant", "content": text}
        if thinking:
            marker = text.find("</think>")
            if marker >= 0:
                message["reasoning_content"] = text[:marker]
                message["content"] = text[marker + len("</think>"):] \
                    .lstrip("\n")
            else:
                message["reasoning_content"] = text
                message["content"] = ""
        self.send_json({
            "id": identifier, "object": "chat.completion",
            "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message,
                "finish_reason": stats.get("stop", "stop")}],
            "usage": {
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("tokens", 0),
                "total_tokens": stats.get("prompt_tokens", 0) +
                                stats.get("tokens", 0)},
        })

    def stream_completion(self, rendered, identifier, created,
                          include_usage, sampling=None, thinking=False):
        # The SSE body carries no Content-Length, so it must use chunked
        # transfer encoding: without the 0-length terminator a keep-alive
        # client never sees the body end and its UI stays in the waiting
        # state after the reply.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def write_chunk(data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data +
                             b"\r\n")
            self.wfile.flush()

        def event(payload):
            write_chunk(f"data: {json.dumps(payload)}\n\n".encode())

        def chunk(delta, finish=None):
            return {"id": identifier, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        try:
            event(chunk({"role": "assistant", "content": ""}))
            splitter = ThinkSplitter(thinking)

            def deliver(text):
                reasoning, content = splitter.feed(text)
                if reasoning:
                    event(chunk({"reasoning_content": reasoning}))
                if content:
                    event(chunk({"content": content}))

            stats = ENGINE.generate(rendered, deliver, sampling)
            tail = splitter.flush()
            if tail:
                if splitter.done:
                    event(chunk({"content": tail}))
                else:
                    event(chunk({"reasoning_content": tail}))
            event(chunk({}, stats.get("stop", "stop")))
            if include_usage:
                prompt_tokens = stats.get("prompt_tokens", 0)
                completion_tokens = stats.get("tokens", 0)
                event({"id": identifier,
                       "object": "chat.completion.chunk",
                       "created": created, "model": MODEL_ID,
                       "choices": [],
                       "usage": {
                           "prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens +
                                           completion_tokens}})
            write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (RuntimeError, BrokenPipeError,
                ConnectionResetError) as failure:
            print(f"stream aborted: {failure}", flush=True)
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for the resident "
                    "Qwen3.8-27B C/Metal runtime.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--model-dir")
    parser.add_argument("--metallib")
    parser.add_argument("--tokenizer")
    parser.add_argument("--context", type=int,
                        default=int(os.environ.get("QWEN38_CONTEXT",
                                                   4096)))
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.environ.get("QWEN38_MAX_TOKENS",
                                                   3072)))
    parser.add_argument("--thinking", action="store_true",
                        default=os.environ.get("QWEN38_THINKING",
                                               "") not in ("", "0"))
    parser.add_argument("--reasoning-effort",
                        default=os.environ.get("QWEN38_REASONING_EFFORT",
                                               "xhigh"),
                        choices=["low", "medium", "xhigh"])
    parser.add_argument("--temperature", type=float,
                        default=float(os.environ.get("QWEN38_TEMPERATURE",
                                                     0)))
    parser.add_argument("--top-k", type=int,
                        default=int(os.environ.get("QWEN38_TOP_K", 1)))
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("QWEN38_SEED", 42)))
    arguments = parser.parse_args()

    global ENGINE, THINKING_DEFAULT, EFFORT_DEFAULT
    THINKING_DEFAULT = bool(arguments.thinking)
    EFFORT_DEFAULT = arguments.reasoning_effort
    ENGINE = Engine(arguments)
    ENGINE.start()

    # Take the resident engine down with the server: without this a
    # killed server orphans a child process holding the wired model.
    def shutdown(*_):
        process = ENGINE.process if ENGINE else None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    atexit.register(shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    server = ThreadingHTTPServer((arguments.host, arguments.port),
                                 Handler)
    print(f"serving OpenAI-compatible API at "
          f"http://{arguments.host}:{arguments.port}/v1 "
          f"(model id '{MODEL_ID}')", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
