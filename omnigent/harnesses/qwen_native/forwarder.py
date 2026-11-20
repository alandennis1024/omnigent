"""TUI→web forwarder for the qwen-native harness.

The ``omnigent qwen`` wrapper launches the real ``qwen`` TUI in a runner-owned
tmux pane with ``--json-file`` pointed at the bridge dir, and
:mod:`omnigent.harnesses.qwen_native.bridge` appends web-UI messages to its ``--input-file``.
That covers the web→TUI direction, but the *embedded terminal* is then the only
surface that reflects the agent's work — the Omnigent conversation view stays
empty because nothing mirrors the transcript back into the session.

This module is that missing mirror — the qwen analog of
:mod:`omnigent.harnesses.goose_native.forwarder`. Where goose has to scrape a SQLite store,
qwen emits a structured **stream-json event stream** (verified Anthropic-shaped
against ``qwen`` v0.18.1 and v0.21.14): we tail the ``--json-file`` NDJSON by byte offset and
POST each new ``user`` / ``assistant`` message as an ``external_conversation_item``
event (which also seeds the session title).

Event shapes consumed (others are ignored defensively):

- ``{"type":"user","message":{"role":"user","content":[{"type":"text","text":...}]}}``
- ``{"type":"assistant","message":{"role":"assistant","content":[{"type":"text"|
  "thinking"|"tool_use",...}]}}`` — only ``text`` blocks are mirrored.
- ``{"type":"control_request","request":{"subtype":"can_use_tool",...},
  "request_id":...}`` and the matching ``control_response`` — the permission
  control plane. NOT handled here: the tool-approval mirror
  (:mod:`omnigent.harnesses.qwen_native.permissions`) tails the same stream and surfaces
  these as web elicitation cards. This forwarder ignores them (they carry no
  transcript prose to mirror).

Qwen v0.18 emits a terminating top-level ``result`` record. Qwen v0.21.14 no
longer emits that record; it ends each model invocation with a nested
``stream_event.message_stop``. For v0.21, a ``message_stop`` is terminal only
when the preceding assistant message is not a ``tool_use`` stop — tool-use
cycles continue with tool-result events and another model invocation. The
forwarder posts ``external_session_status`` from either real turn boundary with
the last assistant text, while the runner's PTY watcher remains a read-only UI
signal.

The forwarder also mirrors the **live model + per-turn token usage** off the
same stream so the composer model chip, the context ring, and session cost
reflect qwen's reality instead of staying hidden/zero: assistant message events
carry ``message.model`` (e.g. ``openai/gpt-oss-120b:free``) and ``message.usage``
(``input_tokens`` is cache-inclusive), and ``system/session_start`` carries
model metadata. Usage is accumulated across turns (each model call bills its
own full input, so the counts sum) and POSTed as ``external_session_usage``
cumulative counters + ``context_tokens`` ring numerator; the window denominator
comes from the curated :data:`_QWEN_CONTEXT_WINDOWS` lookup. The model is
POSTed as ``external_model_change``. Both posts are SET-semantics and idempotent,
so a failed POST is simply retried with the same snapshot on a later poll.

This module also hosts the **compaction mirror** (:func:`supervise_qwen_compaction_mirror`).
qwen compaction (its *compression*) is invisible on the ``--json-file`` stream
(``session_start``'s ``supported_events`` omits it — verified live, ``qwen``
v0.18.2), but qwen writes a ``{"type":"system","subtype":"chat_compression",
"systemPayload":{"info":{originalTokenCount,newTokenCount,compressionStatus}}}``
record to its on-disk chat recording the instant compression finishes. The mirror
tails that recording and POSTs ``external_compaction_status`` (``completed`` on
success, ``failed`` otherwise) — the completion half of the web ``/compact`` →
qwen ``/compress`` flow whose ``in_progress`` edge the runner raises on injection.
It fires for both explicit ``/compress`` and auto-compaction.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from omnigent.harnesses.qwen_native.bridge import events_file_path
from omnigent.inner.native_attachments import ATTACHMENT_MARKER_STRIP_PATTERN

_logger = logging.getLogger(__name__)

#: Seconds between event-file polls. qwen flushes events per streaming step, so a
#: sub-second cadence keeps the mirrored chat tracking the terminal step by step.
_DEFAULT_POLL_INTERVAL_S = 0.4
_POST_TIMEOUT_S = 30.0

# Supervisor backoff (mirrors goose_native_forwarder.supervise_goose_forwarder).
_SUPERVISOR_INITIAL_BACKOFF_S = 1.0
_SUPERVISOR_MAX_BACKOFF_S = 30.0
_SUPERVISOR_HEALTHY_UPTIME_S = 60.0

_STATE_FILE = "qwen_forwarder.json"

# Dedup window: the number of most-recently-posted event uuids persisted so a
# truncation/relaunch re-read (offset rewinds to 0) doesn't re-post them. The
# window must keep the *most recent* uuids, so ``seen`` is an insertion-ordered
# mapping (a ``dict`` used as an ordered set), not a ``set`` — ``list(set)`` is
# hash-ordered, which would make the ``[-_DEDUP_WINDOW:]`` cap keep an arbitrary
# subset and re-post recent history on a long-session relaunch.
_DEDUP_WINDOW = 512


def _new_seen(uuids: Iterable[str] | None = None) -> dict[str, None]:
    """Build the insertion-ordered dedup set (``dict`` used as an ordered set)."""
    return dict.fromkeys(uuids or [])


# The executor injects ``[Attached: <path>]`` (or the could-not-load marker
# from native_attachments) for web-UI attachments before submitting; strip them
# from the mirrored bubble (internal bridge details).
_ATTACHMENT_MARKER_RE = re.compile(ATTACHMENT_MARKER_STRIP_PATTERN)


@dataclass
class _ForwardState:
    """Durable forwarder cursor, persisted to ``bridge_dir/qwen_forwarder.json``.

    :param offset: Byte offset into the ``--json-file`` already consumed. The
        event file is append-only within a TUI lifetime; a relaunched terminal
        truncates it (see :func:`~omnigent.harnesses.qwen_native.bridge.prepare_bridge_files`),
        which we detect as ``size < offset`` and reset to 0.
    :param seen_uuids: Recently posted event uuids, for idempotent dedup across a
        truncation/restart. Bounded to the most recent entries.
    :param usage_input_tokens: Cumulative non-cached input tokens reported so
        far (SET semantics — the server monotonic-clamps, so a restart resumes
        safely from the persisted counter).
    :param usage_output_tokens: Cumulative output tokens reported so far.
    :param usage_cache_read_tokens: Cumulative cache-read tokens reported so
        far (the cached portion split out of qwen's cache-inclusive input).
    :param last_context_tokens: Latest observed ``total_tokens`` — the context
        ring numerator.
    :param observed_model: Sticky last live model seen on the stream.
    :param posted_model: Model successfully reported via ``external_model_change``.
    :param posted_usage: Last ``external_session_usage`` snapshot accepted by
        the server; the next report is skipped while unchanged.
    """

    offset: int = 0
    seen_uuids: list[str] | None = None
    last_assistant_text: str = ""
    # ``None`` means no assistant message is awaiting its ``message_stop``;
    # ``""`` represents qwen's JSON null/missing terminal stop reason.
    last_assistant_stop_reason: str | None = None
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    usage_cache_read_tokens: int = 0
    last_context_tokens: int | None = None
    observed_model: str | None = None
    posted_model: str | None = None
    posted_usage: dict[str, object] | None = None


def _read_state(bridge_dir: Path) -> _ForwardState:
    """Load the persisted forward cursor, or a cold default."""
    try:
        raw = (bridge_dir / _STATE_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return _ForwardState(offset=0, seen_uuids=[])
    offset = data.get("offset")
    seen = data.get("seen_uuids")
    posted_usage = data.get("posted_usage")
    return _ForwardState(
        offset=offset if isinstance(offset, int) and offset >= 0 else 0,
        seen_uuids=[u for u in seen if isinstance(u, str)] if isinstance(seen, list) else [],
        last_assistant_text=(
            data.get("last_assistant_text")
            if isinstance(data.get("last_assistant_text"), str)
            else ""
        ),
        last_assistant_stop_reason=(
            data.get("last_assistant_stop_reason")
            if isinstance(data.get("last_assistant_stop_reason"), str)
            else None
        ),
        usage_input_tokens=data.get("usage_input_tokens") if isinstance(data.get("usage_input_tokens"), int) else 0,
        usage_output_tokens=data.get("usage_output_tokens") if isinstance(data.get("usage_output_tokens"), int) else 0,
        usage_cache_read_tokens=data.get("usage_cache_read_tokens") if isinstance(data.get("usage_cache_read_tokens"), int) else 0,
        last_context_tokens=data.get("last_context_tokens") if isinstance(data.get("last_context_tokens"), int) else None,
        observed_model=data.get("observed_model") if isinstance(data.get("observed_model"), str) else None,
        posted_model=data.get("posted_model") if isinstance(data.get("posted_model"), str) else None,
        posted_usage=posted_usage if isinstance(posted_usage, dict) else None,
    )


def _write_state(bridge_dir: Path, state: _ForwardState) -> bool:
    """Atomically persist the forward cursor (tmp write + rename)."""
    try:
        bridge_dir.mkdir(parents=True, exist_ok=True)
        tmp = bridge_dir / (_STATE_FILE + ".tmp")
        # Cap the dedup window so the state file can't grow unbounded. The list
        # is insertion-ordered (see _new_seen), so this keeps the most recent
        # _DEDUP_WINDOW uuids — the ones a relaunch re-read is most likely to hit.
        seen = (state.seen_uuids or [])[-_DEDUP_WINDOW:]
        tmp.write_text(
            json.dumps(
                {
                    "offset": state.offset,
                    "seen_uuids": seen,
                    "last_assistant_text": state.last_assistant_text,
                    "last_assistant_stop_reason": state.last_assistant_stop_reason,
                    "usage_input_tokens": state.usage_input_tokens,
                    "usage_output_tokens": state.usage_output_tokens,
                    "usage_cache_read_tokens": state.usage_cache_read_tokens,
                    "last_context_tokens": state.last_context_tokens,
                    "observed_model": state.observed_model,
                    "posted_model": state.posted_model,
                    "posted_usage": state.posted_usage,
                }
            ),
            encoding="utf-8",
        )
        os.replace(tmp, bridge_dir / _STATE_FILE)
        return True
    except OSError:
        _logger.warning("qwen forwarder could not persist state to %s", bridge_dir, exc_info=True)
        return False


def clear_qwen_bridge_state(bridge_dir: Path) -> None:
    """Remove the persisted forward cursor so a re-created terminal starts clean."""
    with contextlib.suppress(OSError):
        (bridge_dir / _STATE_FILE).unlink()


@dataclass
class _MirrorItem:
    """One conversation item ready to POST, plus the event uuid that produced it."""

    uuid: str
    item_type: str
    item_data: dict[str, object]
    response_id: str


@dataclass
class _TerminalStatus:
    """One authoritative qwen turn-completion record."""

    uuid: str
    status: str
    output: str | None
    response_id: str


def _text_from_content(content: object) -> str:
    """Join the ``text`` blocks of a stream-json message ``content`` array.

    ``thinking`` and ``tool_use`` blocks are skipped — only user-facing prose is
    mirrored into the chat bubble. Tolerant of a bare string or odd shapes so a
    schema tweak degrades to "best available text" rather than dropping the row.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _event_to_item(event: dict[str, object], agent_name: str) -> _MirrorItem | None:
    """Convert one qwen stream-json event to a mirror item, or ``None`` to skip it."""
    etype = event.get("type")
    if etype not in ("user", "assistant"):
        # control_request / control_response (the permission control plane) carry
        # no transcript prose; the tool-approval mirror
        # (omnigent.harnesses.qwen_native.permissions) owns them off the same stream.
        return None
    uuid = event.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    text = _ATTACHMENT_MARKER_RE.sub("", _text_from_content(message.get("content"))).strip()
    if not text:
        return None  # tool-only / thinking-only turn with no prose
    response_id = f"qwen:{uuid}"
    if etype == "user":
        return _MirrorItem(
            uuid=uuid,
            item_type="message",
            item_data={"role": "user", "content": [{"type": "input_text", "text": text}]},
            response_id=response_id,
        )
    return _MirrorItem(
        uuid=uuid,
        item_type="message",
        item_data={
            "role": "assistant",
            "agent": agent_name,
            "content": [{"type": "output_text", "text": text}],
        },
        response_id=response_id,
    )


def _assistant_stop_reason(event: dict[str, object]) -> str | None:
    """Return the stop reason from a top-level assistant event.

    ``None`` means *not an assistant event*. Qwen serializes a successful final
    answer with a JSON null (and older fixtures omit the key), represented here
    as ``""`` so it remains distinguishable from no assistant context.
    """
    if event.get("type") != "assistant":
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    stop_reason = message.get("stop_reason")
    return stop_reason if isinstance(stop_reason, str) else ""


def _event_to_terminal(
    event: dict[str, object],
    last_assistant_text: str,
    last_assistant_stop_reason: str | None = None,
) -> _TerminalStatus | None:
    """Convert qwen's version-specific terminating record to session status."""
    event_type = event.get("type")
    if event_type == "result":
        # Qwen's result builder permits caller-defined success subtypes;
        # is_error is the protocol's authoritative discriminator.
        failed = event.get("is_error") is True
        result = event.get("result")
        result_text = result.strip() if isinstance(result, str) else ""
        output = result_text or last_assistant_text or None
    elif event_type == "stream_event":
        stream_event = event.get("event")
        if (
            not isinstance(stream_event, dict)
            or stream_event.get("type") != "message_stop"
            or last_assistant_stop_reason is None
            or last_assistant_stop_reason == "tool_use"
        ):
            return None
        # message_stop carries no error channel: a v0.21 turn that dies is
        # surfaced by the runner's terminal-exit watcher, not this edge.
        failed = False
        output = last_assistant_text or None
    else:
        return None
    uuid = event.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        return None
    return _TerminalStatus(
        uuid=uuid,
        status="failed" if failed else "idle",
        output=output,
        response_id=f"qwen:{uuid}",
    )


# --- live model + token usage mirror (model chip / context ring / cost) --------

#: Curated context windows (tokens) for models qwen drives, matched by
#: substring (lowercased, longest/most-specific token first). Windows follow
#: the providers' published limits (Alibaba Model Studio qwen3-coder family,
#: OpenRouter gpt-oss). Anything unrecognized falls back to
#: :data:`_QWEN_DEFAULT_CONTEXT_WINDOW` — the same figure the bridge's synth
#: session advertises — so the ring renders instead of staying hidden.
_QWEN_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("qwen3-coder-plus", 1_000_000),
    ("qwen3-coder-flash", 1_000_000),
    ("qwen3-coder", 262_144),
    ("qwen-max", 131_072),
    ("qwen-plus", 131_072),
    ("qwen-turbo", 1_000_000),
    ("gpt-oss-120b", 131_072),
    ("gpt-oss-20b", 131_072),
)
_QWEN_DEFAULT_CONTEXT_WINDOW = 131_072


def _context_window_for(model: str | None) -> int:
    """Resolve the context window for a live qwen model id."""
    if model:
        lowered = model.lower()
        for token, window in _QWEN_CONTEXT_WINDOWS:
            if token in lowered:
                return window
    return _QWEN_DEFAULT_CONTEXT_WINDOW


def _int_token(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _normalize_qwen_usage(raw: dict[str, object]) -> dict[str, int]:
    """Normalize one ``message.usage`` block for accumulation and reporting.

    Qwen's ``input_tokens`` is cache-inclusive, but the server's cost
    computation prices ``input_tokens`` at the full rate and
    ``cache_read_input_tokens`` at the cache rate — so the cached portion is
    split out of the input total (mirroring the ACP qwen executor).
    """
    cached = _int_token(raw.get("cache_read_input_tokens"))
    return {
        "input_tokens": max(0, _int_token(raw.get("input_tokens")) - cached),
        "output_tokens": _int_token(raw.get("output_tokens")),
        "cache_read_input_tokens": cached,
        "total_tokens": _int_token(raw.get("total_tokens")),
    }


def _usage_and_model_from_event(event: dict[str, object]) -> tuple[dict[str, int] | None, str | None]:
    """Extract per-call usage + the live model from one stream-json event.

    Returns ``(usage, model)`` with either side ``None`` when the event
    doesn't carry it. Only assistant and system events contribute: assistant
    messages carry ``message.usage`` + ``message.model`` (verified qwen
    v0.18.2), ``system/session_start`` carries model metadata (v0.18 wraps the
    payload in ``data``).
    """
    etype = event.get("type")
    usage: dict[str, int] | None = None
    model: str | None = None
    if etype == "assistant":
        message = event.get("message")
        if isinstance(message, dict):
            raw_usage = message.get("usage")
            if isinstance(raw_usage, dict):
                usage = _normalize_qwen_usage(raw_usage)
            raw_model = message.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()
    elif etype == "system":
        payload = event.get("data") if isinstance(event.get("data"), dict) else event
        raw_model = payload.get("model")
        if not (isinstance(raw_model, str) and raw_model.strip()):
            models = payload.get("models")
            raw_model = next(
                (m for m in models if isinstance(m, str) and m.strip())
                if isinstance(models, list)
                else (),
                None,
            )
        if isinstance(raw_model, str) and raw_model.strip():
            model = raw_model.strip()
    return usage, model


def _read_new_forward_events(
    events_file: Path,
    offset: int,
    seen: Collection[str],
    agent_name: str,
    last_assistant_text: str,
    last_assistant_stop_reason: str | None = None,
) -> tuple[
    list[_MirrorItem | _TerminalStatus],
    int,
    str,
    str | None,
    list[tuple[dict[str, int] | None, str | None]],
]:
    """Read mirror items, terminal records and usage/model observations past *offset*.

    The usage/model observations are returned per consumed event in wire order
    (``(usage, model)`` with either side ``None``) and are folded into the
    cumulative report separately from the item dedup: they aggregate even for
    events that produce no mirror item (tool-only turns still bill tokens).
    """
    try:
        size = events_file.stat().st_size
    except OSError:
        return [], offset, last_assistant_text, last_assistant_stop_reason, []
    if size < offset:
        offset = 0
        last_assistant_text = ""
        last_assistant_stop_reason = None
    if size == offset:
        return [], offset, last_assistant_text, last_assistant_stop_reason, []
    try:
        with open(events_file, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size - offset)
    except OSError:
        return [], offset, last_assistant_text, last_assistant_stop_reason, []
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset, last_assistant_text, last_assistant_stop_reason, []
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    actions: list[_MirrorItem | _TerminalStatus] = []
    observations: list[tuple[dict[str, int] | None, str | None]] = []
    encountered = set(seen)
    for raw in consumed.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "user":
            last_assistant_stop_reason = None
        assistant_stop_reason = _assistant_stop_reason(event)
        if assistant_stop_reason is not None:
            last_assistant_stop_reason = assistant_stop_reason
        usage, model = _usage_and_model_from_event(event)
        if usage is not None or model is not None:
            observations.append((usage, model))
        item = _event_to_item(event, agent_name)
        if item is not None and item.uuid not in encountered:
            actions.append(item)
            encountered.add(item.uuid)
            last_assistant_text = _assistant_text_after_action(last_assistant_text, item)
            continue
        terminal = _event_to_terminal(event, last_assistant_text, last_assistant_stop_reason)
        if terminal is not None and terminal.uuid not in encountered:
            actions.append(terminal)
            encountered.add(terminal.uuid)
            last_assistant_text = ""
            last_assistant_stop_reason = None
    return actions, new_offset, last_assistant_text, last_assistant_stop_reason, observations


def _assistant_text_after_action(current: str, action: _MirrorItem | _TerminalStatus) -> str:
    """Return the assistant buffer after one successfully delivered action."""
    if isinstance(action, _TerminalStatus):
        return ""
    role = action.item_data.get("role")
    if role == "user":
        return ""
    if role != "assistant":
        return current
    content = action.item_data.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return current
    text = content[0].get("text")
    return text if isinstance(text, str) else current


async def _post_conversation_item(
    client: httpx.AsyncClient, *, session_id: str, item: _MirrorItem
) -> None:
    """POST one mirrored item as an ``external_conversation_item`` event."""
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={
            "type": "external_conversation_item",
            "data": {
                "item_type": item.item_type,
                "item_data": item.item_data,
                "response_id": item.response_id,
            },
        },
    )
    resp.raise_for_status()


async def _post_external_session_status(
    client: httpx.AsyncClient, *, session_id: str, terminal: _TerminalStatus
) -> None:
    """POST one authoritative qwen terminal edge."""
    data: dict[str, object] = {
        "status": terminal.status,
        "response_id": terminal.response_id,
    }
    if terminal.output is not None:
        data["output"] = terminal.output
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_status", "data": data},
    )
    resp.raise_for_status()


async def _post_external_session_usage(
    client: httpx.AsyncClient, *, session_id: str, state: _ForwardState
) -> None:
    """POST the cumulative usage + context-window snapshot; skip while unchanged."""
    if state.last_context_tokens is None and state.observed_model is None:
        return  # nothing observed yet — nothing the server would accept
    snapshot: dict[str, object] = {
        "cumulative_input_tokens": state.usage_input_tokens,
        "cumulative_output_tokens": state.usage_output_tokens,
        "cumulative_cache_read_input_tokens": state.usage_cache_read_tokens,
        "context_tokens": state.last_context_tokens,
        "context_window": _context_window_for(state.observed_model),
    }
    if state.observed_model is not None:
        snapshot["model"] = state.observed_model
    if snapshot == state.posted_usage:
        return
    data = {key: value for key, value in snapshot.items() if value is not None}
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_session_usage", "data": data},
    )
    resp.raise_for_status()
    state.posted_usage = snapshot


async def _post_external_model_change(
    client: httpx.AsyncClient, *, session_id: str, model: str
) -> None:
    """POST one ``external_model_change`` so surfaces show the pane's live model."""
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_model_change", "data": {"model": model}},
    )
    resp.raise_for_status()


async def _report_usage_and_model(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    state: _ForwardState,
    observations: list[tuple[dict[str, int] | None, str | None]],
) -> None:
    """Fold one batch's usage/model observations into the report and POST it.

    Usage accumulates across observations (each model call bills its own full
    input). Posts are best-effort and idempotent — a failure leaves
    ``posted_usage`` / ``posted_model`` behind so the next poll retries with
    the same SET-semantics snapshot, and the counters are persisted by the
    caller even after a failure so the retry can't double-count.
    """
    for usage, model in observations:
        if usage is not None:
            state.usage_input_tokens += usage["input_tokens"]
            state.usage_output_tokens += usage["output_tokens"]
            state.usage_cache_read_tokens += usage["cache_read_input_tokens"]
            state.last_context_tokens = usage["total_tokens"]
        if model is not None:
            state.observed_model = model
    try:
        await _post_external_session_usage(client, session_id=session_id, state=state)
    except httpx.HTTPError:
        _logger.warning(
            "qwen forwarder usage report failed; session=%s", session_id, exc_info=True
        )
    if state.observed_model is not None and state.observed_model != state.posted_model:
        try:
            await _post_external_model_change(
                client, session_id=session_id, model=state.observed_model
            )
        except httpx.HTTPError:
            _logger.warning(
                "qwen forwarder model report failed; session=%s", session_id, exc_info=True
            )
        else:
            state.posted_model = state.observed_model


async def _deliver_forward_actions(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    bridge_dir: Path,
    state: _ForwardState,
    actions: list[_MirrorItem | _TerminalStatus],
) -> None:
    """Deliver actions in order and durably commit each successful POST."""
    seen = _new_seen(state.seen_uuids)
    for action in actions:
        if isinstance(action, _MirrorItem):
            await _post_conversation_item(client, session_id=session_id, item=action)
        else:
            await _post_external_session_status(client, session_id=session_id, terminal=action)
        seen[action.uuid] = None
        state.seen_uuids = list(seen)
        state.last_assistant_text = _assistant_text_after_action(state.last_assistant_text, action)
        # Keep the prior offset until every action from this read is delivered.
        # On a later POST failure, retry re-reads the batch but skips each
        # already-delivered uuid while retaining the assistant output.
        _write_state(bridge_dir, state)


async def forward_qwen_events_to_session(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    events_file: Path | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    auth: httpx.Auth | None = None,
) -> None:
    """Tail qwen's ``--json-file`` and mirror new messages into the AP session.

    Polls the event file past a persisted byte offset, posting each new
    user/assistant message as an ``external_conversation_item``. The offset +
    dedup set are persisted to ``bridge_dir`` so a supervisor restart resumes
    without re-posting.

    :param base_url: Omnigent server base URL.
    :param headers: Static HTTP headers (auth normally via ``auth``).
    :param session_id: Omnigent session/conversation id.
    :param bridge_dir: The qwen-native bridge dir (holds the persisted cursor).
    :param agent_name: Agent label stamped on mirrored assistant items.
    :param events_file: qwen ``--json-file`` path; defaults to the bridge dir's.
    :param poll_interval_s: Seconds between event-file polls.
    :param auth: Optional refresh-capable httpx Auth for remote deployments.
    :returns: Never normally returns; cancel the task to stop it.
    """
    target = events_file or events_file_path(bridge_dir)
    state = _read_state(bridge_dir)
    timeout = httpx.Timeout(_POST_TIMEOUT_S)
    from omnigent.cli_auth import open_server_client

    async with open_server_client(base_url, headers=headers, auth=auth, timeout=timeout) as client:
        while True:
            try:
                (
                    actions,
                    new_offset,
                    next_assistant_text,
                    next_assistant_stop_reason,
                    observations,
                ) = await asyncio.to_thread(
                    _read_new_forward_events,
                    target,
                    state.offset,
                    _new_seen(state.seen_uuids),
                    agent_name,
                    state.last_assistant_text,
                    state.last_assistant_stop_reason,
                )
                await _deliver_forward_actions(
                    client,
                    session_id=session_id,
                    bridge_dir=bridge_dir,
                    state=state,
                    actions=actions,
                )
                await _report_usage_and_model(
                    client, session_id=session_id, state=state, observations=observations
                )
                if new_offset != state.offset or actions:
                    state.offset = new_offset
                    state.last_assistant_text = next_assistant_text
                    state.last_assistant_stop_reason = next_assistant_stop_reason
                    _write_state(bridge_dir, state)
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.exception(
                    "qwen forwarder poll failed; session=%s bridge_dir=%s",
                    session_id,
                    bridge_dir,
                )
            await asyncio.sleep(poll_interval_s)


def _supervisor_monotonic() -> float:
    """Indirection so tests can stub the supervisor's clock."""
    return time.monotonic()


async def _supervisor_sleep(seconds: float) -> None:
    """Indirection so tests can stub the supervisor's backoff sleep."""
    await asyncio.sleep(seconds)


async def supervise_qwen_forwarder(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    bridge_dir: Path,
    agent_name: str,
    events_file: Path | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    auth: httpx.Auth | None = None,
) -> None:
    """Run :func:`forward_qwen_events_to_session` under a restart supervisor.

    Mirrors :func:`omnigent.harnesses.goose_native.forwarder.supervise_goose_forwarder`:
    bounded exponential backoff, :class:`asyncio.CancelledError` propagates for
    clean teardown, and the persisted offset means restarts resume exactly where
    they left off.

    :returns: Never normally returns; cancel the task to stop it.
    """
    backoff_s = _SUPERVISOR_INITIAL_BACKOFF_S
    while True:
        run_started_at = _supervisor_monotonic()
        crash_exc: Exception | None = None
        try:
            await forward_qwen_events_to_session(
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                bridge_dir=bridge_dir,
                agent_name=agent_name,
                events_file=events_file,
                poll_interval_s=poll_interval_s,
                auth=auth,
            )
            _logger.warning(
                "qwen forwarder returned unexpectedly; restarting; session=%s bridge_dir=%s",
                session_id,
                bridge_dir,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — supervisor restarts on any Exception
            crash_exc = exc
        if _supervisor_monotonic() - run_started_at >= _SUPERVISOR_HEALTHY_UPTIME_S:
            backoff_s = _SUPERVISOR_INITIAL_BACKOFF_S
        if crash_exc is not None:
            _logger.error(
                "qwen forwarder crashed; restarting in %.1fs; session=%s bridge_dir=%s",
                backoff_s,
                session_id,
                bridge_dir,
                exc_info=crash_exc,
            )
        await _supervisor_sleep(backoff_s)
        backoff_s = min(backoff_s * 2.0, _SUPERVISOR_MAX_BACKOFF_S)


# --- Compaction mirror (chat-recording tail → external_compaction_status) ------

#: qwen's CompressionStatus enum (verified, qwen v0.18.2): 1 = COMPRESSED (success),
#: 2/3 = COMPRESSION_FAILED_*. 1 → completed; anything else → failed.
_COMPRESSION_STATUS_OK = 1


def _compaction_status_from_record(record: dict[str, object]) -> str | None:
    """Map a chat-recording line to a compaction status, or ``None`` to skip it.

    Returns ``"completed"`` for a successful ``chat_compression`` record,
    ``"failed"`` for a failed one, and ``None`` for any other line.
    """
    if record.get("type") != "system" or record.get("subtype") != "chat_compression":
        return None
    payload = record.get("systemPayload")
    info = payload.get("info") if isinstance(payload, dict) else None
    status = info.get("compressionStatus") if isinstance(info, dict) else None
    return "completed" if status == _COMPRESSION_STATUS_OK else "failed"


def _read_new_compaction_statuses(recording: Path, offset: int) -> tuple[list[str], int]:
    """Read NDJSON lines past *offset*, returning new compaction statuses + offset.

    Same tail discipline as :func:`_read_new_forward_events` (truncation rewind, only
    newline-terminated lines consumed), but scoped to ``chat_compression`` records.
    """
    try:
        size = recording.stat().st_size
    except OSError:
        return [], offset  # recording not created yet — retry next poll
    if size < offset:
        offset = 0  # truncated/recreated
    if size == offset:
        return [], offset
    try:
        with open(recording, "rb") as fh:
            fh.seek(offset)
            data = fh.read(size - offset)
    except OSError:
        return [], offset
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    statuses: list[str] = []
    for raw in consumed.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        status = _compaction_status_from_record(record)
        if status is not None:
            statuses.append(status)
    return statuses, new_offset


async def _post_external_compaction_status(
    client: httpx.AsyncClient, *, session_id: str, status: str
) -> None:
    """POST one ``external_compaction_status`` event; the server republishes the SSE."""
    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "external_compaction_status", "data": {"status": status}},
    )
    resp.raise_for_status()


async def supervise_qwen_compaction_mirror(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    recording_path: Path,
    auth: httpx.Auth | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Tail qwen's chat recording and mirror compaction completions to the session.

    Seeds the read offset at the recording's current end of file so only
    compactions that happen *after* launch are posted — a resumed session's
    recording already holds prior ``chat_compression`` records, and re-posting them
    would flash stale "Conversation compacted" dividers. Self-healing: any error is
    logged and the loop continues (a transient blip never abandons the mirror);
    cancellation propagates for clean teardown. Best-effort, like the approval
    mirror — the offset is in-memory, so a forwarder restart may miss a compaction
    that lands during the gap, which only drops one divider.

    :param recording_path: qwen's chat recording for this session (see
        :func:`omnigent.harnesses.qwen_native.bridge.qwen_session_recording_path`).
    """
    try:
        offset = recording_path.stat().st_size
    except OSError:
        offset = 0  # not created yet; first poll reads from the start
    timeout = httpx.Timeout(_POST_TIMEOUT_S)
    from omnigent.cli_auth import open_server_client

    async with open_server_client(base_url, headers=headers, auth=auth, timeout=timeout) as client:
        while True:
            try:
                statuses, offset = await asyncio.to_thread(
                    _read_new_compaction_statuses, recording_path, offset
                )
                for status in statuses:
                    await _post_external_compaction_status(
                        client, session_id=session_id, status=status
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                _logger.exception(
                    "qwen compaction mirror poll failed; session=%s recording=%s",
                    session_id,
                    recording_path,
                )
            await asyncio.sleep(poll_interval_s)
