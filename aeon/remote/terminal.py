"""PTY-to-WebSocket bridge for attaching a browser to one tmux session."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pty
import re
import signal
import struct
import subprocess
import termios

from fastapi import WebSocket, WebSocketDisconnect

from .instances import InstanceError


_TERMINAL_RESPONSE_PATTERNS = (
    # xterm.js primary and secondary device attributes. Keep the primary reply
    # exact; bound every numeric component of the versioned secondary reply.
    re.compile(r"\x1b\[\?1;2c"),
    re.compile(r"\x1b\[>[0-9]{1,5};[0-9]{1,5};[0-9]{1,5}c"),
    # tmux asks xterm.js for the current foreground and background colors. The
    # pinned browser terminal answers with ST-terminated OSC 10/11 RGB reports.
    re.compile(
        r"\x1b\](?:10|11);rgb:[0-9A-Fa-f]{1,4}/"
        r"[0-9A-Fa-f]{1,4}/[0-9A-Fa-f]{1,4}\x1b\\"
    ),
)


_BARE_LINE_FEED = re.compile(br"(?<!\r)\n")


def _normalize_terminal_snapshot(data: bytes) -> bytes:
    """Give tmux's line-oriented capture normal terminal CRLF semantics.

    ``tmux capture-pane -p`` separates display rows with bare line feeds.  A
    real PTY emits carriage-return/line-feed pairs, and xterm.js deliberately
    preserves the cursor column for a bare line feed when ``convertEol`` is
    disabled.  Normalize only the initial textual snapshot; the attached PTY
    stream remains byte-for-byte authoritative.
    """

    return _BARE_LINE_FEED.sub(b"\r\n", data)


def _is_terminal_response(data: str) -> bool:
    """Recognize only complete, bounded xterm replies to tmux startup probes."""

    return any(
        pattern.fullmatch(data) is not None
        for pattern in _TERMINAL_RESPONSE_PATTERNS
    )


def _resize(fd: int, rows: int, cols: int) -> None:
    rows = max(10, min(int(rows), 300))
    cols = max(20, min(int(cols), 500))
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _resize_attached_client(
    fd: int, client_pid: int, rows: int, cols: int
) -> None:
    """Resize the PTY and notify the exact tmux client that owns it."""

    _resize(fd, rows, cols)
    os.kill(client_pid, signal.SIGWINCH)


async def _write_terminal_response(fd: int, data: str) -> bool:
    """Return one proven terminal report to the read-only tmux attach client."""

    payload = data.encode("ascii")
    loop = asyncio.get_running_loop()
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except BlockingIOError:
            writable = loop.create_future()

            def ready() -> None:
                if not writable.done():
                    writable.set_result(None)

            loop.add_writer(fd, ready)
            try:
                await writable
            finally:
                loop.remove_writer(fd)
            continue
        except OSError:
            return False
        if written <= 0:
            return False
        offset += written
    return True


def _enqueue_terminal_output(
    queue: asyncio.Queue[bytes | None], chunk: bytes | None
) -> None:
    """Bound display buffering while never losing the EOF sentinel."""

    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(chunk)


async def _forward_terminal_input(
    websocket: WebSocket,
    manager,
    instance_id: str,
    data: str,
    expected_generation: int,
) -> bool:
    accepted = await asyncio.to_thread(
        manager.send_terminal_input,
        instance_id,
        data,
        expected_generation=expected_generation,
    )
    if accepted:
        return True
    # The attachment predates a lifecycle generation or tmux could not prove
    # delivery. Close it so queued pre-transition bytes cannot be retried into
    # the new terminal/agent mode.
    await websocket.close(code=1012)
    return False


async def _forward_terminal_scroll(
    websocket: WebSocket,
    manager,
    instance_id: str,
    lines: int,
    expected_generation: int,
) -> bool:
    """Move through tmux history without treating wheel motion as pane input."""

    accepted = await asyncio.to_thread(
        manager.scroll_terminal,
        instance_id,
        lines,
        expected_generation=expected_generation,
    )
    if accepted:
        return True
    await websocket.close(code=1012)
    return False


async def _forward_browser_input(
    websocket: WebSocket,
    manager,
    instance_id: str,
    data: str,
    expected_generation: int,
    attach_fd: int,
) -> bool:
    """Separate browser-generated terminal reports from actual user input."""

    if _is_terminal_response(data):
        accepted = await _write_terminal_response(attach_fd, data)
        if not accepted:
            await websocket.close(code=1011)
        return accepted
    return await _forward_terminal_input(
        websocket,
        manager,
        instance_id,
        data,
        expected_generation,
    )


async def bridge_terminal(
    websocket: WebSocket,
    manager,
    instance_id: str,
    *,
    prepared: tuple[list[str], dict, bytes, int] | None = None,
) -> None:
    if prepared is None:
        # Never block the ASGI event loop behind a lifecycle transition or tmux
        # subprocess. The manager takes one atomic read-only snapshot in a worker.
        prepared = await asyncio.to_thread(
            manager.prepare_terminal_attachment, instance_id
        )
    args, env, initial, attachment_generation = prepared
    if initial:
        await websocket.send_bytes(_normalize_terminal_snapshot(initial))

    master_fd, slave_fd = pty.openpty()
    _resize(master_fd, 36, 120)
    process = None
    loop = asyncio.get_running_loop()
    output_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)

    try:
        process = subprocess.Popen(
            args,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        os.set_blocking(master_fd, False)

        def readable() -> None:
            try:
                chunk = os.read(master_fd, 65536)
            except BlockingIOError:
                return
            except OSError:
                chunk = b""
            if not chunk:
                try:
                    loop.remove_reader(master_fd)
                except Exception:
                    pass
                _enqueue_terminal_output(output_queue, None)
                return
            # A slow phone should not grow host RAM without bound. Drop the
            # oldest display chunk; tmux retains the authoritative scrollback.
            _enqueue_terminal_output(output_queue, chunk)

        loop.add_reader(master_fd, readable)

        async def send_output() -> None:
            while True:
                chunk = await output_queue.get()
                if chunk is None:
                    return
                await websocket.send_bytes(chunk)

        async def receive_input() -> None:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 131072:
                    await websocket.close(code=1009)
                    return
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = message.get("type")
                if kind == "input":
                    data = message.get("data")
                    if isinstance(data, str) and len(data) <= 65536:
                        # Only xterm's exact bounded replies to tmux's startup
                        # probes return to the read-only attach client. Every
                        # user byte still uses a private tmux buffer under the
                        # lifecycle lock, so queued input cannot cross modes and
                        # passwords never appear in process arguments.
                        if not await _forward_browser_input(
                            websocket,
                            manager,
                            instance_id,
                            data,
                            attachment_generation,
                            master_fd,
                        ):
                            return
                elif kind == "resize":
                    try:
                        _resize_attached_client(
                            master_fd,
                            process.pid,
                            message.get("rows", 36),
                            message.get("cols", 120),
                        )
                    except (TypeError, ValueError, OverflowError):
                        continue
                    except OSError:
                        await websocket.close(code=1011)
                        return
                elif kind == "scroll":
                    lines = message.get("lines")
                    if (
                        isinstance(lines, int)
                        and not isinstance(lines, bool)
                        and lines != 0
                        and abs(lines) <= 100
                    ):
                        if not await _forward_terminal_scroll(
                            websocket,
                            manager,
                            instance_id,
                            lines,
                            attachment_generation,
                        ):
                            return
                elif kind == "ping":
                    await websocket.send_json({"type": "pong"})

        sender = asyncio.create_task(send_output())
        receiver = asyncio.create_task(receive_input())
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)
    except (WebSocketDisconnect, InstanceError):
        pass
    finally:
        try:
            loop.remove_reader(master_fd)
        except Exception:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        if slave_fd >= 0:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        if process and process.poll() is None:
            # This process group contains only the tmux attach client. The tmux
            # session and Aeon worker deliberately remain alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
