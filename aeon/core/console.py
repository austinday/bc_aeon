"""Single owner of terminal input for an interactive session.

Why this exists
---------------
The terminal's default (canonical) line mode gives a poor editor: you can't
backspace across a line that wrapped to a second screen row, and a paste longer
than the ~4 KB canonical buffer is silently truncated. So every interactive read
— the REPL prompt, ``get_user_input``, the Ctrl+C guidance prompt, AND the
mid-run type-ahead the agent can be interrupted with — goes through ONE reader
here, built on prompt_toolkit: full line editing, arrow keys/history, unlimited
bracketed paste, and MULTI-LINE paste kept as one input.

A single background reader thread performs the reads (via prompt_toolkit) so the
user can type WHILE the agent works; ``patch_stdout`` renders the agent's output
cleanly above the live input line. Each submitted line is classified at SUBMIT
time by the state current then:

  * a caller is waiting for a line (REPL / get_user_input / guidance) -> deliver;
  * else, if a previously started terminal read returns a non-empty line ->
    append it to a FIFO for the REPL to consume as a later turn;
  * the private Nexus stop control alone interrupts the active turn, without
    terminating the Aeon process or discarding that FIFO.

If prompt_toolkit is unavailable it falls back to ``input()`` (readline editing,
which still fixes wrap-aware backspace and the paste-size limit, just without
multi-line paste). Non-TTY stdin (piped, headless ``-n``, a sub-agent) degrades
to a plain ``input()`` with no background thread.
"""
import _thread
import queue
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

from aeon.core.continuous_mode import NEXUS_CONTINUOUS_WAKE_COMMAND

_EOF = object()  # sentinel queued when stdin hits end-of-file
_STOP = object()  # sentinel delivered when Nexus requests a turn-only stop

# Nexus sends this exact private control line through the managed pane.  It is a
# turn-control message, never a user objective and never part of the transcript.
NEXUS_STOP_TURN_COMMAND = "/__nexus_stop_current_turn_7f30a9c2__"


class TurnStopRequested(KeyboardInterrupt):
    """The active objective should yield without terminating the Aeon process."""


class ConsoleInput:
    def __init__(self):
        self._q = queue.Queue()
        self._cond = threading.Condition()
        self._awaiting = False        # a caller is blocking in readline() for a line
        self._typeahead = False       # unsolicited lines may be queued while a run is active
        self._prompt = ""             # prompt to show for the pending solicited read
        self._reading = False         # reader thread is currently blocked in _read()
        self._pending = deque()       # FIFO of unsolicited lines for later turns
        self._pending_controls = deque()  # server-built wake signals, never user turns
        self._pending_lock = threading.Lock()
        self._stop_requested = False  # consumed by the active Worker interrupt
        self._interruptible_depth = 0  # only model calls may receive interrupt_main()
        self._started = False
        self._tty = self._is_tty()
        self._session = None          # prompt_toolkit PromptSession, lazily built
        self._use_pt = False          # whether prompt_toolkit is available/working
        if self._tty:
            # readline is the fallback editor (and serves the model picker's bare
            # input()); loading it is harmless when prompt_toolkit is present.
            try:
                import readline  # noqa: F401
            except Exception:
                pass

    @staticmethod
    def _is_tty():
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except Exception:
            return False

    def _ensure_thread(self):
        if self._started or not self._tty:
            return
        self._started = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _read(self, prompt):
        """Read one line, blocking, on the reader thread. Uses prompt_toolkit when
        available (multi-line paste, unlimited length), else input() (readline)."""
        if self._use_pt:
            from prompt_toolkit.formatted_text import ANSI
            # ANSI() so colored/multi-line prompt strings render, not print literally.
            return self._session.prompt(ANSI(prompt) if prompt else "")
        return input(prompt)

    def _dispatch_line(self, line):
        """Classify one complete input line and wake only the intended consumer."""

        # Managed browser/portal messages cross tmux before their transcript
        # append. Do not expose that transport ordering to the Worker: an exact
        # private envelope becomes user input only after its durable record can
        # be read back. Failed/ambiguous appends are dropped rather than acted on.
        from aeon.core.chat_transcript import committed_chat_delivery_from_environment

        line = committed_chat_delivery_from_environment(line)
        if line is None:
            return

        queue_for_later = False
        queue_control = False
        interrupt = False
        # Classify under the lock so awaiting/typeahead/interruptible state is
        # observed atomically. Keep the lock through interrupt_main(): leaving an
        # interruptible model-call scope needs this same lock, so the main thread
        # cannot advance into tool execution before the stop signal is issued.
        with self._cond:
            stop = line.strip() == NEXUS_STOP_TURN_COMMAND
            continuous_wake = line.strip() == NEXUS_CONTINUOUS_WAKE_COMMAND
            if stop:
                was_awaiting = self._awaiting
                self._awaiting = False
                self._prompt = ""
                self._stop_requested = True
                deliver = was_awaiting
                interrupt = (
                    not was_awaiting
                    and self._typeahead
                    and self._interruptible_depth > 0
                )
            elif continuous_wake:
                if self._awaiting:
                    self._awaiting = False
                    self._prompt = ""
                    deliver = True
                else:
                    deliver = False
                    # A type-ahead read can complete just after a Worker run
                    # disables its listener. Preserve the private wake in its
                    # separate control queue across that hand-off window.
                    queue_control = True
            elif self._awaiting:
                self._awaiting = False
                self._prompt = ""
                deliver = True
            else:
                deliver = False
                # The reader may already be blocked in _read() when type-ahead
                # is disabled between Worker.run() and the next REPL/continuous
                # decision. Every complete non-empty line from that owned read
                # must remain FIFO input; otherwise Nexus can report a durable
                # PTY delivery while Aeon silently discards it.
                queue_for_later = bool(line.strip())
            if interrupt:
                _thread.interrupt_main()

        if stop and deliver:
            self._q.put(_STOP)
        elif deliver:
            self._q.put(line)
        elif queue_for_later:
            with self._pending_lock:
                self._pending.append(line)
        elif queue_control:
            with self._pending_lock:
                self._pending_controls.append(line)

    def _loop(self):
        # Prefer prompt_toolkit; fall back to plain input() if it can't load.
        patch = None
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout
            self._session = PromptSession()
            self._use_pt = True
            # raw=True is REQUIRED: the default StdoutProxy writes app output via
            # Vt100_Output.write(), which sanitizes ESC (\x1b -> '?') to stop
            # untrusted text from injecting control codes. Aeon's own output is
            # heavily ANSI-colored and trusted, so without raw=True every color
            # code renders as a literal "?[0m". raw=True uses write_raw(), which
            # preserves the escape bytes so the terminal colors the text.
            patch = patch_stdout(raw=True)
            patch.__enter__()   # route all stdout writes above the live prompt
        except Exception:
            self._use_pt = False
        try:
            while True:
                # Park until there is a reason to read: a caller is waiting, or the
                # agent is running and wants type-ahead. Parking (rather than
                # reading eagerly) lets a solicited read show its prompt correctly.
                with self._cond:
                    while not (self._awaiting or self._typeahead):
                        self._cond.wait()
                    prompt = self._prompt if self._awaiting else ""
                    self._reading = True
                try:
                    line = self._read(prompt)
                except EOFError:
                    with self._cond:
                        was_await = self._awaiting
                        self._awaiting = False
                    if was_await:
                        self._q.put(_EOF)
                    else:
                        # Unsolicited read hit EOF. On a live TTY the next read
                        # blocks for input; but if stdin is genuinely closed this
                        # would busy-loop, so back off before trying again.
                        time.sleep(0.2)
                    continue
                except KeyboardInterrupt:
                    # prompt_toolkit runs the terminal in raw mode with signals off,
                    # so Ctrl+C surfaces HERE (in the reader) instead of as SIGINT to
                    # the main thread. Re-dispatch it to preserve Ctrl+C semantics:
                    #   * during a solicited read -> unblock the caller as EOF (the
                    #     REPL treats that as quit, like the old Ctrl+C at '> ');
                    #   * during a run (type-ahead) -> pause the agent exactly as a
                    #     bare Ctrl+C used to (the main loop then prompts for guidance).
                    with self._cond:
                        was_await = self._awaiting
                        typeahead = self._typeahead
                        self._awaiting = False
                    if was_await:
                        self._q.put(_EOF)
                    elif typeahead:
                        _thread.interrupt_main()
                    continue
                except Exception:
                    # A broken prompt-toolkit terminal or closed input stream
                    # can fail persistently. Fall back to the simpler reader and
                    # back off so the daemon thread cannot hot-loop at 100% CPU
                    # while delivering no user input.
                    self._use_pt = False
                    time.sleep(0.2)
                    continue
                finally:
                    with self._cond:
                        self._reading = False
                # Ordinary type-ahead is queued FIFO. The private stop control
                # may interrupt only an explicitly scoped model call.
                self._dispatch_line(line)
        finally:
            if patch is not None:
                try:
                    patch.__exit__(None, None, None)
                except Exception:
                    pass

    # ---- public API -------------------------------------------------------
    def readline(self, prompt=""):
        """Block until the user submits a line; return it. Raises EOFError at EOF.
        On non-TTY stdin this is a plain input()."""
        if not self._tty:
            return input(prompt)
        self._ensure_thread()
        # A message submitted during the preceding turn is already a complete,
        # edited line. Consume it before asking the reader thread for new input.
        with self._pending_lock:
            if self._pending:
                return self._pending.popleft()
            if self._pending_controls:
                return self._pending_controls.popleft()
        with self._cond:
            if self._stop_requested and not self._typeahead:
                raise TurnStopRequested
        with self._cond:
            self._prompt = prompt
            self._awaiting = True
            # If the reader is ALREADY blocked in a bare type-ahead read (the
            # normal state mid-run), it can't show this prompt — the read began
            # before the prompt existed. Print it here so the user actually sees
            # what is being asked (get_user_input, the Ctrl+C guidance prompt);
            # the typed line is still classified correctly at submit time.
            show_now = self._reading and bool(prompt)
            self._cond.notify_all()
        if show_now:
            print(prompt, end="", flush=True)
        item = self._q.get()
        if item is _STOP:
            raise TurnStopRequested
        if item is _EOF:
            raise EOFError
        return item

    def enable_typeahead(self):
        """Queue complete unsolicited lines while an agent run is active."""
        if not self._tty:
            return
        self._ensure_thread()
        with self._cond:
            self._typeahead = True
            self._cond.notify_all()

    def disable_typeahead(self):
        with self._cond:
            self._typeahead = False

    def take_pending(self):
        """Atomically fetch the oldest queued unsolicited line."""
        with self._pending_lock:
            return self._pending.popleft() if self._pending else None

    def has_pending(self):
        """Return whether a complete unsolicited user message is queued.

        Workers use this as a non-consuming interruption boundary immediately
        before a mutation. The REPL remains the sole consumer, so the exact user
        text becomes the next user-role turn instead of being rewritten or lost.
        """
        with self._pending_lock:
            return bool(self._pending)

    @contextmanager
    def interruptible(self):
        """Allow the private stop control to interrupt only this model call."""

        with self._cond:
            if self._stop_requested:
                raise TurnStopRequested
            self._interruptible_depth += 1
        try:
            yield
        finally:
            with self._cond:
                self._interruptible_depth = max(0, self._interruptible_depth - 1)

    def has_stop_request(self):
        """Return whether Nexus requested a cooperative turn stop."""

        with self._cond:
            return self._stop_requested

    def take_stop_request(self):
        """Return and clear the process-local turn-stop marker."""
        with self._cond:
            requested = self._stop_requested
            self._stop_requested = False
            return requested

    @property
    def active(self):
        return self._tty


_console = None


def console():
    """Process-wide singleton."""
    global _console
    if _console is None:
        _console = ConsoleInput()
    return _console
