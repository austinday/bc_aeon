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
  * else, if type-ahead is on (agent mid-run) and the line is non-empty -> stash
    it and raise KeyboardInterrupt in the main thread, exactly like Ctrl+C then
    typing a message.

If prompt_toolkit is unavailable it falls back to ``input()`` (readline editing,
which still fixes wrap-aware backspace and the paste-size limit, just without
multi-line paste). Non-TTY stdin (piped, headless ``-n``, a sub-agent) degrades
to a plain ``input()`` with no background thread.
"""
import sys
import time
import queue
import threading
import _thread

_EOF = object()  # sentinel queued when stdin hits end-of-file


class ConsoleInput:
    def __init__(self):
        self._q = queue.Queue()
        self._cond = threading.Condition()
        self._awaiting = False        # a caller is blocking in readline() for a line
        self._typeahead = False       # unsolicited lines interrupt the main thread
        self._prompt = ""             # prompt to show for the pending solicited read
        self._reading = False         # reader thread is currently blocked in _read()
        self._pending = None          # last unsolicited (type-ahead) line
        self._pending_lock = threading.Lock()
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
                    continue
                finally:
                    with self._cond:
                        self._reading = False
                # Classify under the lock so awaiting/typeahead are read atomically.
                # Interrupt the main thread ONLY if type-ahead is STILL on: a run
                # can end (disable_typeahead) while the user is mid-line, and a
                # stray interrupt then would surface as a bare KeyboardInterrupt at
                # the REPL and quietly exit the program.
                with self._cond:
                    if self._awaiting:
                        self._awaiting = False
                        self._prompt = ""
                        deliver, interrupt = True, False
                    else:
                        deliver = False
                        interrupt = self._typeahead and bool(line.strip())
                if deliver:
                    self._q.put(line)
                elif interrupt:
                    with self._pending_lock:
                        self._pending = line
                    _thread.interrupt_main()
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
        if item is _EOF:
            raise EOFError
        return item

    def enable_typeahead(self):
        """While enabled, a line submitted when NO caller is waiting is treated as
        a mid-run interruption (stashed + KeyboardInterrupt raised in main)."""
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
        """Atomically fetch and clear the last unsolicited type-ahead line."""
        with self._pending_lock:
            m = self._pending
            self._pending = None
            return m

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
