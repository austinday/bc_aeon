(() => {
  "use strict";

  const state = {
    csrf: "",
    username: "",
    instances: [],
    activeId: null,
    workspaces: { roots: [], workspaces: [] },
    terminal: null,
    fit: null,
    socket: null,
    terminalDisposables: [],
    terminalGeneration: 0,
    reconnectTimer: null,
    reconnectAttempts: 0,
  };

  const MAX_RECONNECT_ATTEMPTS = 6;

  const $ = (id) => document.getElementById(id);
  const loginView = $("login-view");
  const appView = $("app-view");
  const loginError = $("login-error");
  const newDialog = $("new-dialog");

  function setVisible(element, visible) {
    element.classList.toggle("hidden", !visible);
  }

  function errorText(error) {
    if (error && error.message) return error.message;
    return String(error || "Something went wrong");
  }

  async function api(path, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    if (!["GET", "HEAD", "OPTIONS"].includes(method) && state.csrf) {
      headers.set("X-CSRF-Token", state.csrf);
    }
    const response = await fetch(path, {
      ...options,
      method,
      headers,
      credentials: "same-origin",
      cache: "no-store",
    });
    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }
    if (response.status === 401) {
      showLogin();
      throw new Error("Your session expired. Sign in again.");
    }
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    return payload;
  }

  function showLogin() {
    disconnectTerminal();
    state.csrf = "";
    loginView.classList.remove("hidden");
    appView.classList.add("hidden");
    setTimeout(() => $("login-username").focus(), 0);
  }

  async function showApp(session) {
    state.csrf = session.csrf_token;
    state.username = session.username;
    loginView.classList.add("hidden");
    appView.classList.remove("hidden");
    $("connection-label").textContent = `Connected as ${state.username}`;
    await Promise.all([refreshWorkspaces(), refreshInstances(), refreshResources()]);
  }

  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    loginError.textContent = "";
    const submit = event.submitter;
    if (submit) submit.disabled = true;
    try {
      const session = await api("/api/login", {
        method: "POST",
        body: JSON.stringify({
          username: $("login-username").value,
          password: $("login-password").value,
          otp: $("login-otp").value,
          remember: $("login-remember").checked,
        }),
      });
      $("login-password").value = "";
      $("login-otp").value = "";
      await showApp(session);
    } catch (error) {
      loginError.textContent = errorText(error);
    } finally {
      if (submit) submit.disabled = false;
    }
  });

  $("logout").addEventListener("click", async () => {
    try {
      await api("/api/logout", { method: "POST" });
    } catch (_) {
      // Local state must still be cleared if the network disappeared.
    }
    showLogin();
  });

  function statusClass(status) {
    return ["running", "starting", "stopping", "error", "interrupted"].includes(status)
      ? status
      : "neutral";
  }

  function renderInstances() {
    const list = $("instance-list");
    list.replaceChildren();
    if (!state.instances.length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "No agent terminals yet.";
      list.append(note);
    }
    for (const instance of state.instances) {
      const button = document.createElement("button");
      button.className = "instance-item";
      if (instance.id === state.activeId) button.classList.add("active");
      button.type = "button";
      button.addEventListener("click", () => selectInstance(instance.id));

      const dot = document.createElement("span");
      dot.className = `status-dot ${statusClass(instance.status)}`;
      const copy = document.createElement("span");
      copy.className = "instance-copy";
      const name = document.createElement("strong");
      name.textContent = instance.name;
      const detail = document.createElement("small");
      const memory = instance.resources
        ? ` · ${formatBytes(instance.resources.rss_bytes)} RAM`
        : "";
      detail.textContent = `${instance.status}${memory}`;
      copy.append(name, detail);
      button.append(dot, copy);
      list.append(button);
    }
  }

  async function refreshInstances() {
    try {
      const payload = await api("/api/instances");
      state.instances = payload.instances || [];
      if (
        state.activeId &&
        !state.instances.some((instance) => instance.id === state.activeId)
      ) {
        state.activeId = null;
        disconnectTerminal();
      }
      renderInstances();
      renderActiveHeader();
    } catch (error) {
      $("connection-label").textContent = errorText(error);
    }
  }

  $("refresh-instances").addEventListener("click", refreshInstances);

  function activeInstance() {
    return state.instances.find((instance) => instance.id === state.activeId) || null;
  }

  function renderActiveHeader() {
    const instance = activeInstance();
    const empty = $("terminal-empty");
    const terminal = $("terminal-wrap");
    const mobileInput = $("mobile-input");
    if (!instance) {
      $("terminal-name").textContent = "No agent selected";
      $("terminal-status").textContent = "idle";
      $("terminal-status").className = "status neutral";
      $("terminal-workspace").textContent = "";
      setVisible(empty, true);
      setVisible(terminal, false);
      setVisible(mobileInput, false);
      for (const id of [
        "activate-aeon",
        "end-agent",
        "resume-instance",
        "stop-instance",
        "force-instance",
        "delete-instance",
      ]) setVisible($(id), false);
      return;
    }
    $("terminal-name").textContent = instance.name;
    $("terminal-status").textContent = instance.status;
    $("terminal-status").className = `status ${statusClass(instance.status)}`;
    $("terminal-workspace").textContent = instance.workspace;
    setVisible(empty, false);
    setVisible(terminal, true);
    setVisible(mobileInput, true);
    const live = ["running", "starting", "stopping"].includes(instance.status);
    const forceRequired = instance.force_stop_required === true;
    const shellBacked = instance.shell_backed === true;
    const terminalMode = shellBacked && instance.kind === "terminal";
    const agentMode =
      shellBacked && ["aeon", "codex", "claude", "grok"].includes(instance.kind);
    setVisible($("activate-aeon"), terminalMode && instance.status === "running");
    setVisible($("end-agent"), agentMode && instance.status === "running");
    setVisible(
      $("stop-instance"),
      instance.status === "running" && !agentMode
    );
    setVisible($("force-instance"), live || forceRequired);
    setVisible(
      $("resume-instance"),
      !forceRequired &&
        ["stopped", "exited", "interrupted", "error"].includes(instance.status)
    );
    setVisible($("delete-instance"), !live && !forceRequired);
  }

  function selectInstance(instanceId) {
    if (state.activeId === instanceId && state.terminal) {
      const socketOpen =
        state.socket &&
        [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState);
      if (!socketOpen) {
        state.reconnectAttempts = 0;
        openTerminalSocket(instanceId, state.terminalGeneration);
      }
      return;
    }
    state.activeId = instanceId;
    renderInstances();
    renderActiveHeader();
    connectTerminal();
  }

  function disconnectTerminal() {
    state.terminalGeneration += 1;
    clearTerminalReconnect();
    state.reconnectAttempts = 0;
    if (state.socket) {
      state.socket.onclose = null;
      state.socket.close();
      state.socket = null;
    }
    for (const disposable of state.terminalDisposables) {
      try { disposable.dispose(); } catch (_) {}
    }
    state.terminalDisposables = [];
    if (state.terminal) {
      state.terminal.dispose();
      state.terminal = null;
    }
    state.fit = null;
    $("terminal").replaceChildren();
  }

  function socketSend(message) {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  function clearTerminalReconnect() {
    if (state.reconnectTimer !== null) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
  }

  function terminalCanConnect(instanceId, generation) {
    const instance = activeInstance();
    return Boolean(
      instance &&
        instance.id === instanceId &&
        state.activeId === instanceId &&
        state.terminal &&
        state.terminalGeneration === generation &&
        !["stopped", "exited", "interrupted", "error"].includes(instance.status)
    );
  }

  function scheduleTerminalReconnect(instanceId, generation) {
    clearTerminalReconnect();
    if (
      !terminalCanConnect(instanceId, generation) ||
      document.hidden ||
      navigator.onLine === false ||
      state.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS
    ) return;
    const delay = Math.min(750 * (2 ** state.reconnectAttempts), 10000);
    state.reconnectAttempts += 1;
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      openTerminalSocket(instanceId, generation);
    }, delay);
  }

  function openTerminalSocket(instanceId, generation) {
    if (!terminalCanConnect(instanceId, generation)) return;
    if (
      state.socket &&
      [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket.readyState)
    ) return;
    if (document.hidden || navigator.onLine === false) {
      scheduleTerminalReconnect(instanceId, generation);
      return;
    }
    clearTerminalReconnect();
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(
      `${scheme}://${location.host}/ws/instances/${encodeURIComponent(instanceId)}`,
      ["aeon-v1", `csrf.${state.csrf}`]
    );
    state.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      if (state.socket !== socket || !terminalCanConnect(instanceId, generation)) {
        socket.close();
        return;
      }
      state.reconnectAttempts = 0;
      setVisible($("terminal-disconnected"), false);
      state.fit.fit();
      socketSend({
        type: "resize",
        rows: state.terminal.rows,
        cols: state.terminal.cols,
      });
    };
    socket.onmessage = (event) => {
      if (
        state.socket === socket &&
        state.terminalGeneration === generation &&
        event.data instanceof ArrayBuffer
      ) {
        state.terminal.write(new Uint8Array(event.data));
      }
    };
    socket.onclose = () => {
      if (state.socket !== socket) return;
      state.socket = null;
      $("terminal-disconnected").textContent =
        "Terminal disconnected. Reconnecting while this session remains available.";
      setVisible($("terminal-disconnected"), true);
      scheduleTerminalReconnect(instanceId, generation);
    };
  }

  function connectTerminal() {
    disconnectTerminal();
    const instance = activeInstance();
    if (!instance || ["interrupted", "error", "stopped", "exited"].includes(instance.status)) {
      $("terminal-disconnected").textContent =
        "This terminal is not running. Resume it to continue from saved state.";
      setVisible($("terminal-disconnected"), true);
      return;
    }
    if (!window.Terminal || !window.FitAddon) {
      $("terminal-disconnected").textContent = "The terminal component failed to load.";
      setVisible($("terminal-disconnected"), true);
      return;
    }
    setVisible($("terminal-disconnected"), false);
    state.terminal = new window.Terminal({
      cursorBlink: true,
      convertEol: false,
      allowProposedApi: false,
      scrollback: 10000,
      fontSize: window.innerWidth < 600 ? 12 : 13,
      fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
      theme: {
        background: "#05070a",
        foreground: "#e9eff9",
        cursor: "#64e7bd",
        selectionBackground: "#355077",
        black: "#10141c",
        brightBlack: "#596579",
        green: "#64e7bd",
        brightGreen: "#89f1d1",
        blue: "#7c9cff",
        brightBlue: "#a8bbff",
        red: "#ff7085",
        brightRed: "#ff9baa",
      },
    });
    state.fit = new window.FitAddon.FitAddon();
    state.terminal.loadAddon(state.fit);
    state.terminal.open($("terminal"));
    requestAnimationFrame(() => {
      state.fit.fit();
      state.terminal.focus();
    });

    openTerminalSocket(instance.id, state.terminalGeneration);
    state.terminalDisposables.push(
      state.terminal.onData((data) => socketSend({ type: "input", data })),
      state.terminal.onResize(({ rows, cols }) =>
        socketSend({ type: "resize", rows, cols })
      )
    );
  }

  window.addEventListener("resize", () => {
    if (state.fit) {
      try { state.fit.fit(); } catch (_) {}
    }
  });

  window.addEventListener("offline", clearTerminalReconnect);
  window.addEventListener("online", () => {
    if (state.activeId && state.terminal && !state.socket) {
      state.reconnectAttempts = 0;
      openTerminalSocket(state.activeId, state.terminalGeneration);
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearTerminalReconnect();
    } else if (state.activeId && state.terminal && !state.socket) {
      state.reconnectAttempts = 0;
      openTerminalSocket(state.activeId, state.terminalGeneration);
    }
  });

  $("command-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("command-input");
    if (input.value && socketSend({ type: "input", data: `${input.value}\r` })) {
      input.value = "";
      if (state.terminal) state.terminal.focus();
    }
  });
  $("command-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("command-form").requestSubmit();
    }
  });
  document.querySelectorAll("[data-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const keys = {
        "ctrl-c": "\u0003",
        escape: "\u001b",
        tab: "\t",
        up: "\u001b[A",
        down: "\u001b[B",
      };
      socketSend({ type: "input", data: keys[button.dataset.key] || "" });
      if (state.terminal) state.terminal.focus();
    });
  });

  async function refreshWorkspaces() {
    state.workspaces = await api("/api/workspaces");
    const workspaceSelect = $("instance-workspace");
    const rootSelect = $("workspace-root");
    workspaceSelect.replaceChildren();
    rootSelect.replaceChildren();
    for (const workspace of state.workspaces.workspaces || []) {
      const option = document.createElement("option");
      option.value = workspace;
      option.textContent = workspace;
      workspaceSelect.append(option);
    }
    for (const root of state.workspaces.roots || []) {
      const option = document.createElement("option");
      option.value = root;
      option.textContent = root;
      rootSelect.append(option);
    }
  }

  $("create-workspace").addEventListener("click", async () => {
    $("new-error").textContent = "";
    try {
      const payload = await api("/api/workspaces", {
        method: "POST",
        body: JSON.stringify({
          root: $("workspace-root").value,
          name: $("workspace-name").value,
        }),
      });
      await refreshWorkspaces();
      $("instance-workspace").value = payload.workspace;
      $("workspace-name").value = "";
    } catch (error) {
      $("new-error").textContent = errorText(error);
    }
  });

  function openNewDialog() {
    $("new-error").textContent = "";
    newDialog.showModal();
    setTimeout(() => $("instance-name").focus(), 0);
  }
  $("new-agent").addEventListener("click", openNewDialog);
  $("empty-new-agent").addEventListener("click", openNewDialog);
  document.querySelectorAll(".close-dialog").forEach((button) =>
    button.addEventListener("click", () => newDialog.close())
  );

  $("new-instance-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("new-error").textContent = "";
    const submit = event.submitter;
    if (submit) submit.disabled = true;
    try {
      const name = $("instance-name").value.trim();
      const body = { workspace: $("instance-workspace").value };
      if (name) body.name = name;
      const payload = await api("/api/terminals", {
        method: "POST",
        body: JSON.stringify(body),
      });
      newDialog.close();
      event.target.reset();
      await refreshInstances();
      selectInstance(payload.instance.id);
    } catch (error) {
      $("new-error").textContent = errorText(error);
    } finally {
      if (submit) submit.disabled = false;
    }
  });

  $("activate-aeon").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance) return;
    await api(`/api/instances/${instance.id}/activate-agent`, {
      method: "POST",
      body: JSON.stringify({ kind: "aeon" }),
    });
    await refreshInstances();
    connectTerminal();
  });

  $("end-agent").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance) return;
    await api(`/api/instances/${instance.id}/end-agent`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    await refreshInstances();
    connectTerminal();
  });

  $("stop-instance").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance || !confirm(`Gracefully stop "${instance.name}"? Its state will be saved.`)) return;
    await api(`/api/instances/${instance.id}/stop`, { method: "POST" });
    await refreshInstances();
  });

  $("resume-instance").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance) return;
    await api(`/api/instances/${instance.id}/resume`, { method: "POST" });
    await refreshInstances();
    connectTerminal();
  });

  $("force-instance").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance) return;
    const confirmation = prompt(
      `Force stop can interrupt an active tool. Type the instance name to continue:\n${instance.name}`
    );
    if (confirmation !== instance.name) return;
    await api(`/api/instances/${instance.id}/force-stop`, {
      method: "POST",
      body: JSON.stringify({ confirmation }),
    });
    disconnectTerminal();
    await refreshInstances();
  });

  $("delete-instance").addEventListener("click", async () => {
    const instance = activeInstance();
    if (!instance) return;
    const confirmation = prompt(
      `Delete this terminal tab? Workspace files are preserved. Type:\n${instance.name}`
    );
    if (confirmation !== instance.name) return;
    await api(`/api/instances/${instance.id}`, {
      method: "DELETE",
      body: JSON.stringify({ confirmation }),
    });
    state.activeId = null;
    disconnectTerminal();
    await refreshInstances();
  });

  function formatBytes(value) {
    if (value === null || value === undefined) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let amount = Number(value);
    let index = 0;
    while (amount >= 1024 && index < units.length - 1) {
      amount /= 1024;
      index += 1;
    }
    return `${amount >= 10 || index === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
  }

  function formatMib(value) {
    return value === null || value === undefined ? "—" : formatBytes(value * 1024 * 1024);
  }

  async function refreshResources() {
    try {
      const data = await api("/api/resources");
      const host = data.host || {};
      $("cpu-value").textContent = `${Math.round(host.cpu_percent || 0)}%`;
      $("load-value").textContent = `load ${(host.load || []).map((v) => Number(v).toFixed(1)).join(" / ")}`;
      $("memory-value").textContent = `${Math.round(host.memory_percent || 0)}%`;
      $("memory-detail").textContent = `${formatBytes(host.memory_used)} / ${formatBytes(host.memory_total)}`;
      $("disk-value").textContent = `${Math.round(host.disk_percent || 0)}%`;
      $("disk-detail").textContent = `${formatBytes(host.disk_used)} / ${formatBytes(host.disk_total)}`;
      const gpuCards = $("gpu-cards");
      gpuCards.replaceChildren();
      for (const gpu of data.gpus || []) {
        const card = document.createElement("div");
        card.className = `resource-card gpu-card ${String(gpu.state || "").toLowerCase()}`;
        const title = document.createElement("span");
        title.textContent = `${gpu.host} · GPU ${gpu.gpu}`;
        const value = document.createElement("strong");
        value.textContent = gpu.state || "unknown";
        const detail = document.createElement("small");
        detail.textContent =
          gpu.memory_total_mib == null
            ? "renter protected"
            : `${formatMib(gpu.memory_used_mib)} used · ${formatMib(gpu.safely_allocatable_mib)} safe`;
        card.append(title, value, detail);
        gpuCards.append(card);
      }
    } catch (_) {
      // The terminal remains useful if resource telemetry temporarily fails.
    }
  }

  $("resources-toggle").addEventListener("click", () => {
    $("resource-strip").classList.toggle("open");
  });

  async function bootstrap() {
    try {
      const session = await api("/api/session");
      await showApp(session);
    } catch (_) {
      showLogin();
    }
    setInterval(() => {
      if (!appView.classList.contains("hidden")) refreshInstances();
    }, 5000);
    setInterval(() => {
      if (!appView.classList.contains("hidden")) refreshResources();
    }, 12000);
  }

  bootstrap();
})();
