from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict

from agent_runtime.schema import Observation, ToolCall, ensure_relative_to


ToolFn = Callable[..., Dict[str, Any]]


class SubprocessToolSandbox:
    """Allowlisted subprocess runner for importable Python function tools."""

    def __init__(self, *, allowed_root: str | Path):
        self.allowed_root = Path(allowed_root).resolve()
        self.tools: Dict[str, str] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if "<locals>" in fn.__qualname__:
            raise ValueError(f"Tool {name} must be a top-level importable function")
        self.tools[name] = f"{fn.__module__}:{fn.__qualname__}"

    def run(self, call: ToolCall) -> Observation:
        started = time.time()
        if call.tool_name not in self.tools:
            return Observation(
                tool_call_id=call.id,
                agent=call.agent,
                tool_name=call.tool_name,
                ok=False,
                error_type="ToolNotAllowed",
                error_message=f"Tool {call.tool_name} is not in allowlist",
                elapsed_seconds=round(time.time() - started, 3),
            )
        cwd = str(self.allowed_root)
        if call.cwd:
            try:
                cwd = str(ensure_relative_to(call.cwd, self.allowed_root))
            except Exception as exc:
                return Observation(
                    tool_call_id=call.id,
                    agent=call.agent,
                    tool_name=call.tool_name,
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    elapsed_seconds=round(time.time() - started, 3),
                )

        payload = {
            "function": self.tools[call.tool_name],
            "args": call.args,
        }
        runner = textwrap.dedent(
            """
            import contextlib
            import importlib
            import io
            import json
            import sys
            import traceback

            payload = json.loads(sys.stdin.read())
            module_name, qualname = payload["function"].split(":", 1)
            obj = importlib.import_module(module_name)
            fn = obj
            for part in qualname.split("."):
                fn = getattr(fn, part)
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    output = fn(**payload.get("args", {})) or {}
                envelope = {
                    "ok": True,
                    "output": output,
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                    "error_type": None,
                    "error_message": None,
                }
            except BaseException as exc:
                envelope = {
                    "ok": False,
                    "output": {},
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue() + traceback.format_exc(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            print(json.dumps(envelope, ensure_ascii=True))
            """
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", runner],
                input=json.dumps(payload, ensure_ascii=True),
                text=True,
                encoding="utf-8",
                cwd=cwd,
                capture_output=True,
                timeout=call.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return Observation(
                tool_call_id=call.id,
                agent=call.agent,
                tool_name=call.tool_name,
                ok=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                error_type="TimeoutError",
                error_message=f"Tool timed out after {call.timeout_seconds}s",
                elapsed_seconds=round(time.time() - started, 3),
            )

        elapsed = round(time.time() - started, 3)
        if completed.returncode != 0:
            return Observation(
                tool_call_id=call.id,
                agent=call.agent,
                tool_name=call.tool_name,
                ok=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error_type="SubprocessError",
                error_message=f"Tool process exited with code {completed.returncode}",
                elapsed_seconds=elapsed,
            )
        try:
            envelope = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return Observation(
                tool_call_id=call.id,
                agent=call.agent,
                tool_name=call.tool_name,
                ok=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error_type=type(exc).__name__,
                error_message=f"Failed to parse tool envelope: {exc}",
                elapsed_seconds=elapsed,
            )
        return Observation(
            tool_call_id=call.id,
            agent=call.agent,
            tool_name=call.tool_name,
            ok=bool(envelope.get("ok")),
            output=envelope.get("output") or {},
            stdout=envelope.get("stdout", ""),
            stderr=envelope.get("stderr", ""),
            error_type=envelope.get("error_type"),
            error_message=envelope.get("error_message"),
            elapsed_seconds=elapsed,
        )


ToolSandbox = SubprocessToolSandbox
