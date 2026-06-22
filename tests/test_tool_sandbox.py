from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_runtime.schema import ToolCall
from sandbox import builtin_tools
from sandbox.tool_sandbox import ToolSandbox


class ToolSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.sandbox = ToolSandbox(allowed_root=self.project_root)
        self.sandbox.register("echo", builtin_tools.echo_tool)
        self.sandbox.register("sleep", builtin_tools.sleep_tool)
        self.sandbox.register("stderr", builtin_tools.stderr_tool)

    def test_rejects_non_allowlisted_tool(self) -> None:
        obs = self.sandbox.run(
            ToolCall(
                agent="Supervisor",
                tool_name="missing",
                args={},
                timeout_seconds=1,
                cwd=str(self.project_root),
            )
        )
        self.assertFalse(obs.ok)
        self.assertEqual(obs.error_type, "ToolNotAllowed")

    def test_rejects_cwd_outside_project_root(self) -> None:
        outside = Path(tempfile.gettempdir()).resolve()
        obs = self.sandbox.run(
            ToolCall(
                agent="Supervisor",
                tool_name="echo",
                args={"message": "hello"},
                timeout_seconds=1,
                cwd=str(outside),
            )
        )
        self.assertFalse(obs.ok)
        self.assertEqual(obs.error_type, "ValueError")
        self.assertIn("outside allowed root", obs.error_message or "")

    def test_timeout_terminates_tool_process(self) -> None:
        obs = self.sandbox.run(
            ToolCall(
                agent="Supervisor",
                tool_name="sleep",
                args={"seconds": 2.0},
                timeout_seconds=0.2,
                cwd=str(self.project_root),
            )
        )
        self.assertFalse(obs.ok)
        self.assertEqual(obs.error_type, "TimeoutError")
        self.assertLess(obs.elapsed_seconds, 2.0)

    def test_captures_stdout_and_stderr(self) -> None:
        stdout_obs = self.sandbox.run(
            ToolCall(
                agent="Supervisor",
                tool_name="echo",
                args={"message": "hello"},
                timeout_seconds=1,
                cwd=str(self.project_root),
            )
        )
        stderr_obs = self.sandbox.run(
            ToolCall(
                agent="Supervisor",
                tool_name="stderr",
                args={"message": "warn"},
                timeout_seconds=1,
                cwd=str(self.project_root),
            )
        )
        self.assertTrue(stdout_obs.ok)
        self.assertIn("hello", stdout_obs.stdout)
        self.assertTrue(stderr_obs.ok)
        self.assertIn("warn", stderr_obs.stderr)


if __name__ == "__main__":
    unittest.main()
