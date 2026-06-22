from __future__ import annotations

import sys
import time
from typing import Any, Dict


def echo_tool(message: str = "ok") -> Dict[str, Any]:
    print(message)
    return {"message": message}


def sleep_tool(seconds: float = 1.0) -> Dict[str, Any]:
    time.sleep(seconds)
    return {"slept": seconds}


def stderr_tool(message: str = "err") -> Dict[str, Any]:
    print(message, file=sys.stderr)
    return {"message": message}
