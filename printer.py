import sys
import threading
import time
from typing import Optional

class ProgressPrinter:
    """
    Thread‑safe in‑place status-line updater with manual clear and optional formatting.
    """

    def __init__(self, debug: Optional[bool] = False) -> None:
        self._lock = threading.Lock()
        self._last_len = 0
        self._debug = debug

    def progress(
        self,
        msg: str,
        prefix: str = "",
        suffix: str = "",
        timestamp: bool = True
    ) -> None:
        """
        # Update the console in place with a single status line.
        # - msg: main message text
        # - prefix/suffix: optional strings to wrap around msg
        # - timestamp: if True, prepend current HH:MM:SS
        # """
        if self._debug:
            with self._lock:
                # Build full line
                ts = time.strftime("%H:%M:%S ") if timestamp else ""
                full = f"{prefix}{ts}{msg}{suffix}"
                # Overwrite old content
                sys.stdout.write("\r" + full.ljust(self._last_len))
                sys.stdout.flush()
                self._last_len = len(full)

    def println(self, msg: str = "") -> None:
        # """
        # Emit a normal newline log, clearing any existing status line first.
        # """
        if self._debug:
            with self._lock:
                sys.stdout.write("\r" + " " * self._last_len + "\r")
                sys.stdout.flush()
                print(msg)
        