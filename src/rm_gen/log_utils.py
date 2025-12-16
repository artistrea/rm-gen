"""
This module provides logging utilities that integrate with tqdm progress bars.
It defines a logger wrapper that allows logging messages to be displayed
correctly when using tqdm for progress indication.
"""


import io
import logging
import sys
from contextlib import contextmanager
from typing import Iterable

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class _NullWriter(io.StringIO):
    def write(self, s):
        return len(s)
    def flush(self):
        pass


class _LoggerWithTQDM:
    """Wraps around logging.Logger and provides logging utility functions
    for when using tqdm.
    """
    def __init__(
        self,
        logger: logging.Logger,
        tqdm_enabled: bool = True,
    ):
        self._logger = logger
        self._tqdm_enabled = tqdm_enabled
        if self._logger.handlers:
            self._stream = self._logger.handlers[0].stream
        else:
            self._stream = sys.stdout

    def _is_console_handler(self):
        if not self._logger:
            return False
        for h in self._logger.handlers:
            if (
                isinstance(h, logging.StreamHandler)
                and h.stream in (sys.stdout, sys.stderr)
            ):
                return True
        return False

    # --------------------------------------------------------------
    # Basic passthrough (standard logging behavior)
    def debug(self, msg: str, *a, **k):
        self._logger.debug(msg, *a, **k)
    def info(self, msg: str, *a, **k):
        self._logger.info(msg, *a, **k)
    def warning(self, msg: str, *a, **k):
        self._logger.warning(msg, *a, **k)
    def error(self, msg: str, *a, **k):
        self._logger.error(msg, *a, **k)
    def critical(self, msg: str, *a, **k):
        self._logger.critical(msg, *a, **k)
    # pylint: disable=invalid-name,multiple-statements
    def setLevel(self, level: int | str):
        self._logger.setLevel(level)
    # --------------------------------------------------------------

    @contextmanager
    def tqdm_progress_bar(
        self,
        iterable: Iterable,
        level: int = logging.INFO,
        **tqdm_opts
    ):
        """
        Wraps an iterable with tqdm progress bar if logging level is enabled.
        If tqdm is not enabled or logging level is not enabled, returns the
        original iterable.

        Usage:
            with logger.tqdm_progress_bar(iterable) as progress_bar:
                for item in progress_bar:
                    # ...
                    logger.info("Something")
        """
        if not self._tqdm_enabled or not self._logger.isEnabledFor(level):
            # just disable but keep interface
            with tqdm(
                iterable,
                disable=True,
                **tqdm_opts
            ) as bar:
                yield bar
            return

        if not self._is_console_handler():
            # keep it enabled so that we can log start and end of iteration
            # make it write to null so it does not interfere with logger
            with tqdm(
                iterable,
                file=_NullWriter(),
                **tqdm_opts
            ) as bar:
                self._logger.log(level, f"TQDM ITERATION START: {bar.desc}")
                yield bar
                bar_info = bar.format_dict
                frac = f"{bar_info['n']}/{bar_info['total'] or '?'}"
                formatted_info = {
                    "processed": f"{frac} {bar_info['unit']}",
                    "elapsed": bar.format_interval(bar_info["elapsed"]),
                }
                if bar_info["postfix"]:
                    formatted_info["postfix"] = bar_info["postfix"]
                self._logger.debug(f"TQDM ITERATION END: {bar.desc}")

                self._logger.info(f"{bar.desc} info: {formatted_info}")
                return

        # normal tqdm with console handler
        # but wrapping loggers to avoid messing up progress bar
        with logging_redirect_tqdm([self._logger], tqdm_class=tqdm):
            with tqdm(iterable, **tqdm_opts) as pbar:
                yield pbar
