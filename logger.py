
"""Simple caller-aware logger.

Creates a new file per session named like YYYY-MM-DD-HH-MM-SS.logs in the same
directory as this module. Each log record gets a `caller` attribute injected
that contains ClassName.method (or module.function) so calls like

	from logger import logger
	logger.info("hello")

will include the class/module and function name in the message.
"""
from __future__ import annotations

import inspect
import logging
from datetime import datetime
import os
from pathlib import Path


_LOG_DIR = Path(__file__).parent
LOG_FOLDER = _LOG_DIR / "logs"

def _session_log_path() -> Path:
	# Create the folder if it doesn't already exist
	os.makedirs(LOG_FOLDER, exist_ok=True)
	ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	return LOG_FOLDER / f"{ts}.logs"


class _CallerFilter(logging.Filter):
	"""Logging filter that injects a `caller` attribute into the LogRecord.

	The `caller` is formatted as `ClassName.method` when called from an
	instance/class method, or as `module.function` for module-level functions.
	"""

	def filter(self, record: logging.LogRecord) -> bool:  # always True to not block records
		try:
			# walk the stack to find the first frame outside this module and logging
			for frame_info in inspect.stack()[1:]:
				mod = inspect.getmodule(frame_info.frame)
				if mod is None:
					continue
				mod_name = mod.__name__
				if mod_name.startswith("logging"):
					continue
				if mod_name == __name__:
					continue

				func = frame_info.function
				# try instance or class first
				local_self = frame_info.frame.f_locals.get("self") or frame_info.frame.f_locals.get("cls")
				if local_self:
					# handle both instances and classes
					cls_name = local_self.__name__ if isinstance(local_self, type) else type(local_self).__name__
				else:
					# fallback to module name when not in a class
					cls_name = mod_name

				record.caller = f"{cls_name}.{func}"
				break
		except Exception:
			record.caller = "<unknown>"
		return True


def get_logger(name: str = "app") -> logging.Logger:
	"""Return a configured logger. Multiple calls will reuse the same logger
	without duplicating handlers.
	"""
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	logger.setLevel(logging.DEBUG)

	fmt = "%(asctime)s [%(levelname)s] [%(caller)s] %(message)s"
	formatter = logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

	# File handler for this session
	fh = logging.FileHandler(_session_log_path(), encoding="utf-8")
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)

	# Console / stream handler
	sh = logging.StreamHandler()
	sh.setLevel(logging.DEBUG)
	sh.setFormatter(formatter)

	caller_filter = _CallerFilter()
	fh.addFilter(caller_filter)
	sh.addFilter(caller_filter)

	logger.addHandler(fh)
	logger.addHandler(sh)

	return logger


# module-level logger instance: call with logger.info(...)
logger = get_logger()
