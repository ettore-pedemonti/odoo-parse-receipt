"""
Retries for calls to external services (here: Google Gemini).

run_with_retries() repeats an async call when the error looks transient (timeout, rate limit,
service unavailable), waiting longer each time (exponential backoff with a little random jitter).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar


R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    jitter_ratio: float = 0.2

    def backoff_seconds(self, attempt_number: int) -> float:
        bounded_attempt = max(1, int(attempt_number))
        base_delay = min(
            max(0.0, self.max_delay_seconds),
            max(0.0, self.base_delay_seconds) * (2 ** (bounded_attempt - 1)),
        )
        if self.jitter_ratio <= 0 or base_delay <= 0:
            return base_delay
        jitter = base_delay * self.jitter_ratio
        return max(0.0, base_delay + random.uniform(-jitter, jitter))


def _exc_message(exc: BaseException) -> str:
    return str(exc).strip().lower()


def _looks_like_connection_error(exc: BaseException) -> bool:
    message = _exc_message(exc)
    return any(
        token in message
        for token in (
            "connection reset",
            "connection refused",
            "connection closed",
            "server closed the connection",
            "broken pipe",
            "temporarily unavailable",
            "try again",
            "eof",
            "socket",
        )
    )


def is_transient_google_error(exc: BaseException) -> bool:
    exc_type_name = type(exc).__name__.lower()
    if any(token in exc_type_name for token in ("deadline", "timeout", "unavailable", "resourceexhausted", "internal")):
        return True
    message = _exc_message(exc)
    if any(token in message for token in ("429", "rate limit", "deadline exceeded", "temporarily unavailable", "internal error")):
        return True
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, ConnectionResetError, BrokenPipeError, OSError)) or _looks_like_connection_error(exc)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def run_with_retries(
    operation: Callable[[], Awaitable[R] | R],
    *,
    dependency: str,
    operation_name: str,
    logger: logging.Logger | None,
    is_transient: Callable[[BaseException], bool],
    retry_policy: RetryPolicy | None = None,
) -> R:
    policy = retry_policy or RetryPolicy()
    last_exc: BaseException | None = None
    for attempt_number in range(1, max(1, policy.max_attempts) + 1):
        try:
            result = await _maybe_await(operation())
            return result
        except Exception as exc:
            last_exc = exc
            transient = bool(is_transient(exc))
            is_final_attempt = attempt_number >= max(1, policy.max_attempts)
            if not transient or is_final_attempt:
                if logger is not None:
                    logger.error(
                        "dependency_call_failed dependency=%s operation=%s attempts=%s transient=%s error_type=%s error=%s",
                        dependency,
                        operation_name,
                        attempt_number,
                        transient,
                        type(exc).__name__,
                        exc,
                    )
                raise

            sleep_seconds = policy.backoff_seconds(attempt_number)
            if logger is not None:
                logger.warning(
                    "dependency_call_retry_scheduled dependency=%s operation=%s attempt=%s max_attempts=%s sleep_seconds=%.3f error_type=%s error=%s",
                    dependency,
                    operation_name,
                    attempt_number,
                    policy.max_attempts,
                    sleep_seconds,
                    type(exc).__name__,
                    exc,
                )
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{dependency}:{operation_name} exhausted retries without an exception")
