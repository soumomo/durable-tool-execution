from __future__ import annotations

import asyncio
import logging
import math
from typing import Self
from uuid import uuid4

from .models import (
    JobRecord,
    JSONValue,
    RuntimeConfig,
    StaleLeaseError,
    ToolFatalError,
    ToolRetryableError,
)
from .registry import RegisteredTool, ToolRegistry
from .store import SQLiteJobStore, _canonical_json, _finish_cleanup

logger = logging.getLogger(__name__)


class DurableToolRuntime:
    """A single-process worker engine; SQLite remains authoritative across restarts."""

    def __init__(self, config: RuntimeConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry
        self.store = SQLiteJobStore(config)
        self.fenced_stale_completions = 0
        self.lost_leases = 0
        self._tools: dict[str, RegisteredTool] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._reaper: asyncio.Task[None] | None = None
        self._state = "new"
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._owner = uuid4().hex

    def _signal(self) -> None:
        # Observers capture the old event BEFORE reading SQLite. Rotating it avoids
        # clearing a notification that arrived between a DB read and a wait.
        previous = self._changed
        self._changed = asyncio.Event()
        previous.set()

    async def _wait_for_change(self, change: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(change.wait(), self.config.poll_interval_ms / 1000)
        except TimeoutError:
            pass

    def _check_loop(self) -> None:
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("Runtime belongs to a different event loop")

    def _check_available(self, *, accepting: bool = False) -> None:
        self._check_loop()
        if self._error is not None:
            raise RuntimeError(
                "Runtime background task failed; stop and inspect the cause"
            ) from self._error
        allowed = ("running",) if accepting else ("running", "draining")
        if self._state not in allowed:
            raise RuntimeError(f"Runtime is {self._state}, not available")

    async def start(self) -> None:
        async with self._lifecycle_lock:
            self._check_loop()
            if self._state == "running":
                self._check_available()
                return
            if self._state != "new":
                raise RuntimeError(
                    "A stopped runtime cannot be restarted; create a new instance"
                )
            self._state = "starting"
            self._loop = asyncio.get_running_loop()
            self._tools = self.registry.snapshot()
            try:
                await self.store.start()
                for job in await self.store.list_unfinished_jobs():
                    tool = self._tools[job.tool_name]
                    if tool.policy.idempotent != job.policy.idempotent:
                        raise ValueError(
                            f"Idempotency policy changed for unfinished tool {job.tool_name!r}"
                        )
                    try:
                        tool.validate_arguments(job.args)
                    except TypeError as exc:
                        raise ValueError(
                            f"Signature changed for unfinished tool {job.tool_name!r}: {exc}"
                        ) from exc
                await self.store.recover_expired_leases()
                self._state = "running"
                self._workers = [
                    asyncio.create_task(
                        self._worker(f"{self._owner}:{index}"),
                        name=f"tool-worker-{index}",
                    )
                    for index in range(self.config.max_workers)
                ]
                self._reaper = asyncio.create_task(
                    self._recover_loop(), name="tool-lease-reaper"
                )
                for task in [*self._workers, self._reaper]:
                    task.add_done_callback(self._background_done)
            except BaseException:
                self._state = "stopping"
                try:
                    await self._cancel_background()
                finally:
                    try:
                        await _finish_cleanup(asyncio.create_task(self.store.close()))
                    finally:
                        self._state = "closed"
                        self._signal()
                raise

    def _report_failure(self, error: BaseException) -> None:
        # Latch before any cleanup await: cancellation must not hide the cause.
        if self._error is not None:
            return
        self._error = error
        logger.error("Runtime infrastructure failure", exc_info=error)
        if self._state in ("running", "draining", "failed"):
            self._state = "failed"
            current = asyncio.current_task()
            for task in [*self._workers, self._reaper]:
                if task is not None and task is not current:
                    self._cancel_once(task)
        self._signal()

    @staticmethod
    def _cancel_once(task: asyncio.Task[object]) -> None:
        if not task.done() and not task.cancelling():
            task.cancel()

    def _background_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            if self._state in ("running", "draining"):
                self._report_failure(
                    RuntimeError(
                        f"Background task {task.get_name()} was unexpectedly cancelled"
                    )
                )
        elif (error := task.exception()) is not None:
            self._report_failure(error)

    async def _cancel_background(self) -> None:
        tasks = [*self._workers]
        if self._reaper is not None:
            tasks.append(self._reaper)
        for task in tasks:
            self._cancel_once(task)
        if tasks:
            joined = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await _finish_cleanup(joined)
            finally:
                # Deferred caller cancellation must not skip harvesting real errors.
                if joined.done() and not joined.cancelled():
                    for result in joined.result():
                        if isinstance(result, BaseException) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            self._report_failure(result)

    async def stop(self, graceful: bool = True) -> None:
        if type(graceful) is not bool:
            raise ValueError("graceful must be a bool")
        async with self._lifecycle_lock:
            self._check_loop()
            if self._state == "closed":
                return
            failure: BaseException | None = None
            cancelled = False
            self._state = (
                "draining" if graceful and self._state != "new" else "stopping"
            )
            self._signal()
            try:
                if self._state == "draining":
                    while True:
                        self._check_available()
                        change = self._changed
                        if await self.store.count_outstanding_jobs() == 0:
                            break
                        await self._wait_for_change(change)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as exc:  # noqa: BLE001 -- propagate after all cleanup stages.
                failure = exc
            finally:
                self._state = "stopping"
                self._signal()
                try:
                    await self._cancel_background()
                except asyncio.CancelledError:
                    cancelled = True
                except BaseException as exc:  # noqa: BLE001 -- close the store before propagating.
                    failure = failure or exc
                try:
                    await _finish_cleanup(asyncio.create_task(self.store.close()))
                except asyncio.CancelledError:
                    cancelled = True
                except BaseException as exc:  # noqa: BLE001 -- finish lifecycle bookkeeping first.
                    failure = failure or exc
                finally:
                    self._state = "closed"
                    self._signal()
            if cancelled:
                raise asyncio.CancelledError
            if failure is not None:
                raise failure
            if self._error is not None:
                raise RuntimeError(
                    "Runtime stopped after a background task failure"
                ) from self._error

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop(graceful=exc[0] is None)

    async def submit(
        self, tool_name: str, args: dict[str, JSONValue], idempotency_key: str
    ) -> str:
        self._check_available(accepting=True)
        tool = self._tools[tool_name]
        tool.validate_arguments(args)
        job, _ = await self.store.admit_job(tool.policy, args, idempotency_key)
        self._signal()
        return job.job_id

    async def wait_for_terminal(
        self, job_id: str, timeout: float | None = None
    ) -> JobRecord:
        if timeout is not None and (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite nonnegative number or None")
        async with asyncio.timeout(timeout):
            while True:
                self._check_available()
                change = self._changed
                job = await self.store.get_job(job_id)
                if job.status.terminal:
                    return job
                await self._wait_for_change(change)

    async def _worker(self, worker_id: str) -> None:
        try:
            while self._state in ("running", "draining"):
                change = self._changed
                job = await self.store.claim_next_job(worker_id)
                if job is None:
                    await self._wait_for_change(change)
                    continue
                await self._run_job(job)
        except Exception as exc:
            self._report_failure(exc)
            raise

    async def _recover_loop(self) -> None:
        try:
            while self._state in ("running", "draining"):
                await asyncio.sleep(self.config.recovery_interval_ms / 1000)
                if await self.store.recover_expired_leases():
                    self._signal()
        except Exception as exc:
            self._report_failure(exc)
            raise

    async def _heartbeat(self, job: JobRecord) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.heartbeat_interval_ms / 1000)
                if not await self.store.heartbeat_lease(job.job_id, job.attempt_count):
                    return
        except Exception as exc:
            self._report_failure(exc)
            raise

    async def _run_job(self, job: JobRecord) -> None:
        operation = asyncio.create_task(
            self._perform(job), name=f"tool-attempt-{job.job_id}-{job.attempt_count}"
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(job),
            name=f"tool-heartbeat-{job.job_id}-{job.attempt_count}",
        )
        try:
            done, _ = await asyncio.wait(
                {operation, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                heartbeat.result()
            if operation in done:
                operation.result()
            else:
                self.lost_leases += 1
                logger.warning(
                    "Lost lease for job %s attempt %s; cancelling execution",
                    job.job_id,
                    job.attempt_count,
                )
        except StaleLeaseError:
            self.fenced_stale_completions += 1
            logger.warning(
                "Fenced stale outcome for job %s attempt %s",
                job.job_id,
                job.attempt_count,
            )
        except Exception as exc:
            self._report_failure(exc)
            raise
        finally:
            for task in (operation, heartbeat):
                self._cancel_once(task)
            try:
                await _finish_cleanup(
                    asyncio.gather(operation, heartbeat, return_exceptions=True)
                )
            finally:
                self._signal()

    async def _call_tool(self, job: JobRecord, tool: RegisteredTool) -> JSONValue:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + job.policy.timeout_ms / 1000
        finished_at: float | None = None

        async def invoke() -> JSONValue:
            nonlocal finished_at
            try:
                return await tool.function(**job.args)
            finally:
                finished_at = loop.time()

        invocation = asyncio.create_task(
            invoke(), name=f"tool-call-{job.job_id}-{job.attempt_count}"
        )
        try:
            # The timer only wakes this waiter; it never cancels the user task.
            # It is disarmed before cancellation/join, protecting async finally blocks.
            done, _ = await asyncio.wait(
                {invocation}, timeout=max(0, deadline - loop.time())
            )
            if invocation not in done or (
                finished_at is not None and finished_at > deadline
            ):
                raise TimeoutError(f"Tool exceeded {job.policy.timeout_ms} ms deadline")
            return invocation.result()
        finally:
            self._cancel_once(invocation)
            await _finish_cleanup(asyncio.gather(invocation, return_exceptions=True))

    async def _perform(self, job: JobRecord) -> None:
        try:
            tool = self._tools[job.tool_name]
            try:
                if tool.policy.idempotent != job.policy.idempotent:
                    raise ValueError(
                        "Tool idempotency is incompatible with the accepted job"
                    )
                tool.validate_arguments(job.args)
            except (TypeError, ValueError) as exc:
                await self.store.fail_or_retry_job(
                    job.job_id,
                    job.attempt_count,
                    f"Invalid tool binding: {exc}",
                    retryable=False,
                    outcome_unknown=False,
                )
                return
            try:
                result = await self._call_tool(job, tool)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                await self._record_failure(
                    job, "Tool cancelled its own execution", retryable=True
                )
            except (TimeoutError, ToolRetryableError) as exc:
                await self._record_failure(
                    job, f"{type(exc).__name__}: {exc}", retryable=True
                )
            except ToolFatalError as exc:
                await self._record_failure(
                    job,
                    f"ToolFatalError: {exc}",
                    retryable=False,
                    outcome_unknown=False,
                )
            except Exception as exc:  # noqa: BLE001 -- tool errors are outcomes, not worker failures.
                await self._record_failure(
                    job, f"{type(exc).__name__}: {exc}", retryable=False
                )
            else:
                try:
                    _canonical_json(result)
                except (TypeError, ValueError) as exc:
                    await self._record_failure(
                        job, f"Invalid JSON result: {exc}", retryable=False
                    )
                else:
                    await self.store.complete_job(job.job_id, job.attempt_count, result)
        except StaleLeaseError:
            raise
        except Exception as exc:
            self._report_failure(exc)
            raise

    async def _record_failure(
        self,
        job: JobRecord,
        error: str,
        *,
        retryable: bool,
        outcome_unknown: bool = True,
    ) -> None:
        await self.store.fail_or_retry_job(
            job.job_id,
            job.attempt_count,
            error,
            retryable=retryable,
            outcome_unknown=outcome_unknown,
        )
