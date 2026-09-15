from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from interface_ai_challenge.application.predicate_evaluator import PredicateEvaluator
from interface_ai_challenge.domain.handlers import Handler, HandlerCategory
from interface_ai_challenge.domain.predicates import Predicate, target_refs

_TERMINAL_CATEGORIES = (HandlerCategory.BUSINESS_OUTCOME, HandlerCategory.TERMINAL)


@dataclass(frozen=True)
class Observation:
    condition_met: bool
    active_handlers: tuple[Handler, ...]
    # A target of the unmet condition matches more than one element.
    ambiguous: bool = False

    @property
    def conflicting(self) -> bool:
        return self.condition_met and any(handler.category in _TERMINAL_CATEGORIES for handler in self.active_handlers)


class StateObserver:
    """Evaluates the expected condition and every known exceptional state on the same observation."""

    def __init__(self, evaluator: PredicateEvaluator, handlers: Sequence[Handler]) -> None:
        self._evaluator = evaluator
        self._handlers = tuple(sorted(handlers, key=lambda handler: handler.priority))

    async def observe(self, condition: Predicate) -> Observation:
        active = tuple([handler for handler in self._handlers if await self._evaluator.holds(handler.when)])
        condition_met = await self._evaluator.holds(condition)
        ambiguous = False
        if not condition_met and not active:
            for ref in sorted(target_refs(condition)):
                if await self._evaluator.candidate_count(ref) > 1:
                    ambiguous = True
                    break
        return Observation(condition_met=condition_met, active_handlers=active, ambiguous=ambiguous)


class ConditionWaiter:
    def __init__(
        self,
        observer: StateObserver,
        poll_seconds: float = 0.2,
        ambiguity_grace_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._observer = observer
        self._poll_seconds = poll_seconds
        self._ambiguity_grace_seconds = ambiguity_grace_seconds
        self._clock = clock
        self._sleep = sleep

    async def wait(self, condition: Predicate, timeout_ms: int) -> Observation:
        deadline = self._clock() + timeout_ms / 1000
        ambiguous_since: float | None = None
        while True:
            observation = await self._observer.observe(condition)
            now = self._clock()
            if observation.condition_met or observation.active_handlers or now >= deadline:
                return observation
            # Waiting cannot resolve a duplicate control. The grace period only absorbs render transitions.
            if not observation.ambiguous:
                ambiguous_since = None
            elif ambiguous_since is None:
                ambiguous_since = now
            elif now - ambiguous_since >= self._ambiguity_grace_seconds:
                return observation
            await self._sleep(self._poll_seconds)
