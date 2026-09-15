import asyncio

import pytest

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.domain.control import ActorKind, ControlState, ControlToken
from interface_ai_challenge.domain.errors import ControlError


async def test_automation_token_is_invalidated_by_pause() -> None:
    control = SessionControl()
    token = control.automation_token()
    await control.pause_for_human()
    with pytest.raises(ControlError):
        async with control.acting(token):
            pass
    with pytest.raises(ControlError):
        control.automation_token()


async def test_only_one_operator_controls_and_hands_back() -> None:
    control = SessionControl()
    await control.pause_for_human()
    first = await control.claim("operator-1")
    with pytest.raises(ControlError):
        await control.claim("operator-2")
    with pytest.raises(ControlError):
        await control.release_to_automation("operator-2")
    async with control.acting(first):
        pass
    await control.release_to_automation("operator-1")
    assert control.snapshot().state is ControlState.AUTOMATION
    with pytest.raises(ControlError):
        async with control.acting(first):
            pass


async def test_forged_epoch_is_rejected() -> None:
    control = SessionControl()
    await control.pause_for_human()
    token = await control.claim("operator-1")
    forged = ControlToken(actor=ActorKind.HUMAN, actor_id="operator-1", epoch=token.epoch - 1)
    with pytest.raises(ControlError):
        async with control.acting(forged):
            pass


async def test_pause_waits_for_in_flight_action() -> None:
    control = SessionControl()
    order: list[str] = []
    action_started = asyncio.Event()

    async def action() -> None:
        async with control.acting(control.automation_token()):
            action_started.set()
            await asyncio.sleep(0.05)
            order.append("action finished")

    task = asyncio.create_task(action())
    await action_started.wait()
    await control.pause_for_human()
    order.append("paused")
    await task
    assert order == ["action finished", "paused"]


async def test_terminated_session_accepts_nobody() -> None:
    control = SessionControl()
    token = control.automation_token()
    await control.terminate()
    with pytest.raises(ControlError):
        async with control.acting(token):
            pass
    with pytest.raises(ControlError):
        await control.claim("operator-1")
