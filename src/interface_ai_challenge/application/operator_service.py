from __future__ import annotations

from typing import Any

from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.intervention_inbox import InMemoryInterventionInbox
from interface_ai_challenge.domain.computer import ClickCommand, ComputerCommand, ScrollCommand, TypeCommand
from interface_ai_challenge.domain.control import ActorKind, ControlState, ControlToken
from interface_ai_challenge.domain.errors import ControlError, StaleObservationError
from interface_ai_challenge.domain.geometry import Screenshot
from interface_ai_challenge.domain.intervention import InterventionDecision
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.surface import ScreenCapture


class OperatorService:
    """Use cases of the human operator on the live session; transport-agnostic."""

    def __init__(
        self,
        control: SessionControl,
        inbox: InMemoryInterventionInbox,
        commands: ComputerCommandExecutor,
        screen: ScreenCapture,
        evidence: EvidenceSink,
    ) -> None:
        self._control = control
        self._inbox = inbox
        self._commands = commands
        self._screen = screen
        self._evidence = evidence
        self._latest_revision = -1

    def status(self) -> dict[str, Any]:
        request = self._inbox.current()
        return {
            "control": self._control.snapshot().model_dump(mode="json"),
            "intervention": request.model_dump(mode="json") if request else None,
        }

    async def screenshot(self) -> Screenshot:
        screenshot = await self._screen.capture()
        self._latest_revision = screenshot.revision
        return screenshot

    async def claim(self, operator_id: str) -> ControlToken:
        if self._inbox.current() is None:
            raise ControlError("there is no pending intervention to take over")
        token = await self._control.claim(operator_id)
        self._evidence.emit("control_claimed", {"operator_id": operator_id, "control_epoch": token.epoch})
        return token

    async def act(self, operator_id: str, epoch: int, revision: int, command: ComputerCommand) -> None:
        if isinstance(command, ClickCommand | ScrollCommand) and revision != self._latest_revision:
            raise StaleObservationError("the action refers to an outdated screenshot")
        token = ControlToken(actor=ActorKind.HUMAN, actor_id=operator_id, epoch=epoch)
        report = await self._commands.execute(token, command)
        self._evidence.emit(
            "human_action",
            {
                "operator_id": operator_id,
                "control_epoch": epoch,
                "command": _sanitized(command),
                "effect": report.effect,
                "control": {"role": report.descriptor.role, "name": report.descriptor.name}
                if report.descriptor and report.descriptor.role != "textbox"
                else None,
            },
        )

    async def hand_back(self, operator_id: str) -> None:
        request = self._require_request()
        await self._control.release_to_automation(operator_id)
        self._evidence.emit("control_returned", {"operator_id": operator_id, "intervention_id": request.id})
        self._inbox.decide(request.id, InterventionDecision.RESUME)

    async def abort(self, operator_id: str) -> None:
        request = self._require_request()
        snapshot = self._control.snapshot()
        if snapshot.state is ControlState.HUMAN and snapshot.owner_id != operator_id:
            raise ControlError("another operator is in control")
        self._evidence.emit("intervention_aborted", {"operator_id": operator_id, "intervention_id": request.id})
        self._inbox.decide(request.id, InterventionDecision.ABORT)

    def _require_request(self) -> Any:
        request = self._inbox.current()
        if request is None:
            raise ControlError("there is no pending intervention")
        return request


def _sanitized(command: ComputerCommand) -> dict[str, Any]:
    if isinstance(command, TypeCommand):
        return {"kind": "type", "text_length": len(command.text)}
    return command.model_dump(mode="json")
