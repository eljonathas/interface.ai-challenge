from __future__ import annotations

from playwright.async_api import Frame, Page

from interface_ai_challenge.domain.targets import FrameRef


class FrameDirectory:
    """Finds frames by the chain of iframe `name` attributes, never by position."""

    def __init__(self, page: Page) -> None:
        self._page = page

    def find(self, ref: FrameRef) -> Frame | None:
        frame = self._page.main_frame
        for name in ref.names:
            children = [
                candidate
                for candidate in self._page.frames
                if candidate.parent_frame is frame and candidate.name == name and not candidate.is_detached()
            ]
            if len(children) != 1:
                return None
            frame = children[0]
        return frame

    def all_frames(self) -> list[Frame]:
        return list(self._page.frames)

    def reference(self, frame: Frame) -> tuple[FrameRef, bool]:
        names: list[str] = []
        current: Frame | None = frame
        while current is not None and current.parent_frame is not None:
            names.append(current.name)
            current = current.parent_frame
        names.reverse()
        return FrameRef(names=tuple(names)), all(names)
