"""Labelled stream metadata shared by recognition and translation context.

Metadata is scene context, not proof of who spoke a given utterance.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from .glossary import Glossary

MAX_CAST = 4


@dataclass(frozen=True)
class StreamContext:
    title: str = ""
    channel: str = ""
    host: str = ""
    host_line: str = ""
    cast: tuple[str, ...] = ()
    chapters: tuple[tuple[float, str], ...] = ()

    def chapter_at(self, position_s: float | None) -> str:
        if position_s is None or not self.chapters:
            return ""
        index = bisect_right([start for start, _ in self.chapters], position_s) - 1
        return self.chapters[index][1] if index >= 0 else ""

    def lines(self, position_s: float | None = None) -> list[str]:
        # Most stable first so consecutive prompts share a cacheable prefix.
        chapter = self.chapter_at(position_s)
        return [line for line in (
            f"配信タイトル：{self.title}" if self.title else "",
            f"チャンネル：{self.channel}" if self.channel else "",
            self.host_line,
            "共演：" + "、".join(self.cast) if self.cast else "",
            f"現在のチャプター：{chapter}" if chapter else "",
        ) if line]

    def recognition_text(self) -> str:
        return "\n".join(part for part in (self.title, self.channel, self.host, *self.cast) if part)


def describe_stream(metadata: dict[str, Any] | None, glossary: Glossary | None = None) -> StreamContext:
    meta = metadata or {}
    title = " ".join(str(meta.get("title") or "").split())[:160]
    channel = " ".join(str(meta.get("channel") or "").split())[:80]
    host = ""
    host_line = ""
    cast: list[str] = []
    if glossary is not None and glossary.talents:
        hosts = glossary.talents_in(channel)
        if hosts:
            host = hosts[0]
            host_line = glossary.host_line(host)
        sources = (title, " ".join(meta.get("tags") or []), str(meta.get("description") or ""))
        for name in glossary.talents_in("\n".join(sources)):
            if name != host and len(cast) < MAX_CAST:
                cast.append(name)
    chapters = []
    for chapter in meta.get("chapters") or []:
        name = " ".join(str(chapter.get("title") or "").split())[:60]
        if name and chapter.get("start_s") is not None:
            chapters.append((float(chapter["start_s"]), name))
    return StreamContext(title=title, channel=channel, host=host, host_line=host_line,
                         cast=tuple(cast), chapters=tuple(sorted(chapters)))
