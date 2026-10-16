"""Các kiểu dữ liệu dùng chung (yêu cầu 2.3): dataclass cho dữ liệu, TypedDict cho thông điệp."""
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


@dataclass
class Document:
    doc_id: str
    title: str
    version: int
    date: str | None
    owner: str
    file: str
    section: str


@dataclass
class Source:
    doc_id: str
    chunk_id: str
    version: int = 1


@dataclass
class Chunk:
    source: Source
    text: str
    document: Document
    kind: Literal["prose", "table_row"] = "prose"
    code: str | None = None  # mã hoạt động nếu chunk là một dòng của bảng


@dataclass
class Activity:
    """Một dòng của bảng mã hóa (mục II.2.a)."""
    code: str
    category: str
    description: str
    points_min: float
    points_max: float
    days_min: float
    days_max: float


@dataclass
class Action:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    status: Literal["answered", "no_answer", "refused", "error"] = "answered"
    verified: bool | None = None  # chỉ có nghĩa khi câu hỏi có ghi dữ liệu
    steps: int = 0


class ToolSpec(TypedDict):
    name: str
    description: str
    inputSchema: dict[str, Any]


class Observation(TypedDict):
    step: int
    tool: str
    arguments: dict[str, Any]
    result: str
