from dataclasses import dataclass, field


@dataclass
class Source:
    doc_id: str
    chunk_id: str
    version: int = 1


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)

@dataclass
class Document:
    doc_id: str
    title: str
    version: int
    date: str | None
    owner: str
    file: str

@dataclass
class Chunk:
    source: Source
    text: str
    document: Document