from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class ChunkMetadata(BaseModel):
    source_document_name: str
    source_document_id: str
    document_title: str
    document_version: str
    effective_date: str
    department: Optional[str] = None
    language: Literal["ru", "kz"] = "ru"
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    parent_rule_number: Optional[str] = None
    chunk_rule_number: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: Literal[
        "paragraph","parent_rule","list_item","table",
        "definition","abbreviation","appendix","appendix_table","section_title"
    ]
    cross_references: List[str] = []

class Chunk(BaseModel):
    chunk_id: str = Field(..., description="PK, e.g. 854-ЦЗ-3.25")
    content: str
    metadata: ChunkMetadata

class Evidence(BaseModel):
    cite: str                  # chunk_id
    doc: Dict[str, Any]        # {id,title,version,effective_date}
    loc: Dict[str, Any]        # {section_number,chunk_rule_number,page_number,chunk_type}
    text: str                  # короткая выдержка
    why: str                   # причина включения

class AnswerRequest(BaseModel):
    query: str
    language: Optional[Literal["ru","kz"]] = None
    k_rules: int = 100
    k_defs: int = 60
    need_versions_note: bool = True
    doc_id: Optional[str] = None
