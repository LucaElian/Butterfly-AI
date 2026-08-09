from dataclasses import dataclass, field
from enum import Enum


class ClaimStatus(str, Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


@dataclass
class Evidence:
    source: str
    excerpt: str = ""
    trust: float = 0.5
    supports: bool | None = None


@dataclass
class VerificationResult:
    claim: str
    status: ClaimStatus
    confidence: float
    method: str
    evidence: list[Evidence] = field(default_factory=list)
    explanation: str = ""
