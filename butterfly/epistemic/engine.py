from .types import VerificationResult, ClaimStatus
from .math_verifier import MathVerifier
from .web import WebResearch
from ..memory import MemoryStore


class EpistemicEngine:
    def __init__(self, memory=None):
        self.memory = memory or MemoryStore()
        self.math = MathVerifier()
        self.web = WebResearch()

    def verify(self, claim: str, allow_web=False):
        if self.math.can_handle(claim):
            result = self.math.verify(claim)
        elif allow_web:
            try:
                evidence = self.web.wikipedia_search(claim)
                # Web snippets are evidence, not proof. We deliberately do not auto-promote to VERIFIED.
                result = VerificationResult(
                    claim=claim,
                    status=ClaimStatus.UNCERTAIN if evidence else ClaimStatus.UNKNOWN,
                    confidence=min(0.65, sum(e.trust for e in evidence) / max(1, len(evidence))),
                    method="web_research",
                    evidence=evidence,
                    explanation="Web search found related evidence, but v0.0001 does not equate search results with truth.",
                )
            except Exception as e:
                result = VerificationResult(claim, ClaimStatus.UNKNOWN, 0.0, "web_research", explanation=str(e))
        else:
            result = VerificationResult(
                claim=claim,
                status=ClaimStatus.UNKNOWN,
                confidence=0.0,
                method="none",
                explanation="No deterministic verifier matched. Use --web or a domain-specific verifier.",
            )
        self.memory.add_claim(
            result.claim,
            result.status.value,
            result.confidence,
            result.method,
            [e.source for e in result.evidence],
            result.explanation,
        )
        return result
