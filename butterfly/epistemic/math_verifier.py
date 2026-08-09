import ast
import operator
import re
from .types import VerificationResult, ClaimStatus, Evidence

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left), visit(node.right)
            if abs(left) > 1e12 or abs(right) > 1e12:
                raise ValueError("number too large")
            return OPS[type(node.op)](left, right)
        raise ValueError("unsupported expression")
    return visit(ast.parse(expr, mode="eval"))


class MathVerifier:
    EQ = re.compile(r"^\s*(.+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")

    def can_handle(self, claim: str):
        return bool(self.EQ.match(claim))

    def verify(self, claim: str):
        m = self.EQ.match(claim)
        if not m:
            return None
        expression, asserted = m.groups()
        try:
            actual = safe_eval(expression.replace("^", "**"))
            asserted_value = float(asserted)
            ok = abs(float(actual) - asserted_value) < 1e-9
            return VerificationResult(
                claim=claim,
                status=ClaimStatus.VERIFIED if ok else ClaimStatus.CONTRADICTED,
                confidence=1.0,
                method="deterministic_calculation",
                evidence=[Evidence("calculator", f"{expression} = {actual}", 1.0, ok)],
                explanation=f"Calculated result is {actual}.",
            )
        except Exception as e:
            return VerificationResult(claim, ClaimStatus.UNKNOWN, 0.0, "deterministic_calculation", explanation=str(e))
