from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, getcontext
import inspect
import json
import re
import unicodedata
from typing import Any

from .config import ROOT
from .epistemic.engine import EpistemicEngine


getcontext().prec = 28
CAPABILITIES_PATH = ROOT / "config" / "capabilities.json"


@dataclass
class RuntimeResponse:
    answer: str
    route: str
    deterministic: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def load_capabilities() -> dict[str, Any]:
    if not CAPABILITIES_PATH.exists():
        raise RuntimeError("Missing config/capabilities.json")
    value = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    if int(value.get("schema_version", 0)) != 1:
        raise RuntimeError("Unsupported capabilities schema.")
    return value


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(text.lower())).strip()


def _clean_exact_target(value: str) -> str | None:
    value = value.strip().strip('"“”\'')
    if not value or len(value) > 160 or "\n" in value or "\r" in value:
        return None
    return value


def parse_exact_output(prompt: str) -> str | None:
    raw = prompt.strip()
    low = _normalized(raw)

    command = re.search(
        r"\b(respond|contest|devolv|devuelv|escrib|pon|decime|dime|respuesta)\w*\b",
        low,
    )
    exactness = re.search(
        r"\b(solo|solamente|unicamente|exacta|exactamente|nada mas|sin texto|sin explicacion|sin agregar)\b",
        low,
    )
    if command and exactness:
        quoted = re.search(r'["“”\']([^"“”\']{1,160})["“”\']', raw)
        if quoted:
            return _clean_exact_target(quoted.group(1))

    patterns = [
        r"(?i)\brespuest[ao]\s+exact[ao]\s+(?P<t>[A-Za-z0-9_-]+)(?:\s+sin\b.*)?$",
        r"(?i)\b(?:respond[eé]?|contesta|contest[aá]?|devolv[eé]?|devuelve|escrib[ií]?|pone|pon[eé]?|decime|dime)\s+(?:solamente|solo|[uú]nicamente|exactamente)(?:\s+con)?(?:\s+la\s+palabra)?\s+(?P<t>[A-Za-z0-9_-]+)(?:\s+y\s+nada\s+m[aá]s|\s+nada\s+m[aá]s|\s+sin\b.*)?$",
        r"(?i)\b(?:tu\s+respuesta|la\s+[uú]nica\s+respuesta|tu\s+salida|la\s+[uú]nica\s+cosa\s+que\s+debes\s+devolver)\b.*?\b(?:solo|solamente|sea|ser)\s+(?P<t>[A-Za-z0-9_-]+)(?:\s+y?\s*nada\s+m[aá]s|\s+sin\b.*)?$",
        r"(?i)\b(?:solo|solamente|[uú]nicamente)\s+(?:devolv[eé]?|devuelve|respond[eé]?|contesta|escrib[ií]?|pone|pon[eé]?)\s+(?P<t>[A-Za-z0-9_-]+)(?:\s+y\s+nada\s+m[aá]s|\s+sin\b.*)?$",
        r"(?i)\b(?:sin\s+comillas(?:\s+ni\s+explicaci[oó]n)?|sin\s+explicaci[oó]n|sin\s+texto\s+extra)\b.*?\b(?:respond[eé]?|contesta|devolv[eé]?|escrib[ií]?|pone|pon[eé]?)\s+(?P<t>[A-Za-z0-9_-]+)\s*$",
        r"(?i)\b(?:escrib[ií]?|devolv[eé]?|respond[eé]?|contesta|pone|pon[eé]?)\s+(?P<t>[A-Za-z0-9_-]+)\s+(?:sin\s+agregar\b.*|sin\s+texto\b.*|y\s+nada\s+m[aá]s)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return _clean_exact_target(match.group("t"))
    return None


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "."))


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_arithmetic(prompt: str) -> str | None:
    raw = prompt.strip()
    low = _normalized(raw)
    numbers = re.findall(r"\d+(?:[.,]\d+)?", low)
    if len(numbers) != 2:
        return None
    if "=" in low or re.search(
        r"\b(me crees|esta bien|es correcto|es verdad|dice|dijo|me dijeron|afirman?)\b",
        low,
    ):
        return None

    symbolic = re.search(
        r"(\d+(?:[.,]\d+)?)\s*([+\-*/x×])\s*(\d+(?:[.,]\d+)?)",
        low,
    )
    if symbolic:
        a, op, b = symbolic.group(1), symbolic.group(2), symbolic.group(3)
    else:
        worded = re.search(
            r"(\d+(?:[.,]\d+)?)\s+(mas|menos|por|dividido por|entre)\s+(\d+(?:[.,]\d+)?)",
            low,
        )
        if not worded:
            return None
        a, word, b = worded.group(1), worded.group(2), worded.group(3)
        op = {"mas": "+", "menos": "-", "por": "*", "dividido por": "/", "entre": "/"}[word]

    direct_marker = re.search(
        r"\b(cuanto|resultado|responde|respuesta|dame|solo|nada mas|sin explicar|calcula|calcular|resuelve|da)\b",
        low,
    )
    bare_expression = re.fullmatch(
        r"\s*\d+(?:[.,]\d+)?\s*[+\-*/x×]\s*\d+(?:[.,]\d+)?\s*",
        low,
    )
    if not direct_marker and not bare_expression:
        return None

    left, right = _decimal(a), _decimal(b)
    op = {"x": "*", "×": "*"}.get(op, op)
    if op == "+":
        result = left + right
    elif op == "-":
        result = left - right
    elif op == "*":
        result = left * right
    elif op == "/":
        if right == 0:
            return "No se puede dividir por cero."
        result = left / right
    else:
        return None
    return _format_decimal(result)


def parse_arithmetic_truth_claim(prompt: str) -> RuntimeResponse | None:
    low = _normalized(prompt)

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*([+\-*/x×])\s*(\d+(?:[.,]\d+)?)\s*=\s*(\d+(?:[.,]\d+)?)",
        low,
    )
    if not match:
        return None

    asks_truth = re.search(
        r"\b(me crees|esta bien|es correcto|es correcta|es verdad|es cierto|"
        r"correcto|correcta|verdad|cierto|te parece|esta correcto|esta correcta)\b",
        low,
    )
    if not asks_truth:
        return None

    left_raw, op, right_raw, claimed_raw = match.groups()
    op = {"x": "*", "×": "*"}.get(op, op)
    left = _decimal(left_raw)
    right = _decimal(right_raw)
    claimed = _decimal(claimed_raw)

    if op == "+":
        actual = left + right
    elif op == "-":
        actual = left - right
    elif op == "*":
        actual = left * right
    elif op == "/":
        if right == 0:
            return RuntimeResponse(
                "No se puede verificar como verdadera porque implica division por cero.",
                "epistemic_verification",
                True,
                {"capability": "epistemic_verification", "status": "CONTRADICTED"},
            )
        actual = left / right
    else:
        return None

    canonical_claim = (
        f"{_format_decimal(left)} {op} {_format_decimal(right)} = "
        f"{_format_decimal(claimed)}"
    )
    verification = EpistemicEngine().verify(canonical_claim)
    status = verification.status.value

    if actual == claimed:
        answer = (
            f"Si. {_format_decimal(left)} {op} {_format_decimal(right)} "
            f"= {_format_decimal(actual)}."
        )
    else:
        answer = (
            f"No. {_format_decimal(left)} {op} {_format_decimal(right)} "
            f"= {_format_decimal(actual)}, no {_format_decimal(claimed)}."
        )

    return RuntimeResponse(
        answer,
        "epistemic_verification",
        True,
        {
            "capability": "epistemic_verification",
            "status": status,
            "claim": canonical_claim,
        },
    )


def parse_explicit_unknown(prompt: str) -> RuntimeResponse | None:
    low = _normalized(prompt)

    invented = re.search(
        r"\b(invente|inventado|inventada|me invente|es ficticio|es ficticia)\b",
        low,
    )
    missing = re.search(
        r"\b(nunca dije|no te dije|no te conte|nunca te conte|"
        r"no conte|no mencione|nunca mencione|no te mencione)\b",
        low,
    )
    if not invented or not missing:
        return None

    question = re.search(
        r"\b(cual|que|como|donde|quien|cuanto|cuanta|cuantos|cuantas)\b",
        low,
    )
    if not question:
        return None

    return RuntimeResponse(
        "No tengo informacion suficiente para saberlo; ese dato no fue proporcionado.",
        "epistemic_unknown",
        True,
        {
            "capability": "epistemic_verification",
            "status": "UNKNOWN",
        },
    )


def route_deterministic(prompt: str) -> RuntimeResponse | None:
    arithmetic = parse_arithmetic(prompt)
    if arithmetic is not None:
        return RuntimeResponse(arithmetic, "arithmetic", True, {"capability": "arithmetic"})

    exact = parse_exact_output(prompt)
    if exact is not None:
        return RuntimeResponse(exact, "exact_binding", True, {"capability": "exact_binding"})

    arithmetic_claim = parse_arithmetic_truth_claim(prompt)
    if arithmetic_claim is not None:
        return arithmetic_claim

    explicit_unknown = parse_explicit_unknown(prompt)
    if explicit_unknown is not None:
        return explicit_unknown

    return None


def _extract_neural_answer(output: str, shaped: str) -> str:
    answer = output[len(shaped):]
    for marker in ("<END>", "\nUser:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    return answer.strip()


class ButterflyRuntime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def neural_respond(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        *,
        max_new_tokens: int = 150,
        temperature: float = 0.58,
        top_k: int = 50,
        repetition_penalty: float = 1.24,
        min_new_tokens: int = 1,
    ) -> RuntimeResponse:
        from .generation import generate

        history = history or []
        context = "".join(
            f"User: {user}\nButterfly: {answer}\n<END>\n"
            for user, answer in history[-4:]
        )
        shaped = context + f"User: {prompt}\nButterfly:"
        output = generate(
            self.model,
            shaped,
            self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
        )
        return RuntimeResponse(
            _extract_neural_answer(output, shaped),
            "neural",
            False,
            {"capability": "neural"},
        )

    def respond(
        self,
        prompt: str,
        history: list[tuple[str, str]] | None = None,
        **generation_kwargs,
    ) -> RuntimeResponse:
        deterministic = route_deterministic(prompt)
        if deterministic is not None:
            return deterministic
        return self.neural_respond(prompt, history, **generation_kwargs)


def runtime_fingerprint_payload() -> dict[str, Any]:
    return {
        "capabilities": load_capabilities(),
        "parse_exact_output": inspect.getsource(parse_exact_output),
        "parse_arithmetic": inspect.getsource(parse_arithmetic),
        "parse_arithmetic_truth_claim": inspect.getsource(parse_arithmetic_truth_claim),
        "parse_explicit_unknown": inspect.getsource(parse_explicit_unknown),
        "route_deterministic": inspect.getsource(route_deterministic),
        "runtime_respond": inspect.getsource(ButterflyRuntime.respond),
    }
