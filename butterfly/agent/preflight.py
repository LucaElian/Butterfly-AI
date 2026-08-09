from dataclasses import dataclass, asdict
import re


@dataclass
class PreflightReport:
    task: str
    risk: int
    missing_information: list[str]
    assumptions: list[str]
    verification_plan: list[str]
    requires_confirmation: bool
    can_proceed: bool

    def to_dict(self):
        return asdict(self)


class PreflightEvaluator:
    HIGH_RISK = ("delete", "erase", "format", "registry", "password", "credential", "purchase", "payment")
    BULK = ("all files", "todos los archivos", "hundreds", "cientos", "entire folder", "carpeta completa")

    def evaluate(self, task: str):
        low = task.lower()
        risk = 0
        missing = []
        assumptions = []
        plan = ["Check that required resources exist", "Validate result after execution"]
        if any(k in low for k in self.HIGH_RISK):
            risk = 4
            plan.insert(0, "Create recovery/backup plan")
        elif any(k in low for k in self.BULK):
            risk = 3
            plan.insert(0, "Create backup or reversible transaction")
        elif any(k in low for k in ("modify", "edit", "change", "correg", "modific")):
            risk = 2
        elif any(k in low for k in ("open", "read", "find", "abr", "leer", "buscar")):
            risk = 0
        else:
            risk = 1

        vague_terms = ("fix it", "arreglalo", "correct it", "corregilo", "make it better", "mejoralo")
        if any(v in low for v in vague_terms):
            missing.append("Success criteria are ambiguous")
        if "correct salaries" in low or "corregir salarios" in low:
            missing.append("Salary correction rule/source")
        requires = risk >= 4 or bool(missing)
        return PreflightReport(task, risk, missing, assumptions, plan, requires, not requires)
