from dataclasses import dataclass, asdict


@dataclass
class PostflightReport:
    task: str
    success_claimed: bool
    checks: dict
    passed: bool

    def to_dict(self):
        return asdict(self)


class PostflightEvaluator:
    def evaluate(self, task: str, checks: dict):
        # Every supplied check must pass. Unknown checks should be represented as False/None.
        passed = bool(checks) and all(v is True for v in checks.values())
        return PostflightReport(task, success_claimed=passed, checks=checks, passed=passed)
