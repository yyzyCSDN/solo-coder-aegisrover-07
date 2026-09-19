"""Task-level capability fallback planning.

Capability negotiation should not turn one mismatched optional capability into an
all-or-nothing session failure. Each task is evaluated independently: mandatory
requirements gate execution, while optional requirements may become recorded
degradations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from aegisrover.runtime.negotiation import Offer, Requirement, negotiate

__all__ = (
    'TaskRequirements', 'CapabilityIssue', 'TaskDecision', 'FallbackPlan',
    'plan_fallback', 'record_fallback_plan',
)

READY = 'ready'
DEGRADED = 'degraded'
BLOCKED = 'blocked'


@dataclass(frozen=True)
class TaskRequirements:
    task_id: str
    required: tuple[Requirement, ...] = ()
    optional: tuple[Requirement, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id is required')

    @classmethod
    def create(cls, task_id: str, required: Iterable[Requirement] = (),
               optional: Iterable[Requirement] = ()) -> 'TaskRequirements':
        return cls(
            task_id,
            tuple(required),
            tuple(Requirement(
                name=item.name,
                major=item.major,
                minor=item.minor,
                features=item.features,
                optional=True,
            ) for item in optional),
        )


@dataclass(frozen=True)
class CapabilityIssue:
    capability: str
    reason: str
    detail: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'capability': self.capability,
            'reason': self.reason,
            'detail': self.detail,
            'severity': self.severity,
        }


@dataclass(frozen=True)
class TaskDecision:
    task_id: str
    status: str
    accepted: dict[str, Offer] = field(default_factory=dict)
    issues: tuple[CapabilityIssue, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.status in (READY, DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status,
            'runnable': self.runnable,
            'accepted': {
                name: {
                    'major': offer.major,
                    'minor': offer.minor,
                    'features': sorted(offer.features),
                }
                for name, offer in self.accepted.items()
            },
            'issues': [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class FallbackPlan:
    decisions: tuple[TaskDecision, ...]

    @property
    def runnable(self) -> tuple[TaskDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.runnable)

    @property
    def blocked(self) -> tuple[TaskDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.status == BLOCKED)

    @property
    def degraded(self) -> tuple[TaskDecision, ...]:
        return tuple(decision for decision in self.decisions if decision.status == DEGRADED)

    @property
    def ok(self) -> bool:
        return not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'runnable_task_ids': [decision.task_id for decision in self.runnable],
            'blocked_task_ids': [decision.task_id for decision in self.blocked],
            'degraded_task_ids': [decision.task_id for decision in self.degraded],
            'tasks': [decision.to_dict() for decision in self.decisions],
        }


def plan_fallback(tasks: Iterable[TaskRequirements], offers: Iterable[Offer]) -> FallbackPlan:
    """Return which tasks may run normally, degraded, or must be blocked."""
    available = tuple(offers)
    decisions = []
    for task in tasks:
        requirements = (*task.required, *task.optional)
        report = negotiate(requirements, available)
        accepted = {
            requirement.name: report.accepted[requirement.name]
            for requirement in (*task.required, *task.optional)
            if requirement.name in report.accepted
        }
        issues: list[CapabilityIssue] = []
        blocked = False
        for requirement in (*task.required, *task.optional):
            rejection = _find_rejection(report, requirement.name, requirement.optional)
            if rejection is None:
                continue
            severity = 'blocking' if requirement in task.required else 'degraded'
            issues.append(CapabilityIssue(
                capability=requirement.name,
                reason=rejection.reason,
                detail=rejection.detail,
                severity=severity,
            ))
            blocked = blocked or requirement in task.required
        if blocked:
            status = BLOCKED
        elif issues:
            status = DEGRADED
        else:
            status = READY
        decisions.append(TaskDecision(task.task_id, status, accepted, tuple(issues)))
    return FallbackPlan(tuple(decisions))


def record_fallback_plan(plan: FallbackPlan, *, actor: str = 'planner',
                         audit=None, events=None, subject: str = 'capability-fallback') -> dict[str, Any]:
    """Persist a compact, tamper-evident audit reason and detailed event payload."""
    payload = plan.to_dict()
    if audit is not None:
        audit.append(
            actor,
            'capability.fallback',
            subject,
            {
                'runnable': payload['runnable_task_ids'],
                'degraded': payload['degraded_task_ids'],
                'blocked': payload['blocked_task_ids'],
            },
        )
    if events is not None:
        events.append('capability.fallback', payload)
    return payload


def _find_rejection(report, name: str, optional: bool):
    source = report.degraded if optional else report.rejected
    for rejection in source:
        if rejection.name == name:
            return rejection
    return None
