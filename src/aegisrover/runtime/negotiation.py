"""Capability negotiation and task-level degradation planning.

Rules: the capability name must match, the major version must match exactly, the
offered minor version must be at least the requested one (minor versions only add
features), and every required feature must be present. A requirement whose major is
``None`` only checks the capability name; this preserves the legacy name-only
mission scope.

Optional requirements are reported as degraded instead of failing the whole
negotiation. At task level, each task is evaluated independently so one robot's
unsupported camera cannot reject an unrelated navigation task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = (
    'Requirement', 'Offer', 'Rejection', 'Negotiation', 'NegotiationError',
    'TaskSpec', 'TaskDecision', 'DegradationPlan', 'negotiate', 'task',
    'plan_tasks', 'plan_degraded_operation', 'requirement_from_value',
    'offer_from_value',
)


@dataclass(frozen=True)
class Requirement:
    name: str
    major: int | None = None
    minor: int = 0
    features: frozenset[str] = frozenset()
    optional: bool = False


@dataclass(frozen=True)
class Offer:
    name: str
    major: int | None = None
    minor: int = 0
    features: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Rejection:
    name: str
    reason: str
    detail: str
    required: dict[str, Any] = field(default_factory=dict)
    offered: list[dict[str, Any]] = field(default_factory=list)
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'reason': self.reason,
            'detail': self.detail,
            'required': self.required,
            'offered': self.offered,
            'optional': self.optional,
        }


@dataclass(frozen=True)
class Negotiation:
    accepted: dict[str, Offer] = field(default_factory=dict)
    degraded: tuple[Rejection, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.rejected

    def require_ok(self) -> 'Negotiation':
        if not self.ok:
            raise NegotiationError(self.rejected)
        return self

    def to_dict(self) -> dict:
        return {
            'ok': self.ok,
            'accepted': {name: offer_to_dict(o) for name, o in self.accepted.items()},
            'degraded': [r.to_dict() for r in self.degraded],
            'rejected': [r.to_dict() for r in self.rejected],
        }


class NegotiationError(RuntimeError):
    def __init__(self, rejections: Iterable[Rejection]):
        self.rejections = tuple(rejections)
        super().__init__('; '.join(f'{r.name}: {r.reason}' for r in self.rejections))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    requirements: tuple[Requirement, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError('task_id is required')
        object.__setattr__(self, 'requirements', tuple(self.requirements))

    def to_dict(self) -> dict[str, Any]:
        return {'task_id': self.task_id,
                'requirements': [requirement_to_dict(r) for r in self.requirements]}


@dataclass(frozen=True)
class TaskDecision:
    task_id: str
    status: str
    selected: dict[str, Offer] = field(default_factory=dict)
    issues: tuple[Rejection, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.status in ('ready', 'degraded')

    @property
    def degraded(self) -> bool:
        return self.status == 'degraded'

    @property
    def blocked(self) -> bool:
        return self.status == 'blocked'

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status,
            'runnable': self.runnable,
            'selected': {name: offer_to_dict(o) for name, o in self.selected.items()},
            'issues': [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DegradationPlan:
    decisions: tuple[TaskDecision, ...] = ()

    @property
    def ready(self) -> tuple[TaskDecision, ...]:
        return tuple(d for d in self.decisions if d.status == 'ready')

    @property
    def degraded(self) -> tuple[TaskDecision, ...]:
        return tuple(d for d in self.decisions if d.status == 'degraded')

    @property
    def blocked(self) -> tuple[TaskDecision, ...]:
        return tuple(d for d in self.decisions if d.status == 'blocked')

    @property
    def runnable(self) -> tuple[TaskDecision, ...]:
        return tuple(d for d in self.decisions if d.runnable)

    @property
    def all_runnable(self) -> bool:
        return all(d.runnable for d in self.decisions)

    def get(self, task_id: str) -> TaskDecision:
        for decision in self.decisions:
            if decision.task_id == task_id:
                return decision
        raise KeyError(task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            'all_runnable': self.all_runnable,
            'ready': [d.task_id for d in self.ready],
            'degraded': [d.task_id for d in self.degraded],
            'blocked': [d.task_id for d in self.blocked],
            'runnable': [d.task_id for d in self.runnable],
            'decisions': [d.to_dict() for d in self.decisions],
        }


def requirement_to_dict(requirement: Requirement) -> dict[str, Any]:
    return {
        'name': requirement.name,
        'major': requirement.major,
        'minor': requirement.minor,
        'features': sorted(requirement.features),
        'optional': requirement.optional,
    }


def offer_to_dict(offer: Offer) -> dict[str, Any]:
    return {'name': offer.name, 'major': offer.major, 'minor': offer.minor,
            'features': sorted(offer.features)}


def requirement_from_value(value: Requirement | dict[str, Any] | str) -> Requirement:
    if isinstance(value, Requirement):
        return value
    if isinstance(value, str):
        return Requirement(value)
    if isinstance(value, dict):
        return Requirement(
            name=value['name'],
            major=value.get('major'),
            minor=int(value.get('minor', 0)),
            features=frozenset(value.get('features') or ()),
            optional=bool(value.get('optional', False)),
        )
    raise TypeError(f'cannot interpret capability requirement {value!r}')


def offer_from_value(value: Offer | dict[str, Any] | str) -> Offer:
    if isinstance(value, Offer):
        return value
    if isinstance(value, str):
        return Offer(value)
    if isinstance(value, dict):
        return Offer(
            name=value['name'],
            major=value.get('major'),
            minor=int(value.get('minor', 0)),
            features=frozenset(value.get('features') or ()),
        )
    raise TypeError(f'cannot interpret offered capability {value!r}')


def task(task_id: str, requirements: Iterable[Requirement | dict[str, Any] | str]) -> TaskSpec:
    return TaskSpec(task_id, tuple(requirement_from_value(r) for r in requirements))


def negotiate(requirements: Iterable[Requirement], offers: Iterable[Offer]) -> Negotiation:
    pool: dict[str, list[Offer]] = {}
    for raw_offer in offers:
        offer = raw_offer if isinstance(raw_offer, Offer) else offer_from_value(raw_offer)
        pool.setdefault(offer.name, []).append(offer)

    accepted: dict[str, Offer] = {}
    degraded: list[Rejection] = []
    rejected: list[Rejection] = []
    for raw_requirement in requirements:
        requirement = (raw_requirement if isinstance(raw_requirement, Requirement)
                       else requirement_from_value(raw_requirement))
        candidates = pool.get(requirement.name, [])
        choice, reason, detail = _select(requirement, candidates)
        if choice is not None:
            accepted[requirement.name] = choice
            missing = requirement.features - choice.features
            if requirement.optional and missing:
                degraded.append(_issue(requirement, choice, 'missing_optional_features',
                                       ','.join(sorted(missing)), [choice]))
            continue

        if requirement.optional:
            reason = 'missing_optional_features'
            if not detail:
                detail = 'optional capability is not usable'
        problem = _issue(requirement, None, reason, detail, candidates)
        (degraded if requirement.optional else rejected).append(problem)
    return Negotiation(accepted=accepted, degraded=tuple(degraded), rejected=tuple(rejected))


def plan_tasks(tasks: Iterable[TaskSpec], offers: Iterable[Offer]) -> DegradationPlan:
    """Return an independently runnable/degraded/blocked decision for each task."""
    normalized_offers = tuple(o if isinstance(o, Offer) else offer_from_value(o) for o in offers)
    decisions: list[TaskDecision] = []
    for raw_task in tasks:
        spec = raw_task if isinstance(raw_task, TaskSpec) else TaskSpec(
            raw_task[0], tuple(requirement_from_value(r) for r in raw_task[1]))
        report = negotiate(spec.requirements, normalized_offers)
        if report.rejected:
            status = 'blocked'
        elif report.degraded:
            status = 'degraded'
        else:
            status = 'ready'
        decisions.append(TaskDecision(
            task_id=spec.task_id,
            status=status,
            selected=dict(report.accepted),
            issues=report.rejected + report.degraded,
        ))
    return DegradationPlan(tuple(decisions))


def plan_degraded_operation(tasks: Iterable[TaskSpec], offers: Iterable[Offer]) -> DegradationPlan:
    """Public alias for callers planning a possibly degraded operation."""
    return plan_tasks(tasks, offers)


def _select(requirement: Requirement, candidates: list[Offer]) -> tuple[Offer | None, str, str]:
    if not candidates:
        return None, 'not_offered', 'robot does not provide this capability'

    if requirement.major is None:
        same_major = list(candidates)
    else:
        same_major = [c for c in candidates if c.major == requirement.major]
    if not same_major:
        majors = ','.join(sorted({str(c.major) for c in candidates}))
        return None, 'major_mismatch', f'offered majors {majors}, need {requirement.major}'

    compatible = [c for c in same_major if c.minor >= requirement.minor]
    if not compatible:
        newest = max(same_major, key=lambda c: c.minor)
        return None, 'minor_too_old', f'offered {newest.minor}, need {requirement.minor}'

    feature_complete = [c for c in compatible if requirement.features <= c.features]
    if feature_complete:
        return max(feature_complete, key=lambda c: c.minor), '', ''

    newest = max(compatible, key=lambda c: c.minor)
    missing = requirement.features - newest.features
    detail = ','.join(sorted(missing))
    if requirement.optional:
        # Prefer a complete offer above. When none exists, the base protocol is
        # usable; callers must avoid the missing features and the issue records
        # which features were disabled.
        return newest, 'missing_optional_features', detail
    return None, 'missing_features', detail


def _issue(requirement: Requirement, chosen: Offer | None, reason: str, detail: str,
           candidates: Iterable[Offer]) -> Rejection:
    offered = [offer_to_dict(candidate) for candidate in candidates]
    if chosen is not None:
        chosen_dict = offer_to_dict(chosen)
        if chosen_dict not in offered:
            offered.append(chosen_dict)
    return Rejection(
        name=requirement.name,
        reason=reason,
        detail=detail,
        required=requirement_to_dict(requirement),
        offered=offered,
        optional=requirement.optional,
    )
