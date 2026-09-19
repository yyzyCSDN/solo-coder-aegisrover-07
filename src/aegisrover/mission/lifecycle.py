"""Mission lifecycle with guarded transitions, persistence and audit.

Every mission is stored as a versioned record. Commands are applied through a
transition table so an operator cannot move a completed mission back into a
running state, and re-sending the same command is idempotent rather than an
error (operators retry after a flaky console connection).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from aegisrover.runtime.negotiation import (
    DegradationPlan, Offer as NegotiationOffer, Requirement, TaskSpec,
    offer_from_value, plan_tasks, requirement_from_value, requirement_to_dict,
)
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import NotFound, Repository, VersionConflict

__all__ = (
    'MISSION_NAMESPACE', 'STATES', 'TERMINAL_STATES', 'TRANSITIONS',
    'Mission', 'MissionError', 'InvalidTransition', 'MissionService',
    'capability_requirements_from_mission',
)

MISSION_NAMESPACE = 'missions'

STATES = ('draft', 'queued', 'assigned', 'running', 'paused', 'completed', 'failed', 'cancelled')
TERMINAL_STATES = ('completed', 'cancelled')

#: command -> {state the command is valid from: state it leads to}
TRANSITIONS: dict[str, dict[str, str]] = {
    'queue': {'draft': 'queued'},
    'assign': {'queued': 'assigned', 'failed': 'assigned'},
    'release': {'assigned': 'queued'},
    'start': {'assigned': 'running'},
    'pause': {'running': 'paused'},
    'resume': {'paused': 'running'},
    'complete': {'running': 'completed'},
    'fail': {'running': 'failed', 'paused': 'failed'},
    'cancel': {'draft': 'cancelled', 'queued': 'cancelled', 'assigned': 'cancelled',
               'running': 'cancelled', 'paused': 'cancelled'},
    'retry': {'failed': 'queued'},
}


class MissionError(RuntimeError):
    pass


class InvalidTransition(MissionError):
    def __init__(self, mission_id: str, command: str, state: str):
        super().__init__(f'{mission_id}: cannot {command} while {state}')
        self.mission_id = mission_id
        self.command = command
        self.state = state


@dataclass(frozen=True)
class Mission:
    mission_id: str
    state: str
    priority: int
    waypoints: tuple[tuple[float, float], ...]
    required_capabilities: frozenset[str] = frozenset()
    capability_requirements: tuple[dict[str, Any], ...] = ()
    capability_status: str = 'ready'
    capability_issues: tuple[dict[str, Any], ...] = ()
    assigned_to: str | None = None
    retries: int = 0
    revision: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    history: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict:
        return {
            'mission_id': self.mission_id,
            'state': self.state,
            'priority': self.priority,
            'waypoints': [list(w) for w in self.waypoints],
            'required_capabilities': sorted(self.required_capabilities),
            'capability_requirements': [dict(r) for r in self.capability_requirements],
            'capability_status': self.capability_status,
            'capability_issues': [dict(issue) for issue in self.capability_issues],
            'assigned_to': self.assigned_to,
            'retries': self.retries,
            'revision': self.revision,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'labels': dict(self.labels),
            'history': [dict(h) for h in self.history],
        }

    @staticmethod
    def from_dict(payload: dict) -> 'Mission':
        return Mission(
            mission_id=payload['mission_id'],
            state=payload['state'],
            priority=int(payload['priority']),
            waypoints=tuple((float(x), float(y)) for x, y in payload['waypoints']),
            required_capabilities=frozenset(payload.get('required_capabilities') or ()),
            capability_requirements=tuple(
                requirement_to_dict(requirement_from_value(item))
                for item in (payload.get('capability_requirements') or ())
            ),
            capability_status=payload.get('capability_status', 'ready'),
            capability_issues=tuple(payload.get('capability_issues') or ()),
            assigned_to=payload.get('assigned_to'),
            retries=int(payload.get('retries', 0)),
            revision=int(payload.get('revision', 1)),
            created_at=float(payload.get('created_at', 0.0)),
            updated_at=float(payload.get('updated_at', 0.0)),
            labels=dict(payload.get('labels') or {}),
            history=tuple(payload.get('history') or ()),
        )


def capability_requirements_from_mission(mission: Mission) -> tuple[Requirement, ...]:
    return tuple(requirement_from_value(item) for item in mission.capability_requirements)


def _normalize_capability_requirements(
    capability_requirements: Iterable[Any],
) -> tuple[dict[str, Any], ...]:
    values = [requirement_from_value(value) for value in capability_requirements]
    return tuple(requirement_to_dict(value) for value in values)


class MissionService:
    """Mission commands with optimistic concurrency and an audit trail."""

    def __init__(self, repository: Repository, audit: AuditLog | None = None, clock=time.time):
        self._repo = repository
        self._audit = audit if audit is not None else AuditLog(repository, clock)
        self._clock = clock

    # -- creation --------------------------------------------------------------
    def create(self, mission_id: str, waypoints: Iterable[Iterable[float]], *, priority: int = 0,
               required_capabilities: Iterable[str] = (),
               capability_requirements: Iterable[Any] = (), labels: dict | None = None,
               actor: str = 'operator') -> Mission:
        if not mission_id:
            raise MissionError('mission_id is required')
        points = tuple((float(p[0]), float(p[1])) for p in waypoints)
        if not points:
            raise MissionError('a mission needs at least one waypoint')
        now = self._clock()
        normalized_requirements = _normalize_capability_requirements(capability_requirements)
        legacy_names = [name for name in required_capabilities
                        if name not in {item['name'] for item in normalized_requirements}]
        normalized_requirements += tuple(requirement_to_dict(Requirement(name))
                                         for name in legacy_names)
        mission = Mission(
            mission_id=mission_id,
            state='draft',
            priority=int(priority),
            waypoints=points,
            required_capabilities=frozenset(required_capabilities),
            capability_requirements=normalized_requirements,
            labels=dict(labels or {}),
            created_at=now,
            updated_at=now,
            history=({'at': now, 'command': 'create', 'actor': actor, 'from': None, 'to': 'draft'},),
        )
        self._repo.put(MISSION_NAMESPACE, mission_id, mission.to_dict())
        self._audit.append(actor, 'mission.create', mission_id,
                           {'priority': mission.priority, 'waypoints': len(points),
                            'capability_requirements': len(normalized_requirements)})
        return mission

    # -- commands --------------------------------------------------------------
    def command(self, mission_id: str, command: str, *, actor: str = 'operator',
                expected_revision: int | None = None, reason: str = '',
                assignee: str | None = None, metadata: dict[str, Any] | None = None) -> dict:
        if command not in TRANSITIONS:
            raise MissionError(f'unknown command {command!r}')
        record = self._repo.get(MISSION_NAMESPACE, mission_id)
        mission = Mission.from_dict(record.payload)
        if expected_revision is not None and expected_revision != mission.revision:
            raise VersionConflict(MISSION_NAMESPACE, mission_id, expected_revision, mission.revision)
        allowed = TRANSITIONS[command]
        if mission.state not in allowed:
            # Replaying a command that already produced the target state is a no-op.
            already = any(mission.state == target for target in allowed.values())
            if not already:
                raise InvalidTransition(mission_id, command, mission.state)
            return {'mission': mission.to_dict(), 'applied': False, 'reason': 'already_applied'}
        target = allowed[mission.state]
        now = self._clock()
        updates: dict[str, Any] = {
            'state': target,
            'revision': mission.revision + 1,
            'updated_at': now,
        }
        if command == 'assign':
            if not assignee:
                raise MissionError('assign needs an assignee')
            updates['assigned_to'] = assignee
            updates['capability_status'] = (metadata or {}).get('capability_status', 'ready')
            updates['capability_issues'] = tuple((metadata or {}).get('capability_issues', ()))
        if command == 'release':
            updates['assigned_to'] = None
            updates['capability_status'] = 'ready'
            updates['capability_issues'] = ()
        if command == 'retry':
            updates['retries'] = mission.retries + 1
            updates['assigned_to'] = None
            updates['capability_status'] = 'ready'
            updates['capability_issues'] = ()
        entry = {'at': now, 'command': command, 'actor': actor, 'from': mission.state,
                 'to': target, 'reason': reason}
        if metadata:
            entry['metadata'] = dict(metadata)
        updated = replace(mission, history=mission.history + (entry,), **updates)
        self._repo.put(MISSION_NAMESPACE, mission_id, updated.to_dict(), expected=record.version)
        self._audit.append(actor, f'mission.{command}', mission_id,
                           {'from': mission.state, 'to': target, 'reason': reason,
                            **({'metadata': dict(metadata)} if metadata else {})})
        return {'mission': updated.to_dict(), 'applied': True}

    # -- reads -----------------------------------------------------------------
    def get(self, mission_id: str) -> Mission:
        return Mission.from_dict(self._repo.get(MISSION_NAMESPACE, mission_id).payload)

    def list_missions(self, *, state: str | None = None) -> list[Mission]:
        missions = [Mission.from_dict(r.payload) for r in self._repo.scan(MISSION_NAMESPACE)]
        if state is not None:
            missions = [m for m in missions if m.state == state]
        return sorted(missions, key=lambda m: (-m.priority, m.created_at, m.mission_id))

    # -- dispatch --------------------------------------------------------------
    def plan_claims(self, worker: str, capabilities: Iterable[Any], *,
                    actor: str | None = None, audit: bool = True) -> DegradationPlan:
        """Classify queued missions for one robot without assigning one.

        The plan keeps hard mismatches separate from optional capability gaps. A
        blocked mission remains queued for another robot; a degraded mission may be
        assigned, with every unavailable version/feature recorded as the reason.
        """
        offers: tuple[NegotiationOffer, ...] = tuple(
            offer_from_value(value) for value in capabilities)
        missions = self.list_missions(state='queued')
        tasks = [TaskSpec(mission.mission_id, capability_requirements_from_mission(mission))
                 for mission in missions]
        plan = plan_tasks(tasks, offers)
        if audit:
            actor_name = actor or worker
            for decision in plan.decisions:
                if decision.status == 'ready':
                    continue
                self._audit.append(actor_name, f'capability.{decision.status}', decision.task_id,
                                   {'worker': worker, 'decision': decision.to_dict()})
        return plan

    def claim_next(self, worker: str, capabilities: Iterable[Any], *,
                   allow_degraded: bool = True, audit: bool = True) -> Mission | None:
        """Hand the best runnable queued mission to ``worker``.

        Higher priority first, then oldest first. A mission whose hard requirements
        do not match is never assigned; its reason is audited. A mission with only
        optional gaps is assigned in degraded mode unless ``allow_degraded=False``.
        """
        plan = self.plan_claims(worker, capabilities, actor=worker, audit=audit)
        for mission in self.list_missions(state='queued'):
            try:
                decision = plan.get(mission.mission_id)
            except KeyError:
                continue
            if not decision.runnable:
                continue
            if decision.degraded and not allow_degraded:
                continue
            metadata = None
            if decision.degraded:
                metadata = {'capability_status': 'degraded',
                            'capability_issues': [issue.to_dict() for issue in decision.issues]}
            try:
                result = self.command(mission.mission_id, 'assign', actor=worker,
                                      assignee=worker, metadata=metadata)
            except VersionConflict:
                continue
            if result['applied']:
                return Mission.from_dict(result['mission'])
        return None

    def reorder(self, mission_id: str, priority: int, *, actor: str = 'operator') -> Mission:
        record = self._repo.get(MISSION_NAMESPACE, mission_id)
        mission = Mission.from_dict(record.payload)
        if mission.state in TERMINAL_STATES:
            raise InvalidTransition(mission_id, 'reorder', mission.state)
        updated = replace(mission, priority=int(priority), revision=mission.revision + 1,
                          updated_at=self._clock())
        self._repo.put(MISSION_NAMESPACE, mission_id, updated.to_dict(), expected=record.version)
        self._audit.append(actor, 'mission.reorder', mission_id,
                           {'priority': int(priority), 'state': mission.state})
        return updated

    def require(self, mission_id: str) -> Mission:
        try:
            return self.get(mission_id)
        except NotFound:
            raise MissionError(f'unknown mission {mission_id!r}') from None
