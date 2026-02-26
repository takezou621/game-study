"""Trigger policies for cooldown and priority management."""

from domain.triggers.policies.cooldown_policy import CooldownPolicy
from domain.triggers.policies.priority_policy import PriorityPolicy

__all__ = [
    "CooldownPolicy",
    "PriorityPolicy",
]
