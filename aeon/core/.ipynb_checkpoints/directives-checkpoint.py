"""
Core Directives and Configuration for the Aeon Agent.
Re-exports from the central prompts module for backward compatibility.
"""

from .prompts import CORE_DIRECTIVES, DOCKER_DIRECTIVES, IMPORTANT_REMINDERS

__all__ = ['CORE_DIRECTIVES', 'DOCKER_DIRECTIVES', 'IMPORTANT_REMINDERS']
