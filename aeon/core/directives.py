"""
Core Directives and Configuration for the Aeon Agent.
This module re-exports directives from the central prompts module for backward compatibility.
"""

from .prompts import CORE_DIRECTIVES, DOCKER_DIRECTIVES

__all__ = ['CORE_DIRECTIVES', 'DOCKER_DIRECTIVES']
