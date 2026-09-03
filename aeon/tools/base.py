"""Abstract base class and executable contract for Aeon tools."""

from abc import ABC, abstractmethod
import inspect
import types
from typing import Any, Dict, List, Union, get_args, get_origin, get_type_hints

from aeon.core.agent_protocol import ToolPolicy, infer_tool_policy
from aeon.core.tool_resources import (
    ToolResourceError,
    ToolResourcePolicy,
    tool_resource_policy,
)


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    # ANSI Color Codes for standardized output across tools
    C_RED = '\033[91m'
    C_YELLOW = '\033[93m'
    C_CYAN = '\033[96m'
    C_GREEN = '\033[92m'
    C_BLUE = '\033[94m'
    C_RESET = '\033[0m'

    def __init__(
        self,
        name: str,
        description: str,
        underlying_model: str = None,
        directives: list = None,
        policy: ToolPolicy = None,
        resource_policy: ToolResourcePolicy | None = None,
    ):
        self.name = name
        self.description = str(description or "")
        self.underlying_model = underlying_model
        self.policy = policy or infer_tool_policy(name)
        # Compute routing is a separate, explicit contract from side effects.
        # Unknown tools fail at the executable registration boundary instead of
        # silently inheriting host or GPU access. Construction remains available
        # for schema-only and isolated unit-test fixtures.
        if resource_policy is not None:
            self.resource_policy: ToolResourcePolicy | None = resource_policy
        else:
            try:
                self.resource_policy = tool_resource_policy(name)
            except ToolResourceError:
                # Construction remains useful for schema/unit-test fixtures.
                # Worker.register_tools is the executable fail-closed boundary.
                self.resource_policy = None
        
        # Load directives from manager if not explicitly provided
        if directives is None:
            from aeon.core.prompts.manager import load_tool_prompt
            self.directives = load_tool_prompt(name)
        else:
            self.directives = directives

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the tool with the given arguments."""
        # pylint: disable=unnecessary-pass
        pass

    @staticmethod
    def _annotation_schema(annotation: Any) -> dict:
        """Translate ordinary Python annotations into grammar-safe JSON Schema."""

        if annotation in {inspect.Parameter.empty, Any, object, None}:
            return {}
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {Union, types.UnionType}:
            branches = [
                BaseTool._annotation_schema(arg)
                for arg in args
                if arg is not type(None)
            ]
            branches = [branch for branch in branches if branch]
            if not branches:
                return {}
            if len(branches) == 1:
                return branches[0]
            return {"anyOf": branches}
        if origin in {list, List, tuple, set, frozenset}:
            item = BaseTool._annotation_schema(args[0]) if args else {}
            schema = {"type": "array"}
            if item:
                schema["items"] = item
            return schema
        if origin in {dict, Dict}:
            return {"type": "object", "additionalProperties": True}
        if annotation is str:
            return {"type": "string"}
        if annotation is bool:
            return {"type": "boolean"}
        if annotation is int:
            return {"type": "integer"}
        if annotation is float:
            return {"type": "number"}
        return {}

    def parameter_schema(self) -> dict:
        """Return the strict argument schema derived from ``execute``.

        A tool can override this method when its accepted values are narrower
        than its Python signature. Unknown/unannotated values remain permissive,
        but unknown *parameter names* are rejected unless the method explicitly
        accepts ``**kwargs``.
        """

        signature = inspect.signature(self.execute)
        try:
            annotations = get_type_hints(self.execute)
        except Exception:
            annotations = {}
        properties = {}
        required = []
        additional = False
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                additional = True
                continue
            annotation = annotations.get(name, parameter.annotation)
            properties[name] = self._annotation_schema(annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": additional,
        }
        if required:
            schema["required"] = required
        return schema

    def validate_parameters(self, parameters: Any) -> str:
        """Return a deterministic validation error, or an empty string.

        Structured decoding normally makes this a no-op. It remains an
        execution-side safety net when a server downgrades to legacy JSON mode or
        a tool call is restored from older session history.
        """

        if not isinstance(parameters, dict):
            return "parameters must be a JSON object"
        schema = self.parameter_schema()
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        missing = [name for name in required if name not in parameters]
        if missing:
            return "missing required parameter(s): " + ", ".join(sorted(missing))
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(parameters) - set(properties))
            if unknown:
                return "unknown parameter(s): " + ", ".join(unknown)

        def matches(value: Any, spec: dict) -> bool:
            if not spec:
                return True
            if "anyOf" in spec:
                return any(matches(value, branch) for branch in spec["anyOf"])
            expected = spec.get("type")
            if expected == "string":
                return isinstance(value, str)
            if expected == "boolean":
                return isinstance(value, bool)
            if expected == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if expected == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if expected == "array":
                if not isinstance(value, list):
                    return False
                item_spec = spec.get("items") or {}
                return all(matches(item, item_spec) for item in value)
            if expected == "object":
                return isinstance(value, dict)
            return True

        invalid = [
            name
            for name, value in parameters.items()
            if name in properties and not matches(value, properties[name])
        ]
        if invalid:
            return "wrong JSON type for parameter(s): " + ", ".join(sorted(invalid))
        return ""

    def format_error_message(
        self,
        error: Exception,
        context: str,
        resolution: str = 'retrying with adjusted parameters'
    ) -> str:
        """Format error into a yellow-colored explanatory message."""
        reason = str(error).splitlines()[0] if str(error) else 'Unknown'
        return (
            f'{self.C_YELLOW}ERROR: Encountered {type(error).__name__} while {context}. '
            f'Reason: {reason}. Resolving by {resolution}.{self.C_RESET}'
        )
