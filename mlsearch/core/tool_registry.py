from __future__ import annotations

import json
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, get_type_hints
from dataclasses import dataclass, asdict

@dataclass
class ToolDefinition:
    """Tool definition for OpenAI function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

class ToolRegistry:
    """Central registry for tools with automatic schema generation."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool with the registry.
        
        Parameters
        ----------
        name : str
            Tool name for function calling.
        function : Callable
            The actual function to call.
        description : str
            Description of what the tool does.
        parameters : Dict[str, Any], optional
            JSON schema for parameters. If None, will be auto-generated.
        """
        if parameters is None:
            parameters = self._generate_parameters_schema(function)
        
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        
        self._tools[name] = tool_def
        self.logger.info(f"🔧 Registered tool: {name}")
    
    def register_from_schema(
        self,
        schema: Dict[str, Any],
        function: Callable
    ) -> None:
        """
        Register a tool from an existing schema definition.
        
        Parameters
        ----------
        schema : Dict[str, Any]
            Tool schema in the format used by existing schema files.
        function : Callable
            The actual function to call.
        """
        tool_def = ToolDefinition(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
            function=function
        )
        
        self._tools[schema["name"]] = tool_def
        self.logger.info(f"🔧 Registered tool from schema: {schema['name']}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get the function for a tool by name."""
        tool = self._tools.get(name)
        return tool.function if tool else None
    
    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def get_tool_functions(self) -> Dict[str, Callable]:
        """Get all tool functions as a dictionary (for backward compatibility)."""
        return {name: tool.function for name, tool in self._tools.items()}
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given arguments.
        
        Parameters
        ----------
        name : str
            Name of the tool to execute.
        arguments : Dict[str, Any]
            Arguments to pass to the tool.
        
        Returns
        -------
        Any
            Result of the tool execution.
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        # Validate arguments against schema
        self._validate_arguments(tool, arguments)
        
        # Execute the function
        import asyncio
        if asyncio.iscoroutinefunction(tool.function):
            return await tool.function(**arguments)
        else:
            return tool.function(**arguments)
    
    def _generate_parameters_schema(self, function: Callable) -> Dict[str, Any]:
        """
        Auto-generate JSON schema from function signature.
        
        Parameters
        ----------
        function : Callable
            Function to generate schema for.
        
        Returns
        -------
        Dict[str, Any]
            JSON schema for the function parameters.
        """
        try:
            sig = inspect.signature(function)
            type_hints = get_type_hints(function)
            
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls']:
                    continue
                
                # Get type information
                param_type = type_hints.get(param_name, str)
                
                # Convert Python types to JSON schema types
                if param_type in [str, type(None)]:
                    schema_type = "string"
                elif param_type in [int]:
                    schema_type = "integer"
                elif param_type in [float]:
                    schema_type = "number"
                elif param_type in [bool]:
                    schema_type = "boolean"
                elif param_type in [list, List]:
                    schema_type = "array"
                elif param_type in [dict, Dict]:
                    schema_type = "object"
                else:
                    schema_type = "string"  # Default fallback
                
                properties[param_name] = {
                    "type": schema_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Check if parameter is required
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to auto-generate schema for {function.__name__}: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def _validate_arguments(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> None:
        """
        Validate arguments against tool schema.
        
        Parameters
        ----------
        tool : ToolDefinition
            Tool definition with schema.
        arguments : Dict[str, Any]
            Arguments to validate.
        """
        schema = tool.parameters
        required_params = schema.get("required", [])
        
        # Check required parameters
        for param in required_params:
            if param not in arguments:
                raise ValueError(f"Missing required parameter '{param}' for tool '{tool.name}'")
        
        # Basic type validation could be added here
        # For now, we'll rely on the function itself to handle validation
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self.logger.info("🧹 Cleared all tools from registry")

# Global registry instance
_global_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry

def register_tool(
    name: str,
    function: Callable,
    description: str,
    parameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a tool with the global registry.
    
    Parameters
    ----------
    name : str
        Tool name for function calling.
    function : Callable
        The actual function to call.
    description : str
        Description of what the tool does.
    parameters : Dict[str, Any], optional
        JSON schema for parameters. If None, will be auto-generated.
    """
    _global_registry.register_tool(name, function, description, parameters)

def register_from_schema(schema: Dict[str, Any], function: Callable) -> None:
    """
    Register a tool from an existing schema definition.
    
    Parameters
    ----------
    schema : Dict[str, Any]
        Tool schema in the format used by existing schema files.
    function : Callable
        The actual function to call.
    """
    _global_registry.register_from_schema(schema, function)