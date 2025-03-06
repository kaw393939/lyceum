# Goliath Platform Docstring Standards

This document defines the standard docstring format for the Goliath Educational Platform.

## Module Docstrings

```python
"""
Module description.

This module provides functionality for...

Examples:
    Basic usage example:

    >>> from module import function
    >>> result = function(param)
"""
```

## Function Docstrings

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """Short description of function purpose.

    Detailed description of function behavior, when needed.

    Args:
        param1: Description of param1
        param2: Description of param2, default is {default}

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised

    Examples:
        >>> function_name("example", 123)
        "example result"
    """
    # function implementation
```

## Class Docstrings

```python
class ClassName:
    """Short description of class purpose.

    Detailed description of class behavior, when needed.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Examples:
        >>> obj = ClassName(param)
        >>> obj.method()
        "result"
    """

    def __init__(self, param1: type, param2: type = default) -> None:
        """Initialize ClassName.

        Args:
            param1: Description of param1
            param2: Description of param2, default is {default}
        """
        # implementation

    def method_name(self, param1: type) -> return_type:
        """Short description of method purpose.

        Args:
            param1: Description of param1

        Returns:
            Description of return value
        """
        # implementation
```

## Tips for Writing Good Docstrings

1. Be concise but informative
2. Include type hints for all parameters and return values
3. Document exceptions that might be raised
4. Use examples for non-trivial functions
5. For complex parameters (e.g., dicts), specify the expected structure
6. Document any side effects

## Integration with Documentation Tools

These docstrings can be parsed by Sphinx to generate HTML documentation:

```bash
# Generate documentation
cd docs
sphinx-apidoc -o source/ ../service_name
make html
```