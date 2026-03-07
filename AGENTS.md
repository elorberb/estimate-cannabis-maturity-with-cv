# Python Development Guide

Python coding guidelines for this repo (Google Python Style Guide–based). Full examples: [app/docs/python-style-guide.md](app/docs/python-style-guide.md).

## Rules

**Do**
- Imports at top of file (after module docstring); never inside functions/classes
- Absolute imports only (e.g. `from myproject.services import UserService`); no relative imports
- Type all functions, methods, and non-obvious variables (not class members)
- Private class members (`_name`), expose via `@property`
- Public methods before private methods in classes
- Specific exceptions (`ValueError`, `TypeError`, custom); never bare `Exception`
- Catch only at top level (API/CLI) or with specific recovery; otherwise let exceptions bubble
- Put all logic in classes (use `@staticmethod` for utilities); no module-level functions
- Use `@classmethod` only for factory methods that return instances
- Fix lint/type issues instead of suppressing when possible
- Use `contextlib.suppress` instead of `try`/`pass` to ignore exceptions
- Empty `__init__.py` (no imports, no code)
- No docstrings; no redundant comments (only non-obvious logic)
- Run `ruff check --fix .` and `pyright` / `ty check` before commit

**Don't**
- Imports inside functions/classes (except circular-import workarounds)
- Relative imports (`from ..x`, `from .x`) or wildcard imports
- Module-level functions; public class members (`self.name` → use `self._name` + `@property`)
- Catching just to log and re-raise; wrapping everything in try/except
- Mixing `@staticmethod` with instance methods in the same class
- `raise Exception(...)`; using `assert` for input validation
- Mutable defaults (`def f(x=[])`); use `x=None` then `if x is None: x = []`
- Named return types in hints; global lint/type ignores in config (use inline; tests/* excepted)
- Multiple `for` clauses in comprehensions; use explicit loops for complex cases

## Quick commands

```bash
ruff check .
ruff check --fix .
ty check
pyright
pytest
pytest -v
pytest tests/test_foo.py
pytest -k test_name
pytest --cov=. --cov-report=html
uv sync
uv add package-name
uv add --dev package-name
```

## Config (reference)

- **Ruff**: line length 120; extend from root `ruff.toml` in services via `pyproject.toml`.
- **Types**: `ty` (gradual) and `pyright`; both must pass. Set `extraPaths` / `extra-paths` in `pyproject.toml` if needed.

## Short examples

**Imports and `__init__.py`**
```python
# Top of file only
from src.pipelines.end_to_end import EndToEndPipe

# __init__.py stays empty; import from the real module:
# from src.pipelines.end_to_end.end_to_end_pipe import EndToEndPipe
```

**Private members and properties**
```python
class TrichomeDetector:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path
```

**Exceptions and mutable defaults**
```python
if not path:
    raise ValueError("path cannot be empty")

def process(items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    return items
```

**Tests**: functions only (no test classes), pytest + `unittest.mock`, descriptive names, mirror source layout in `tests/`.
