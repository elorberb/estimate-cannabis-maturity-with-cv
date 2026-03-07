# Python Style Guide — Full Examples

Extended examples for the rules in [AGENTS.md](../AGENTS.md). This doc is for human reference; agents should follow AGENTS.md first.

---

## Import Patterns

All imports at the top of the file (after module docstring). Use absolute paths only.

**Correct: absolute imports, imports at top**
```python
import json
from myproject.backend.services import UserService

def get_user(user_id: str) -> User:
    service = UserService()
    return service.get_user(user_id)
```

**Wrong: relative imports**
```python
from ..services import UserService
from .utils import helper
```

**Wrong: imports inside functions/classes**
```python
def get_user(user_id: str) -> User:
    from myproject.services import UserService  # L
    ...
```

**Correct: empty `__init__.py`**
```python
# __init__.py — completely empty
```

**Wrong: re-exports in `__init__.py`**
```python
# __init__.py
from myproject.services.user_service import UserService
__all__ = ["UserService"]
```
Import directly from the module instead: `from myproject.services.user_service import UserService`.

---

## Documentation & Comments

No docstrings. Only comment non-obvious logic.

**Correct**
```python
def process_user_age(age: str) -> int:
    try:
        age_int = int(age)
    except ValueError as e:
        raise ValueError(f"Age must be a valid integer, got: {age}") from e
    if age_int < 0:
        raise ValueError(f"Age must be positive, got: {age_int}")
    return age_int

# Only comment when necessary
def backoff(attempt: int) -> float:
    # Cap at 60s to avoid excessive delays
    return min(2 ** attempt, 60)
```

**Wrong: docstrings and redundant comments**
```python
def process_user_age(age: str) -> int:
    """Process and validate user age..."""  # no docstrings
    # Validate email  # redundant
    if "@" not in email: ...
```

---

## Exception Handling

Use specific exceptions; re-raise or let bubble. Catch only at top level or when you have a clear recovery.

**Correct: specific exception, re-raise with chain**
```python
try:
    age_int = int(age)
except ValueError as e:
    raise ValueError(f"Age must be a valid integer, got: {age}") from e
```

**Correct: catch at top level or for recovery**
```python
def get_cached_or_fetch(key: str) -> dict:
    try:
        return cache.get(key)
    except CacheExpiredError:
        return fetch_from_database(key)
```

**Correct: contextlib.suppress**
```python
from contextlib import suppress
with suppress(FileNotFoundError):
    os.remove(temp_file)
```

**Wrong: catch and re-raise**
```python
except Exception as e:
    logger.error(...)
    raise  # let it propagate without wrapping
```

**Wrong: assert for validation**
```python
assert user_id, "user_id required"  # use: raise ValueError(...)
```

---

## Properties & Private Members

All class attributes private; expose via `@property`. Public methods before private.

**Correct**
```python
class User:
    def __init__(self, name: str, age: int) -> None:
        self._name = name
        self._age = age

    @property
    def name(self) -> str:
        return self._name

    @property
    def age(self) -> int:
        return self._age
```

**Correct: read-only and cached_property**
```python
from functools import cached_property

class DataProcessor:
    @cached_property
    def sum(self) -> int:
        return sum(self._data)
```

**Wrong: public attributes**
```python
class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name  # use _name + @property
```

**Correct: method order**
```python
class UserService:
    def get_user(self, user_id: str) -> User: ...
    def create_user(self, name: str, email: str) -> User: ...
    def _validate_email(self, email: str) -> str: ...
```

---

## Class Methods & Static Methods

`@classmethod` only for factories that return instances. `@staticmethod` only in all-static utility classes. No module-level functions.

**Correct: classmethod factories**
```python
class User:
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(name=data["name"], email=data["email"])
```

**Correct: utility class, all static**
```python
import re

class StringUtils:
    @staticmethod
    def to_snake_case(text: str) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
```

**Wrong: module-level functions**
```python
def get_user(user_id: str) -> User: ...
def validate_email(email: str) -> bool: ...
```
Use a class with methods instead.

**Wrong: mixing static and instance**
```python
class UserService:
    def get_user(self, user_id: str) -> User: ...
    @staticmethod
    def validate_email(email: str) -> bool: ...  # use instance method
```

---

## Type Annotations

Annotate functions, methods, and non-obvious variables. Prefer specific types over `Any`.

**Correct**
```python
def get_user(self, user_id: str) -> User:
    ...
def list_users(self, limit: int = 100) -> list[User]:
    ...
def serialize_user(user: User) -> dict[str, str | int | bool]:
    return {"name": user.name, "age": user.age}
```

**Wrong: missing annotations or overuse of Any**
```python
def get_user(self, user_id): ...
def serialize_user(user: User) -> dict[str, Any]: ...
```

---

## Suppressing Warnings

Prefer fixing. If suppressing: inline with a short reason. Per-file ignores in config only for tests when justified.

**Correct: inline**
```python
result = some_function()  # noqa: ANN201 - temporary until refactor
result = some_function()  # type: ignore[return-value] - known safe cast
```

**Correct: tests only in pyproject.toml**
```toml
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["SLF001"]
```

**Wrong: global ignores for main code**
```toml
[tool.ruff.lint]
ignore = ["ANN001", "S101"]
```

---

## Default Arguments

No mutable defaults. Use `None` and assign inside the function.

**Correct**
```python
def foo(a: str, b: list[str] | None = None) -> None:
    if b is None:
        b = []
```

**Wrong**
```python
def foo(a: str, b: list[str] = []): ...
def foo(a: str, b: dict = {}): ...
```

---

## Comprehensions

Keep comprehensions simple. For multiple loops or complex conditions, use explicit loops.

**Correct**
```python
result = [f(x) for x in items if cond(x)]
return {k: transform(k) for k in items if k is not None}
```

**Wrong: nested for in one comprehension**
```python
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
# Prefer:
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))
```

---

## Async

This repo does not use a gRPC/PGVector-style async stack. If you add async code, use `async`/`await` consistently and avoid blocking I/O inside async functions; use async APIs or run blocking work in a thread/process when appropriate.

---

## Testing

Tests as functions (no test classes). Pytest + unittest.mock. Descriptive names; mirror source layout.

**Correct**
```python
def test_user_service_creates_user():
    service = UserService(repository=MockRepository())
    user = service.create_user("Alice", "alice@example.com")
    assert user.name == "Alice"

def test_user_service_validates_email():
    service = UserService(repository=MockRepository())
    with pytest.raises(ValueError, match="Invalid email"):
        service.create_user("Bob", "invalid-email")
```

**Correct: fixtures and mocking**
```python
@pytest.fixture
def user_service(user_repository):
    return UserService(repository=user_repository)

def test_create_user(user_service):
    user = user_service.create_user("Alice", "alice@example.com")
    assert user.name == "Alice"
```

**Wrong: test classes**
```python
class TestUserService:
    def test_creates_user(self): ...
```
