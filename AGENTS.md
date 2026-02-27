# Python Development Guide

General Python coding guidelines based on the Google Python Style Guide.

## ✅ Dos

- **Imports at Top**: ALL imports must be at the top of the file (after module docstring) - never inside functions/classes
- **Absolute Imports Only**: Use full package paths (e.g., `from myproject.services import UserService`) - NO relative imports
- **Type Everything**: All functions, methods, and non-obvious variables must have type annotations (not include class members)
- **Private Members**: ALL class members must be private (prefix with `_`), expose via `@property`
- **Method Ordering**: Public methods always come before private methods in class definitions
- **Specific Exceptions**: Use specific exception types (`ValueError`, `TypeError`) or custom exceptions - never bare `Exception`
- **Let Exceptions Bubble**: Only catch exceptions at top level (API handlers, CLI) or with specific recovery intent
- **No Module Functions**: ALL functions must be under classes - use `@staticmethod` for utility classes
- **@classmethod for Factories**: Use `@classmethod` ONLY for factory methods that return instances
- **Fix, Don't Suppress**: Try to fix linting/type issues rather than suppress warnings
- **Run Linting**: Run `ruff check --fix .` - must pass before commit
- **Run Type Checking**: Run `pyright` and `ty check` - must pass before commit
- **Contextlib.suppress**: Use `contextlib.suppress` instead of `try/pass` for ignoring exceptions
- **Empty __init__.py Files**: ALL `__init__.py` files must be empty - no imports, no code
- **No Docstrings**: Do NOT add docstrings to functions, methods, or classes - code should be self-documenting
- **No Redundant Comments**: Avoid comments that restate what the code does - only comment for non-obvious logic

## ❌ Don'ts

- **Imports Inside Functions/Classes**: Never place imports inside functions or classes (except for circular import workarounds)
- **Relative Imports**: Never use `from ..module import X` or `from .utils import X`
- **Wildcard Imports**: Never use `from module import *`
- **Module-Level Functions**: Functions must be in classes, not at module level
- **Public Class Members**: Don't use `self.name` - use `self._name` with `@property`
- **Wrap Everything in try/except**: Only catch exceptions where you can handle them meaningfully
- **Catch, Log, Re-raise**: Don't catch just to log and re-raise - let it bubble up
- **Mix Static and Instance**: If using `@staticmethod`, ALL methods in the class must be static
- **Bare Exception**: Never `raise Exception("message")` - use specific types
- **Use assert for Validation**: `assert` is for invariants only, not input validation
- **Mutable Default Arguments**: Never use `def foo(items=[])` - use `items=None` with `if items is None: items = []`
- **Named Return Values**: Avoid named returns in type hints - use explicit tuple types
- **Config Ignores**: Only suppress warnings via comments in code, not in `pyproject.toml` (except `tests/*`)
- **Docstrings**: Never add docstrings - they add noise and become outdated
- **Redundant Comments**: Don't comment obvious code like `# Create user` above `user = User()`


## Quick Commands

```bash
# Lint
ruff check .
ruff check --fix .               # Auto-fix

# Type check
ty check                          # Gradual type checker
pyright                           # Static type checker

# Run tests
pytest
pytest -v                         # Verbose
pytest tests/test_foo.py          # Specific file
pytest -k test_name               # Specific test

# Run with coverage
pytest --cov=. --cov-report=html

# Install dependencies (uv)
uv sync
uv add package-name               # Add new dependency
uv add --dev package-name         # Add dev dependency
```

## Linting & Type Checking Configuration

### Ruff Configuration

Example `ruff.toml`:
- Line length: 120 characters
- Enabled: pycodestyle, Pyflakes, flake8-bugbear, isort, type checking, and more
- Each service extends via `pyproject.toml`: `[tool.ruff] extend = "../ruff.toml"`

### Type Checking Configuration

**ty**: Gradual type checker focusing on return types and argument types
**Pyright**: Static type checker for structural validation

Both must pass. Configure in each service's `pyproject.toml`:

```toml
[tool.ty.environment]
extra-paths = ["../common", "../other_dependency"]

[tool.pyright]
extends = "../pyright.toml"
extraPaths = ["../common", "../other_dependency"]
```

## Import Patterns

**CRITICAL: All imports MUST be at the top of the file** - immediately after the module docstring (if present). Never place imports inside functions, methods, or class definitions unless absolutely necessary (e.g., circular import workarounds).

### ✅ Correct: Absolute Imports

```python
# Always use full package paths
from myproject.backend.services import UserService, OrderService
from myproject.backend.models import User, Order
from myproject.backend.utils import parse_date, format_currency

# Import modules
import os
import sys

# Import specific objects
from typing import Optional, List
```

### ❌ Wrong: Relative Imports

```python
# Never do this!
from ..services import UserService  # L
from .utils import helper  # L
from . import models  # L
```

### ❌ Wrong: Wildcard Imports

```python
# Never do this!
from myproject.services import *  # L
```

### ❌ Wrong: Imports Inside Functions/Classes

```python
# Never do this!
def get_user(user_id: str) -> User:
    from myproject.services import UserService  # L - Import should be at top
    service = UserService()
    return service.get_user(user_id)

class MyService:
    def process_data(self, data: str) -> dict:
        import json  # L - Import should be at top
        return json.loads(data)
```

### ✅ Correct: All Imports at Top

```python
# All imports at the top of the file
import json
from myproject.services import UserService

def get_user(user_id: str) -> User:
    service = UserService()
    return service.get_user(user_id)

class MyService:
    def process_data(self, data: str) -> dict:
        return json.loads(data)
```

### ✅ Correct: Empty __init__.py Files

```python
# __init__.py
# This file should be completely empty - no imports, no code
```

### ❌ Wrong: __init__.py with Re-exports

```python
# __init__.py
# ❌ Never do this!
from myproject.services.user_service import UserService
from myproject.services.order_service import OrderService

__all__ = ["UserService", "OrderService"]
```

Instead, import directly from the module:
```python
# ✅ Correct - import directly from the module
from myproject.services.user_service import UserService
from myproject.services.order_service import OrderService
```

## Documentation & Comments

### ✅ Correct: Self-Documenting Code, Minimal Comments

```python
# No docstrings - clear function names and types are sufficient
def process_user_age(age: str) -> int:
    try:
        age_int = int(age)
    except ValueError as e:
        raise ValueError(f"Age must be a valid integer, got: {age}") from e

    if age_int < 0:
        raise ValueError(f"Age must be positive, got: {age_int}")

    return age_int

def calculate_average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

# Only comment non-obvious logic
def calculate_exponential_backoff(attempt: int) -> float:
    # Cap backoff at 60 seconds to prevent excessive delays
    return min(2 ** attempt, 60)

# Comment complex business logic
def calculate_pricing_tier(revenue: float, employees: int) -> str:
    # Enterprise tier requires both high revenue AND large team
    # to prevent small companies from gaming the system
    if revenue > 1_000_000 and employees > 50:
        return "enterprise"
    return "standard"
```

### ❌ Wrong: Docstrings and Redundant Comments

```python
# ❌ Don't add docstrings - they add maintenance burden
def process_user_age(age: str) -> int:
    """Process and validate user age.

    Args:
        age: The age as a string to process.

    Returns:
        The validated age as an integer.

    Raises:
        ValueError: If age is not a valid positive integer.
    """
    try:
        age_int = int(age)
    except ValueError as e:
        raise ValueError(f"Age must be a valid integer, got: {age}") from e

    if age_int < 0:
        raise ValueError(f"Age must be positive, got: {age_int}")

    return age_int

# ❌ Don't add redundant comments
def create_user(name: str, email: str) -> User:
    # Validate email
    if "@" not in email:
        raise ValueError("Invalid email")

    # Create user object
    user = User(name=name, email=email)

    # Save to database
    repository.save(user)

    # Return the user
    return user

# ✅ Better: Let the code speak for itself
def create_user(name: str, email: str) -> User:
    if "@" not in email:
        raise ValueError("Invalid email")

    user = User(name=name, email=email)
    repository.save(user)
    return user
```

## Exception Handling Patterns

### ✅ Correct: Specific Exceptions, Let Bubble Up

```python
def process_user_age(age: str) -> int:
    try:
        age_int = int(age)
    except ValueError as e:
        raise ValueError(f"Age must be a valid integer, got: {age}") from e

    if age_int < 0:
        raise ValueError(f"Age must be positive, got: {age_int}")

    return age_int

def calculate_average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

def read_config_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
```

### ✅ Correct: Catch at Top Level Only

```python
# In a gRPC service handler (top level)
def GetUser(self, request, context):
    try:
        user = self.user_service.get_user(request.user_id)
        return user_pb2.User(...)
    except UserNotFoundError as e:
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(str(e))
        return user_pb2.User()
    except ValueError as e:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(str(e))
        return user_pb2.User()

# With specific recovery intent
def get_cached_or_fetch(key: str) -> dict:
    try:
        return cache.get(key)
    except CacheExpiredError:
        # Specific recovery: fetch fresh data
        return fetch_from_database(key)
```

### ✅ Correct: Use contextlib.suppress

```python
from contextlib import suppress

# Explicitly suppress specific exceptions
with suppress(FileNotFoundError):
    os.remove(temp_file)
```

### ❌ Wrong: Catch and Re-raise

```python
# Don't do this - just adds noise
def read_config_file(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read config: {e}")
        raise  # ❌ Pointless - just let it propagate!
```

### ❌ Wrong: Catch Everything

```python
# Don't wrap everything in try/except
def calculate_average(numbers: list[float]) -> float:
    try:
        return sum(numbers) / len(numbers)
    except Exception as e:  # ❌ Too broad, pointless wrapping
        raise
```

### ❌ Wrong: Use assert for Validation

```python
# Wrong - assert can be disabled with -O flag
def process_user(user_id: str) -> User:
    assert user_id, "user_id required"  # ❌ Use ValueError instead
    return fetch_user(user_id)

# Correct
def process_user(user_id: str) -> User:
    if not user_id:
        raise ValueError("user_id cannot be empty")
    return fetch_user(user_id)
```

## Properties & Private Members Pattern

### ✅ Correct: Private Members with Properties

```python
class User:
    def __init__(self, name: str, age: int) -> None:
        self._name = name
        self._age = age

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

user = User("Alice", 30)
print(user.name)
user.age = 31
```

### ✅ Correct: Read-Only Properties

```python
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def area(self) -> float:
        return 3.14159 * self._radius ** 2

    @property
    def diameter(self) -> float:
        return self._radius * 2
```

### ✅ Correct: Lazy Evaluation with cached_property

```python
from functools import cached_property

class DataProcessor:
    def __init__(self, data: list[int]) -> None:
        self._data = data

    @property
    def data(self) -> list[int]:
        return self._data

    @cached_property
    def sum(self) -> int:
        return sum(self._data)

    @cached_property
    def average(self) -> float:
        return sum(self._data) / len(self._data)
```

### ❌ Wrong: Public Class Members

```python
class User:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
```

## Method Ordering Pattern

### ✅ Correct: Public Methods Before Private Methods

```python
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        return self._repository.find_by_id(user_id)

    def create_user(self, name: str, email: str) -> User:
        validated_email = self._validate_email(email)
        return self._repository.create(name, validated_email)

    def _validate_email(self, email: str) -> str:
        if "@" not in email:
            raise ValueError("Invalid email format")
        return email.lower()

    def _send_welcome_email(self, user: User) -> None:
        pass
```

### ❌ Wrong: Mixed Public and Private Methods

```python
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        return self._repository.find_by_id(user_id)

    def _validate_email(self, email: str) -> str:
        if "@" not in email:
            raise ValueError("Invalid email format")
        return email.lower()

    def create_user(self, name: str, email: str) -> User:
        validated_email = self._validate_email(email)
        return self._repository.create(name, validated_email)
```

## Class Methods & Static Methods Patterns

### ✅ Correct: @classmethod for Factory Methods Only

```python
class User:
    def __init__(self, name: str, email: str, role: str) -> None:
        self._name = name
        self._email = email
        self._role = role

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            name=data["name"],
            email=data["email"],
            role=data.get("role", "user")
        )

    @classmethod
    def create_admin(cls, name: str, email: str) -> "User":
        return cls(name=name, email=email, role="admin")

user1 = User.from_dict({"name": "Alice", "email": "alice@example.com"})
user2 = User.create_admin("Bob", "bob@example.com")
```

### ✅ Correct: @staticmethod for Pure Utility Classes

```python
class StringUtils:
    @staticmethod
    def to_snake_case(text: str) -> str:
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()

    @staticmethod
    def to_camel_case(text: str) -> str:
        components = text.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    @staticmethod
    def truncate(text: str, max_length: int) -> str:
        return text[:max_length] + "..." if len(text) > max_length else text

snake = StringUtils.to_snake_case("HelloWorld")
camel = StringUtils.to_camel_case("hello_world")
```

### ✅ Correct: No Module-Level Functions - Use Classes

```python
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        return self._repository.find_by_id(user_id)

    def validate_email(self, email: str) -> bool:
        return "@" in email and "." in email.split("@")[1]
```

### ❌ Wrong: Module-Level Functions

```python
# user_service.py
# ❌ Never do this - use classes instead!
def get_user(user_id: str) -> User:
    repository = UserRepository()
    return repository.find_by_id(user_id)

def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]
```

### ❌ Wrong: Mix Static and Instance Methods

```python
# ❌ Don't mix staticmethod with instance methods
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        return self._repository.find_by_id(user_id)

    @staticmethod
    def validate_email(email: str) -> bool:
        return "@" in email

# ✅ Correct: Keep validate_email as instance method
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        return self._repository.find_by_id(user_id)

    def validate_email(self, email: str) -> bool:
        return "@" in email
```

## Type Annotation Patterns

### ✅ Correct: Fully Typed

```python
class UserService:
    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: str) -> User:
        user = self._repository.find_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")
        return user

    def list_users(self, limit: int = 100) -> list[User]:
        return self._repository.find_all(limit=limit)

users: list[User] = []
config: dict[str, Any] = load_config()
cache: dict[str, User | None] = {}

count = 0
name = "Alice"
is_active = True
```

### ✅ Correct: Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Repository(Generic[T]):
    def __init__(self, model_class: type[T]) -> None:
        self._model_class = model_class

    def find_by_id(self, id: str) -> T | None:
        pass

    def find_all(self) -> list[T]:
        pass

user_repo: Repository[User] = Repository(User)
user: User | None = user_repo.find_by_id("123")
```

### ✅ Correct: Use Specific Types, Not Any

```python
def serialize_user(user: User) -> dict[str, str | int | bool]:
    return {"name": user.name, "age": user.age, "active": user.is_active}
```

### ❌ Wrong: Missing Type Annotations

```python
# ❌ Missing type hints!
class UserService:
    def __init__(self, repository):
        self._repository = repository

    def get_user(self, user_id):
        user = self._repository.find_by_id(user_id)
        if user is None:
            raise UserNotFoundError(f"User {user_id} not found")
        return user
```

### ❌ Wrong: Overuse Any

```python
# ❌ Too vague - be more specific!
def serialize_user(user: User) -> dict[str, Any]:
    return {"name": user.name, "age": user.age, "active": user.is_active}
```

## Suppressing Warnings

### ✅ Correct: Inline Suppression with Explanation

```python
# Suppress a specific Ruff rule on the next line
result = eval(user_input)  # noqa: S307 - Safe in this context because...

# Suppress multiple rules
value = some_function()  # noqa: ARG001, ANN201 - Temporary until refactor

# ty suppression
result = some_function()  # ty: ignore - Complex type inference issue

# Pyright suppression with error code
result = some_function()  # type: ignore[return-value] - Known safe cast
```

### ✅ Correct: File-Level Suppression (When Necessary)

```python
# ruff: noqa: S311, S113
"""Module that needs to ignore specific security warnings."""

# ty: ignore
# Entire file has complex typing issues being resolved

# pyright: reportAttributeAccessIssue=false
```

### ✅ Correct: Per-File Ignores in pyproject.toml (Tests Only)

```toml
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["SLF001"]  # Allow accessing private members in tests
```

### ❌ Wrong: Global Ignores in pyproject.toml

```toml
# ❌ Don't add global ignores for non-test code
[tool.ruff.lint]
ignore = ["ANN001", "S101"]  # Wrong - use inline comments instead
```

## Default Argument Values

### ✅ Correct: Avoid Mutable Defaults

```python
def foo(a, b=None):
    if b is None:
        b = []

def foo(a, b: Sequence | None = None):
    if b is None:
        b = []

def foo(a, b: Sequence = ()):  # Empty tuple OK since tuples are immutable
    ...
```

### ❌ Wrong: Mutable Defaults

```python
def foo(a, b=[]):  # ❌ Never use mutable defaults
    ...

def foo(a, b=time.time()):  # ❌ Evaluated at module load time
    ...

def foo(a, b: Mapping = {}):  # ❌ Could still get passed to unchecked code
    ...
```

## Comprehensions

### ✅ Correct: Simple Comprehensions

```python
result = [mapping_expr for value in iterable if filter_expr]

result = [
    is_valid(metric={'key': value})
    for value in interesting_iterable
    if a_longer_filter_expression(value)
]

descriptive_name = [
    transform({'key': key, 'value': value}, color='black')
    for key, value in generate_iterable(some_input)
    if complicated_condition_is_met(key, value)
]

# Dict and set comprehensions
return {x: complicated_transform(x) for x in items if x is not None}
unique_names = {user.name for user in users if user is not None}
```

### ❌ Wrong: Multiple For Clauses

```python
# ❌ Too complex - use explicit loops
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]

# ✅ Use explicit loops instead
result = []
for x in range(10):
    for y in range(5):
        if x * y > 10:
            result.append((x, y))
```

## Async/Await & Blocking I/O Patterns

### Overview

Python services use **grpc.aio** which runs on a **single event loop**. Blocking I/O operations freeze the entire event loop, causing all concurrent requests to queue up and timeout. Always use async alternatives for I/O operations.

### ✅ Correct: Use Async Alternatives

```python
# ✅ Async gRPC calls
from common.grpc.async_client import AsyncGrpcClient

endpoints_client = AsyncGrpcClient(endpoint, EndpointsServiceStub).stub()
response = await endpoints_client.GetEndpoint(request)

# ✅ Async gRPC streaming
async for item in endpoints_client.StreamEndpoints(request):
    process(item)

# ✅ Async LLM requests
from common.llm_client.llm_client import LLMClient

llm_client = LLMClient()
response = await llm_client.async_llm_request(messages, options)

# ✅ Async database operations
from common.db.async_database_connection import AsyncDatabaseConnection
from common.pgvector.pgvector_handler import AsyncPGVectorHandler

db_connection = AsyncDatabaseConnection(config.database)
pg_vector_handler = AsyncPGVectorHandler(embedder=embedder, db_connection=db_connection)
await pg_vector_handler.initialize()

schema = await pg_vector_handler.retrieve_schema(datasource_id, tenant_name)
fields = await pg_vector_handler.retrieve_important_fields(datasource_id, tenant_name)

# ✅ Parallel async execution
results = await asyncio.gather(
    fetch_schema(datasource_id),
    fetch_fields(datasource_id),
    fetch_operators(datasource_id),
)
```

### ❌ Wrong: Blocking I/O in Async Context

```python
# ❌ Blocking gRPC calls
from common.grpc.client import GrpcClient

endpoints_client = GrpcClient(endpoint, EndpointsServiceStub).stub()
response = endpoints_client.GetEndpoint(request)  # BLOCKS EVENT LOOP!

# ❌ Blocking LLM requests
response = llm_client.llm_request(messages, options)  # BLOCKS EVENT LOOP!

# ❌ Blocking database operations
from common.pgvector.pgvector_handler import PGVectorHandler

pg_vector_handler = PGVectorHandler(embedder=embedder, db_connection=db_connection)
schema = pg_vector_handler.retrieve_schema(datasource_id, tenant_name)  # BLOCKS EVENT LOOP!

# ❌ ThreadPoolExecutor (use asyncio.gather instead)
with ThreadPoolExecutor() as executor:
    results = list(executor.map(fetch_data, items))
```

### ❌ Wrong: Wrapping Async-Capable Operations with asyncio.to_thread()

```python
# ❌ Don't wrap async-capable operations
schema = await asyncio.to_thread(
    pg_vector_handler.retrieve_schema,  # This has async version!
    datasource_id,
    tenant_name
)

# ✅ Use async version directly instead
schema = await async_pg_vector_handler.retrieve_schema(datasource_id, tenant_name)
```

### When to Use asyncio.to_thread()

Only use `asyncio.to_thread()` when:
1. No async alternative exists
2. The operation is CPU-bound (not I/O-bound)
3. You're calling a third-party library without async support

```python
import hashlib

async def compute_hash(data: bytes) -> str:
    return await asyncio.to_thread(hashlib.sha256, data)

import some_sync_library

async def process_data(data: str) -> dict:
    return await asyncio.to_thread(some_sync_library.process, data)
```

### Detecting Blocking Code

#### 1. Runtime Detection with @blocking_io Decorator

Use the `@BlockingIOUtils.blocking_io` decorator to catch blocking calls at runtime:

```python
from common.utils.blocking import BlockingIOUtils

class LLMClient:
    @BlockingIOUtils.blocking_io
    def llm_request_sync(self, messages: list) -> str:
        ...

async def my_handler():
    client = LLMClient()
    result = client.llm_request_sync(messages)
```

```python
# Pyright CANNOT catch this:
async def execute(self):
    result = self._verify(...)  # Legal to call sync from async

def _verify(self):  # Sync function
    response = self.llm_client.llm_request(...)  # Pyright doesn't know this blocks!

# Pyright CAN catch this:
async def execute(self):
    result = self._verify(...)  # ERROR: Missing await on coroutine!

async def _verify(self):  # Now async
    response = await self.llm_client.async_llm_request(...)
```

### Common Async Patterns

#### Initialize Async Resources on Startup

```python
from common.app.base_app import BaseApp

class MyApp(BaseApp):
    async def _setup_grpc_dependencies(self) -> None:
        # Initialize async handlers
        await self.pg_vector_handler.initialize()
        await self.normalized_fields_holder.start()
```

1. **ALWAYS use async versions**: `AsyncGrpcClient`, `async_llm_request()`, `AsyncPGVectorHandler`
2. **NEVER wrap async-capable operations** with `asyncio.to_thread()`
3. **Mark blocking functions** with `@BlockingIOUtils.blocking_io`
4. **Initialize async resources** on app startup (not lazily in handlers)
5. **Use asyncio.gather()** for parallel execution (not `ThreadPoolExecutor`)
6. **Only use asyncio.to_thread()** for CPU-bound operations or third-party sync libraries

## Testing Patterns

### ✅ Correct: Write Tests as Functions (Not Classes)

```python
def test_user_service_creates_user():
    service = UserService(repository=MockRepository())
    user = service.create_user("Alice", "alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"

def test_user_service_validates_email():
    service = UserService(repository=MockRepository())
    with pytest.raises(ValueError, match="Invalid email"):
        service.create_user("Bob", "invalid-email")
```

### ✅ Correct: Use unittest.mock for Mocking

```python
from unittest.mock import Mock, patch, MagicMock

def test_user_service_calls_repository():
    mock_repository = Mock()
    mock_repository.save.return_value = User(id="123", name="Alice")

    service = UserService(repository=mock_repository)
    user = service.create_user("Alice", "alice@example.com")

    mock_repository.save.assert_called_once()
    assert user.name == "Alice"

@patch('ai.services.external_api.requests.post')
def test_external_api_call(mock_post):
    mock_post.return_value.json.return_value = {"status": "success"}

    result = call_external_api()

    assert result["status"] == "success"
    mock_post.assert_called_once()
```

### ✅ Correct: Use Pytest Fixtures for Setup

```python
import pytest

@pytest.fixture
def user_repository():
    return MockUserRepository()

@pytest.fixture
def user_service(user_repository):
    return UserService(repository=user_repository)

def test_create_user(user_service):
    user = user_service.create_user("Alice", "alice@example.com")
    assert user.name == "Alice"

def test_validate_email(user_service):
    with pytest.raises(ValueError):
        user_service.create_user("Bob", "invalid")
```

### ❌ Wrong: Test Classes

```python
# ❌ Don't use test classes!
class TestUserService:
    def test_creates_user(self):
        service = UserService(repository=MockRepository())
        user = service.create_user("Alice", "alice@example.com")
        assert user.name == "Alice"
```

### Key Testing Rules

1. **Write tests as functions**, not classes - prefix with `test_`
2. **Use pytest** for test structure (not unittest)
3. **Use unittest.mock** for mocking (`Mock`, `patch`, `MagicMock`)
4. **Use pytest fixtures** for shared setup code
5. **Descriptive test names** - clearly describe what's being tested
6. **Mirror source structure** in tests directory


## Critical Notes

- **Imports at top**: ALL imports must be at the top of the file (after module docstring) - never inside functions/classes
- **Absolute imports only**: NO relative imports (`from ..module`) - use full package paths
- **Private members**: ALL class attributes must be `_private`, expose via `@property` decorators
- **No module functions**: ALL functions must be under classes - use `@staticmethod` for utilities
- **@classmethod factories**: Use `@classmethod` ONLY for factory methods returning instances
- **Type everything**: All functions, methods, and non-obvious variables must have type annotations (not include class members)
- **Let exceptions bubble**: Only catch at top level (handlers) or with specific recovery logic
- **Specific exceptions**: Use `ValueError`, `TypeError`, custom exceptions - never bare `Exception`
- **contextlib.suppress**: Use `contextlib.suppress` instead of `try/pass` for ignoring exceptions
- **No mutable defaults**: Never `def foo(items=[])` - use `items=None` with check
- **No docstrings**: NEVER add docstrings to functions, methods, or classes - code should be self-documenting
- **No redundant comments**: Only comment non-obvious logic - avoid restating what code does
- **Tests as functions**: Write tests as functions (not classes), use pytest + unittest.mock
- **Linting required**: `ruff check .` must pass with zero warnings before commit
- **Type checking required**: `pyright` + `ty check` must pass before commit
- **Suppress inline only**: Use inline comments for suppressions, not `pyproject.toml` (except tests)
- **Simple comprehensions**: Avoid multiple `for` clauses - use explicit loops for complex logic
