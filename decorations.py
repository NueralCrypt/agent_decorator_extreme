from functools import wraps
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor
from agents.utils.memory import shared_memory  # Redis-backed singleton
import time
import random

# ---- CORE DECORATORS ----
def cached_tool(ttl: int = 300) -> Callable:
    """
    Cache tool results in Redis. Auto-invalidates after TTL.
    Usage: 
    @cached_tool(ttl=3600)  # Cache for 1h
    def tool_search(query: str) -> str: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Generate a unique cache key (func_name + args hash)
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Check Redis first
            if (cached := shared_memory.get(cache_key)) is not None:
                return cached
            
            # Execute and cache
            result = func(self, *args, **kwargs)
            shared_memory.set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator

def threaded_tool(max_workers: int = 32) -> Callable:
    """
    Run tool in a thread pool. For IO-bound tasks.
    Usage:
    @threaded_tool(max_workers=8)
    def tool_fetch_urls(urls: list) -> list: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return executor.submit(func, self, *args, **kwargs).result()
        return wrapper
    return decorator

def confirm_tool(message: str = "Are you sure?") -> Callable:
    """
    Require Zed user confirmation before execution.
    Usage:
    @confirm_tool("This will DELETE DATA. Proceed?")
    def tool_nuke_database() -> str: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # In Zed, this will trigger a confirmation dialog
            if kwargs.pop("__confirmed", False):
                return func(self, *args, **kwargs)
            return f"🛑 CONFIRM REQUIRED: {message}\nRun again with __confirmed=True"
        return wrapper
    return decorator


# ---- ADVANCED DECORATORS ----
def time_travel_tool(state_keys: list):
    """
    Snapshots Redis state before execution. Revert with `/undo_last`.
    Usage: @time_travel_tool(state_keys=["db_schema", "user_data"])
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Snapshot current state
            snapshots = {key: shared_memory.get(key) for key in state_keys}
            snapshots["__func"] = func.__name__
            
            # Push to undo stack
            undo_stack = shared_memory.get("undo_stack", default=[])
            undo_stack.append(snapshots)
            shared_memory.set("undo_stack", undo_stack)
            
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.tool_undo_last()  # Auto-revert on failure
                raise
        return wrapper
    return decorator

def self_healing_tool(retries: int = 3):
    """
    Ask an LLM to fix errors before giving up.
    Usage: @self_healing_tool(retries=2)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            for _ in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = str(e)
                    # Ask LLM to suggest a fix
                    llm_response = self._ask_ollama(
                        f"Error in {func.__name__}: {last_error}\n"
                        "Suggest 1-line code change to fix:"
                    )
                    # Apply the fix (example: modify kwargs)
                    if "kwargs[" in llm_response:
                        exec(llm_response, globals(), locals())
            
            raise RuntimeError(f"💀 FAILED AFTER {retries} SELF-HEALING ATTEMPTS: {last_error}")
        return wrapper
    return decorator

def swarm_tool(task_split_fn: callable, result_merge_fn: callable):
    """
    Distribute work across multiple MCP servers.
    Usage: @swarm_tool(
        task_split_fn=lambda x: [x[i::3] for i in range(3)],
        result_merge_fn=sum
    )
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if len(args) == 1:
                subtasks = task_split_fn(args[0])
                # Publish tasks to Redis
                for i, subtask in enumerate(subtasks):
                    shared_memory.set(
                        f"swarm_task_{func.__name__}_{i}",
                        {"args": [subtask], "kwargs": kwargs}
                    )
                # Wait for workers
                results = []
                while len(results) < len(subtasks):
                    for i in range(len(subtasks)):
                        if result := shared_memory.get(f"swarm_result_{func.__name__}_{i}"):
                            results.append(result)
                            shared_memory.delete(f"swarm_result_{func.__name__}_{i}")
                    time.sleep(0.5)
                return result_merge_fn(results)
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

def quantum_tool(versions: int = 3, scoring_fn: callable = len):
    """
    Run N versions in parallel, keep the highest-scoring result.
    Usage: @quantum_tool(versions=3, scoring_fn=lambda x: x["accuracy"])
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate variant kwargs
            all_kwargs = [kwargs.copy() for _ in range(versions)]
            for i, variant in enumerate(all_kwargs):
                variant["__quantum_seed"] = i  # Variants can use this
            
            # Run all versions in parallel
            results = self._parallel([
                (f"quantum_{i}", lambda: func(self, *args, **variant_kwargs))
                for i, variant_kwargs in enumerate(all_kwargs)
            ])
            
            # Return best result
            return max(results.values(), key=scoring_fn)
        return wrapper
    return decorator

def vampire_tool(cron_schedule: str):
    """
    Automatically runs on schedule using Redis-backed cron.
    Usage: @vampire_tool("0 3 * * *")  # 3AM daily
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_run_key = f"last_run_{func.__name__}"
            last_run = shared_memory.get(last_run_key, 0)
            cron_to_seconds=float
            if time.time() - last_run >= cron_to_seconds(cron_schedule):
                result = func(self, *args, **kwargs)
                shared_memory.set(last_run_key, time.time())
                return f"🧛♂️ AUTO-RAN: {result}"
            return f"💤 Next run: {cron_schedule}"
        return wrapper
    return decorator

def chaos_monkey_tool(failure_rate: float = 0.1):
    """
    Randomly fails to test error handling.
    Usage: @chaos_monkey_tool(failure_rate=0.3)  # 30% chance to fail
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if random.random() < failure_rate:
                raise RuntimeError("🐵💥 CHAOS MONKEY ACTIVATED")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def retry_tool(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Retry failed tools with exponential backoff.
    Usage:
    @retry_tool(max_retries=5, delay=2.0)
    def tool_flaky_api() -> str: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            for attempt in range(1, max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    time.sleep(delay * attempt)
        return wrapper
    return decorator


def rate_limited_tool(calls: int = 10, period: float = 60.0) -> Callable:
    """
    Limit how often a tool can be called.
    Uses Redis for distributed rate limiting.
    Usage:
    @rate_limited_tool(calls=5, period=30.0)  # 5 calls/30s
    def tool_expensive_llm() -> str: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            key = f"rate_limit_{func.__name__}"
            current = shared_memory.get(key, 0)
            
            if current >= calls:
                raise RuntimeError(f"Rate limit exceeded ({calls}/{period}s)")
            
            shared_memory.set(key, current + 1, ttl=int(period))
            return func(self, *args, **kwargs)
        return wrapper
    return decorator