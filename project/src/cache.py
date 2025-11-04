"""
Caching utilities for the application.

This module provides a simple caching mechanism using the `diskcache` library
to memoize the results of expensive function calls.
"""
from diskcache import Cache

cache = Cache("./cache")


def memoize(func):
    """
    A decorator to memoize function calls using diskcache.

    Args:
        func (function): The function to be memoized.

    Returns:
        function: The wrapped function.
    """

    def wrapper(*args, **kwargs):
        """
        The wrapper function that implements the caching logic.
        """
        # Create a cache key from the function name, args, and kwargs
        key = (func.__name__, args, frozenset(kwargs.items()))

        # Check if the result is already in the cache
        if key in cache:
            return cache[key]

        # If not, call the function and store the result in the cache
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper
