from diskcache import Cache

cache = Cache('./cache')

def memoize(func):
    """A simple decorator to memoize function calls."""
    def wrapper(*args, **kwargs):
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
