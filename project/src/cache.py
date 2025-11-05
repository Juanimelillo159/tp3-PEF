"""
Utilidades de caché para la aplicación.

Este módulo proporciona un mecanismo de caché simple utilizando la biblioteca `diskcache`
para memoizar los resultados de llamadas a funciones costosas.
"""
from diskcache import Cache

cache = Cache("./cache")


def memoize(func):
    """
    Un decorador para memoizar llamadas a funciones usando diskcache.

    Args:
        func (function): La función a ser memoizada.

    Returns:
        function: La función envuelta.
    """

    def wrapper(*args, **kwargs):
        """
        La función envoltorio que implementa la lógica de caché.
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
