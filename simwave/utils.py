import inspect
import logging
import functools
from functools import wraps

def verbosity_control(level=logging.WARNING):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.basicConfig(level=level)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def set_verbosity(level=logging.WARNING):
    logging.basicConfig(level=level)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with logging.root.manager.disable_existing_loggers():
                return func(*args, **kwargs)
        return wrapper

    # Get all functions in the current module and imported modules
    module_list = [__import__(__name__)]
    functions = []
    for module in module_list:
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                for _, method in inspect.getmembers(obj, predicate=inspect.ismethod):
                    if hasattr(method, 'verbosity_control'):
                        method = method.verbosity_control(level=level)(method)
            elif callable(obj) and hasattr(obj, '__wrapped__'):
                if hasattr(obj, 'verbosity_control'):
                    obj = obj.verbosity_control(level=level)(obj)
                functions.append(obj)

    # Apply the decorator to all decorated functions
    for function in functions:
        if hasattr(function, 'verbosity_control'):
            function = function.verbosity_control(level=level)(function)


