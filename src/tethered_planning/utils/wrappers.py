import cProfile
import functools
import io
import os
import pstats
import time
import tracemalloc


# Decorator to measure time and memory usage of a function
def measureStatsBase(f: callable) -> callable:
    """
    Custom decorator to measure time and memory usage of a function.

    Args:
        f (callable): The function to measure.

    Returns:
        callable: A wrapper function that returns the original function's result, and
            prints the execution time and peak memory usage.

    Usage:
        @measureStatsBase
        def my_function():
            ...

        result, time, mem = my_function()
    """

    @functools.wraps(f)
    def wrapper(*params) -> tuple:

        # Start memory and time measurement
        tracemalloc.start()
        start = time.process_time()

        # Call function
        result = f(*params)

        # Evaluate time and memory usage
        end = time.process_time()
        tot_time = end - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Print stats
        print(f"Stats for function: {f.__name__}")
        print(f"Execution time: {tot_time:.6f} seconds")
        print(f"Peak memory usage: {peak / 10**6:.6f} MB")
        print("\n")

        # Return results
        return result

    return wrapper


def measureStats(f: callable) -> callable:
    """
    Custom decorator to measure time and memory usage of a function.

    Args:
        f (callable): The function to measure.

    Returns:
        callable: A wrapper function that returns the original function's result, and
            prints the execution time and peak memory usage.

    Usage:
        @measureStats
        def my_function():
            ...
    """
    # Configuration settings
    n_rows = 30
    save_stats = True

    @functools.wraps(f)
    def wrapper(*params) -> tuple:

        # Start memory and time measurement
        pr = cProfile.Profile()
        pr.enable()

        # Call function
        result = f(*params)

        # Evaluate stats
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")

        # Print stats
        print(f"Stats for function: {f.__name__}")
        ps.print_stats(n_rows)  # top 30 lines
        print(s.getvalue())
        print("\n")

        if save_stats:
            print(f"Current folder: {os.getcwd()}")
            print(f"Saving to: results/stats_{f.__name__}.prof")
            pr.dump_stats(f"results/stats_{f.__name__}.prof")  # visualize with snakeviz

        # Return results
        return result

    return wrapper
