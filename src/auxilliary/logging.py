"""Tools for performance measurement"""

from time import perf_counter
import numpy as np
from collections import defaultdict, namedtuple
from contextlib import ContextDecorator

__all__ = ["PerformanceLog", "log_summary"]


class PerformanceLog(ContextDecorator):
    """Class for measuring the time of events"""

    data = defaultdict(list)

    def __init__(self, label):
        """Initialise object

        :arg label: name of timer
        """
        self.label = label

    def __enter__(self):
        """Enter context"""
        self.time = perf_counter()
        return self

    def __exit__(self, *exc):
        """Exit context"""
        t_elapsed = perf_counter() - self.time
        PerformanceLog.data[self.label].append(t_elapsed)


def log_summary():
    """Print summary of performance logging"""
    if len(PerformanceLog.data) == 0:
        return
    head_label = "timer"
    head_calls = "ncall"
    head_total = "total"
    head_avg = "avg"
    head_std = "std"
    Timing = namedtuple("Timing", ["label", "n_call", "total", "avg", "std"])
    print(
        f"{head_label:>32s} : {head_calls:>6s}    {head_total:>10s} {head_avg:>10s} {head_std:>10s}"
    )
    print(77 * "-")
    summary_data = []
    for label, timings in PerformanceLog.data.items():
        timings = np.asarray(timings)
        n_call = len(timings)
        t_total = np.sum(timings)
        t_avg = np.average(timings)
        t_std = np.std(timings)
        summary_data.append(Timing(label, n_call, t_total, t_avg, t_std))

    for t in sorted(summary_data, key=lambda x: x.total, reverse=True):
        print(
            f"{t.label:>32s} : {t.n_call:6d}    {t.total:10.4e} {t.avg:10.4e} {t.std:10.4e}"
        )
