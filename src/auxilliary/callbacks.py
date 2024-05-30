"""Classes for performing regular callbacks during ttimestepping"""

from abc import ABC, abstractmethod
from firedrake.output import VTKFile

__all__ = ["AnimationCallback"]


class Callback(ABC):
    """Abstract base class"""

    @abstractmethod
    def __call__(self, Q, p, t):
        """Invoke the callback for particular velocity/pressure fields at a given time

        :arg Q: velocity field at time t
        :arg p: pressure field at time t
        :arg t: time t
        """

    @abstractmethod
    def reset(self):
        """Reset callback"""


class AnimationCallback(Callback):
    """Save fields to disk"""

    def __init__(self, filename):
        """Initialise new instance

        :arg filename: name of file to write fields to
        """
        self.filename = filename
        self.reset()

    def reset(self):
        """Re-open file"""
        self.outfile = VTKFile(self.filename, mode="w")

    def __call__(self, Q, p, t):
        """Save velocity/pressure fields to disk at a given time

        :arg Q: velocity field at time t
        :arg p: pressure field at time t
        :arg t: time t
        """
        self.outfile.write(Q, p, time=t)
