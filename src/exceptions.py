# exceptions.py

class LPError(Exception):
    """Base class for Linear Programming errors."""
    pass

class UnboundedError(LPError):
    """Raised when the linear program is unbounded."""
    def __init__(self, message="The linear program is unbounded."):
        self.message = message
        super().__init__(self.message)

class InfeasibleError(LPError):
    """Raised when the linear program is infeasible."""
    def __init__(self, message="The linear program is infeasible."):
        self.message = message
        super().__init__(self.message)

class DegeneracyError(LPError):
    """Raised when the linear program encounters degeneracy."""
    def __init__(self, message="The linear program is degenerate."):
        self.message = message
        super().__init__(self.message)