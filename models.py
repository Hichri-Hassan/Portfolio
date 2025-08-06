"""Model training utilities.

For now we re-export `ActionableTradingPredictor` from the original implementation
in *testtt.py* so that the rest of the codebase can simply do
`from models import ActionableTradingPredictor`.

If you later decide to refactor the class into smaller units you only need to
update this single file – all other modules will continue to work unchanged.
"""

from testtt import ActionableTradingPredictor  # type: ignore[F401]

__all__ = ["ActionableTradingPredictor"]