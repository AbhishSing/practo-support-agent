"""Print the data-derived appointment escalation threshold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tools import (
    DAYS_SINCE_CREATED_P80,
    ESCALATION_THRESHOLD,
)


def main() -> None:
    print(f"DAYS_SINCE_CREATED_P80 = {DAYS_SINCE_CREATED_P80:.1f}")
    print(f"ESCALATION_THRESHOLD = {ESCALATION_THRESHOLD:.4f}")
    print(
        "Formula: 0.6 * follow_up_required + "
        "0.4 * (days_since_created / 30)"
    )


if __name__ == "__main__":
    main()
