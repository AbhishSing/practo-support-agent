"""Generate a deterministic sample of appointment records."""

import random
from collections import Counter

SEED = 42
NUM_APPOINTMENTS = 60

CATEGORY_WEIGHTS = {
    "General Medicine": 35,
    "Cardiology": 15,
    "Dermatology": 20,
    "Pediatrics": 15,
    "Orthopedics": 15,
}
STATUS_WEIGHTS = {
    "Scheduled": 30,
    "Completed": 35,
    "Cancelled": 15,
    "No-Show": 10,
    "Rescheduled": 10,
}
FOLLOW_UP_WEIGHT = 0.2
FEE_RANGES_BY_CATEGORY = {
    # These category-specific ranges model higher fees for specialist consultations while staying within INR 300-2,000.
    "General Medicine": (300, 800),
    "Cardiology": (800, 2000),
    "Dermatology": (500, 1500),
    "Pediatrics": (400, 1200),
    "Orthopedics": (700, 1800),
}

CATEGORIES = tuple(CATEGORY_WEIGHTS)
STATUSES = tuple(STATUS_WEIGHTS)


def generate_appointments() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    appointments = []

    for index in range(1, NUM_APPOINTMENTS + 1):
        category = rng.choices(
            CATEGORIES, weights=tuple(CATEGORY_WEIGHTS.values()), k=1
        )[0]
        status = rng.choices(
            STATUSES, weights=tuple(STATUS_WEIGHTS.values()), k=1
        )[0]
        fee_min, fee_max = FEE_RANGES_BY_CATEGORY[category]
        follow_up_required = rng.choices(
            (False, True), weights=(1 - FOLLOW_UP_WEIGHT, FOLLOW_UP_WEIGHT), k=1
        )[0]

        appointments.append(
            {
                "record_id": f"APT-{index:04d}",
                "category": category,
                "status": status,
                "consultation_fee_inr": rng.randint(fee_min, fee_max),
                "days_since_created": rng.randint(0, 30),
                "follow_up_required": follow_up_required,
            }
        )

    category_counts = Counter(appointment["category"] for appointment in appointments)
    status_counts = Counter(appointment["status"] for appointment in appointments)
    assert all(category_counts[category] >= 3 for category in CATEGORIES)
    assert all(status_counts[status] >= 1 for status in STATUSES)
    return appointments


APPOINTMENTS = generate_appointments()


def print_summary() -> None:
    category_counts = Counter(appointment["category"] for appointment in APPOINTMENTS)
    status_counts = Counter(appointment["status"] for appointment in APPOINTMENTS)
    follow_up_count = sum(
        appointment["follow_up_required"] for appointment in APPOINTMENTS
    )
    follow_up_percentage = follow_up_count / len(APPOINTMENTS) * 100

    print("Count per category:")
    for category in CATEGORIES:
        print(f"  {category}: {category_counts[category]}")

    print("Count per status:")
    for status in STATUSES:
        print(f"  {status}: {status_counts[status]}")

    print(f"Follow-up required: {follow_up_percentage:.1f}%")
    if not 10 <= follow_up_percentage <= 30:
        print(
            "Warning: follow-up percentage is outside 10%-30%; "
            "change SEED or FOLLOW_UP_WEIGHT."
        )


if __name__ == "__main__":
    print_summary()
