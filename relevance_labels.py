"""Hand-labelled relevant KB document IDs for retrieval evaluation."""

RELEVANCE_LABELS = {
    "How much is a cardiology consultation?": {
        "consultation-fee structure by specialty.md"
    },
    "Can I cancel or move my appointment to another day?": {
        "cancellation_rescheduling window.md",
        "appointment-booking policy.md",
    },
    "How long do lab results take to come back?": {
        "lab-test turnaround times.md"
    },
    "Do I get a discount on a visit after a recent consultation?": {
        "follow-up-visit discount policy.md",
        "consultation-fee structure by specialty.md",
    },
    "Can a doctor come to my house for a checkup?": {
        "home-visit eligibility.md",
        "telemedicine eligibility.md",
    },
}
