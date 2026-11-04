"""Fifteen-query evaluation set covering all clinic policy topics."""

TESTSET = [
    {
        "query": "How far in advance can I book an appointment, and what do I need?",
        "expected_topic": "appointment-booking",
    },
    {
        "query": "How late can I cancel or reschedule without a fee?",
        "expected_topic": "cancellation-rescheduling",
    },
    {
        "query": "What is the consultation fee for Cardiology?",
        "expected_topic": "consultation-fees",
    },
    {
        "query": "How do I submit a cashless insurance claim?",
        "expected_topic": "insurance-claims",
    },
    {
        "query": "How soon can I refill a chronic medication prescription?",
        "expected_topic": "prescription-refills",
    },
    {
        "query": "When will my blood test results be available?",
        "expected_topic": "lab-test-turnaround",
    },
    {
        "query": "Can I use telemedicine for a routine follow-up?",
        "expected_topic": "telemedicine",
    },
    {
        "query": "What should I do if I arrive with severe chest pain?",
        "expected_topic": "emergency-care",
    },
    {
        "query": "Who can see my medical records, and can I export them?",
        "expected_topic": "patient-data-privacy",
    },
    {
        "query": "Is there a discount if I return for a follow-up within a week?",
        "expected_topic": "follow-up-discounts",
    },
    {
        "query": "How do I request a second opinion from a specialist?",
        "expected_topic": "second-opinions",
    },
    {
        "query": "Who qualifies for a home visit and how early should I request one?",
        "expected_topic": "home-visits",
    },
    {
        "query": "Is visitor parking free on weekends?",
        "expected_topic": "out-of-scope",
    },
    {
        "query": "What are the clinic's opening hours on public holidays?",
        "expected_topic": "out-of-scope",
    },
    {
        "query": "Can you tell me the current status of appointment APT-9999?",
        "expected_topic": "out-of-scope",
    },
]

assert len(TESTSET) == 15
