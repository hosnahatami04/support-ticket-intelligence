from __future__ import annotations

import argparse
import statistics
import time

import requests

DEFAULT_URL = "http://localhost:8000/triage"
# A representative ticket; the benchmark measures routing only, never the
# summarizer, which is the KPI the spec sets a budget for (section 4.3).
PAYLOAD = {
    "subject": "Double charged for my subscription",
    "body": (
        "My monthly subscription was charged twice during this billing cycle. "
        "Please refund the duplicate charge and confirm by email."
    ),
    "summarize": False,
}


def run(url: str, requests_count: int, warmup: int) -> dict[str, float]:
    session = requests.Session()
    for _ in range(warmup):
        session.post(url, json=PAYLOAD, timeout=30)

    timings: list[float] = []
    for _ in range(requests_count):
        start = time.perf_counter()
        response = session.post(url, json=PAYLOAD, timeout=30)
        response.raise_for_status()
        timings.append((time.perf_counter() - start) * 1000)

    timings.sort()
    return {
        "requests": float(len(timings)),
        "p50_ms": statistics.median(timings),
        "p95_ms": timings[int(0.95 * (len(timings) - 1))],
        "mean_ms": statistics.fmean(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure POST /triage latency without summaries.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    results = run(args.url, args.requests, args.warmup)
    for key, value in results.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()
