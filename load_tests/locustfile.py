"""Locust load testing scenario for the LLM inference API.

Run with:
    locust -f load_tests/locustfile.py --host http://localhost:8000

This simulates realistic user traffic patterns to benchmark throughput
and latency under concurrent load.
"""

from __future__ import annotations

from locust import HttpUser, between, task


class InferenceUser(HttpUser):
    """Simulated user sending generation requests to the API."""

    wait_time = between(1, 3)

    @task(weight=3)
    def generate_short(self) -> None:
        """Send a short-form generation request."""
        self.client.post(
            "/generate",
            json={
                "prompt": "Explain the concept of transfer learning in one paragraph.",
                "max_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

    @task(weight=2)
    def generate_medium(self) -> None:
        """Send a medium-length generation request."""
        self.client.post(
            "/generate",
            json={
                "prompt": (
                    "Write a detailed comparison between supervised and "
                    "unsupervised learning, covering key algorithms, "
                    "use cases, and trade-offs."
                ),
                "max_tokens": 512,
                "temperature": 0.8,
                "top_p": 0.95,
            },
        )

    @task(weight=1)
    def generate_long(self) -> None:
        """Send a long-form generation request to stress-test KV cache."""
        self.client.post(
            "/generate",
            json={
                "prompt": (
                    "You are an expert systems architect. Provide a "
                    "comprehensive design document for a real-time "
                    "recommendation engine that handles 10 million daily "
                    "active users. Include data pipeline, model serving "
                    "strategy, caching layer, and monitoring."
                ),
                "max_tokens": 1024,
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
            },
        )

    @task(weight=5)
    def health_check(self) -> None:
        """Periodic health probe (simulates load balancer behavior)."""
        self.client.get("/health")
