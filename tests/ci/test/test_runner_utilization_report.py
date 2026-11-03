"""Hosted CPU jobs must not create GPU-fleet utilization categories."""

from datetime import datetime, timezone

import pytest

from scripts.ci.runner_utilization_report import classify_job
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["mostik-linux-32core"], []),
        (["mostik-linux-arm64-32core"], []),
        (["self-hosted", "Linux", "X64", "h200", "4gpu"], ["h200", "4gpu"]),
    ],
)
def test_classify_job_excludes_hosted_labels_but_retains_gpu_labels(labels, expected):
    info = classify_job(
        {
            "status": "queued",
            "created_at": "2026-11-03T00:00:00Z",
            "labels": labels,
        },
        datetime(2026, 11, 3, 0, 1, tzinfo=timezone.utc),
    )

    assert info is not None
    assert info["labels"] == expected
    assert info["queue_time"] == 60
