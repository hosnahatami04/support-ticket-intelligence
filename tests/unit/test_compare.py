from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.data

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from ticketintel.evaluation import compare  # noqa: E402


def test_render_comparison_has_header_and_one_row_per_entry() -> None:
    rows = [
        {
            "task": "queue",
            "model": "tfidf",
            "macro_f1": 0.5,
            "train_seconds": 12.0,
            "latency_p95_ms": 1.5,
            "size_mb": 6.6,
        }
    ]
    out = compare.render_comparison(rows)
    assert out.startswith("# Model comparison")
    assert "| task | model | macro_f1 | train_seconds | latency_p95_ms | size_mb |" in out
    # One data row with the formatted values.
    assert "| queue | tfidf | 0.5000 | 12.0 | 1.50 | 6.6 |" in out


def test_dir_size_mb_counts_file_and_directory(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 2_000_000)
    assert compare.dir_size_mb(f) == pytest.approx(2.0, abs=0.01)

    nested = tmp_path / "model"
    (nested / "sub").mkdir(parents=True)
    (nested / "w.bin").write_bytes(b"y" * 1_000_000)
    (nested / "sub" / "v.bin").write_bytes(b"z" * 500_000)
    assert compare.dir_size_mb(nested) == pytest.approx(1.5, abs=0.01)


def test_latency_p95_runs_warmup_then_times_repeats() -> None:
    calls: list[int] = []

    def predict(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return np.zeros((len(texts), 2))

    p95 = compare.single_ticket_latency_p95(predict, ["a", "b"], repeats=10, warmup=3)
    assert p95 >= 0.0
    assert len(calls) == 13  # 3 warmup + 10 timed, one ticket each


def test_evaluate_model_uses_argmax_and_reports_latency() -> None:
    class _FakeModel:
        classes_ = np.array(["billing", "tech"])

        def predict_proba(self, texts: list[str]) -> np.ndarray:
            # Always confident about "tech" (index 1).
            return np.tile([0.1, 0.9], (len(texts), 1))

    result = compare.evaluate_model_on_validation(_FakeModel(), ["x", "y"], ["tech", "tech"])
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["latency_p95_ms"] >= 0.0
