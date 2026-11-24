from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch

from scripts.train_cctdh import run as run_cctdh
from scripts.train_sde import run as run_sde


class FakeRuntime:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.05)

    def trainable_parameters(self):
        return [self.parameter]

    def response_nll(self, rows, target_field):
        assert all(target_field in row for row in rows)
        return torch.stack(
            [(self.parameter.sum() - 1.0).square() for _ in rows]
        )

    def retention_gradients(self, rows, target_field, batch_size):
        raise AssertionError("the test supplies a precomputed basis")

    def detail_loss(self, rows):
        return (self.parameter[0] + 0.5).square()

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.parameter.detach(), output_dir / "adapter.pt")


def _write(frame, path):
    frame.to_json(path, orient="records", lines=True)


def _basis(path):
    torch.save({"basis": torch.empty(2, 0)}, path)


def test_sde_training_entrypoint_runs_fixed_pools(tmp_path):
    examples = pd.DataFrame(
        [
            {"qa_id": "r", "source_id": "1", "gold_answer": "g", "example_state": "REPAIR"},
            {"qa_id": "p", "source_id": "2", "gold_answer": "g", "example_state": "PROTECT"},
            {"qa_id": "o", "source_id": "3", "gold_answer": "g", "example_state": "NO_ACTION"},
        ]
    )
    responses = pd.DataFrame(
        [
            {"qa_id": name, "gold_response": "gold", "base_response": "base"}
            for name in ("r", "p", "o")
        ]
    )
    example_path, response_path = tmp_path / "examples.jsonl", tmp_path / "responses.jsonl"
    _write(examples, example_path)
    _write(responses, response_path)
    basis = tmp_path / "basis.pt"
    _basis(basis)
    output = tmp_path / "sde"
    runtime = FakeRuntime()
    before = runtime.parameter.detach().clone()
    summary = run_sde(
        Namespace(
            config=Path("configs/qwen3b.yaml"),
            examples=example_path,
            responses=response_path,
            runtime="unused:factory",
            runtime_arg=[],
            retention_basis=basis,
            output=output,
            batch_size=1,
            basis_batch_size=1,
            steps=2,
            seed=18,
            gradient_clip_norm=None,
            require_resolved=False,
            overwrite=False,
        ),
        runtime=runtime,
    )
    assert summary["steps"] == 2
    assert not torch.equal(before, runtime.parameter)
    assert (output / "adapter/adapter.pt").is_file()


def test_cctdh_training_entrypoint_runs_gradient_sequence(tmp_path):
    repair = pd.DataFrame([{"qa_id": "r", "gold_response": "gold"}])
    protect = pd.DataFrame(
        [{"qa_id": "p", "retention_target_response": "parent"}]
    )
    anchor = pd.DataFrame(
        [{"qa_id": "a", "retention_target_response": "parent"}]
    )
    paths = []
    for name, frame in (("repair", repair), ("protect", protect), ("anchor", anchor)):
        path = tmp_path / f"{name}.jsonl"
        _write(frame, path)
        paths.append(path)
    basis = tmp_path / "basis.pt"
    _basis(basis)
    output = tmp_path / "cctdh"
    runtime = FakeRuntime()
    summary = run_cctdh(
        Namespace(
            config=Path("configs/qwen3b.yaml"),
            validated_repair=paths[0],
            protect=paths[1],
            anchor=paths[2],
            runtime="unused:factory",
            runtime_arg=[],
            retention_basis=basis,
            output=output,
            repair_target_field="gold_response",
            retention_target_field="retention_target_response",
            batch_size=1,
            basis_batch_size=1,
            steps=2,
            seed=18,
            gradient_clip_norm=1.0,
            require_resolved=False,
            overwrite=False,
        ),
        runtime=runtime,
    )
    assert summary["steps"] == 2
    assert "conflict_triggered" in summary["last"]
    assert (output / "adapter/adapter.pt").is_file()
