import unittest

import torch
import torch.nn as nn

from cybernoodles.training.watchdog import (
    assert_finite_gradients,
    assert_finite_tensors,
    ensure_finite_scalar,
    ensure_optimizer_advanced,
    ensure_parameter_moved,
    optimizer_step_total,
    parameter_delta_l2,
    parameter_snapshot,
)


class TrainingWatchdogTests(unittest.TestCase):
    def test_rejects_nonfinite_scalar_and_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            ensure_finite_scalar("loss", float("nan"))

        with self.assertRaisesRegex(RuntimeError, "contains non-finite"):
            assert_finite_tensors("batch", [torch.tensor([1.0, float("inf")])])

    def test_detects_optimizer_step_and_parameter_movement(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        before_steps = optimizer_step_total(optimizer)
        before_params = parameter_snapshot(model)

        loss = model(torch.ones(4, 2)).pow(2).mean()
        ensure_finite_scalar("loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert_finite_gradients("grad", model)
        optimizer.step()

        ensure_optimizer_advanced("optimizer", before_steps, optimizer_step_total(optimizer))
        ensure_parameter_moved("model", parameter_delta_l2(model, before_params))

    def test_detects_missing_optimizer_progress(self):
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            ensure_optimizer_advanced("optimizer", 3, 3)

        with self.assertRaisesRegex(RuntimeError, "did not change"):
            ensure_parameter_moved("model", 0.0)


if __name__ == "__main__":
    unittest.main()
