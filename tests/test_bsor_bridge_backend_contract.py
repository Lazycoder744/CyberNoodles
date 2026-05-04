import unittest
from types import SimpleNamespace
from unittest import mock

from cybernoodles import bsor_bridge


class BsorBridgeBackendContractTests(unittest.TestCase):
    def test_load_bsor_rust_backend_does_not_fallback_to_python(self):
        parsed = object()
        with mock.patch.object(bsor_bridge, "_run_bsor_tools", side_effect=RuntimeError("rust failed")), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=b"python bsor"),
        ) as open_file, mock.patch.object(bsor_bridge, "make_bsor", return_value=parsed) as make_bsor:
            with self.assertRaisesRegex(RuntimeError, "Unable to parse BSOR"):
                bsor_bridge.load_bsor("fake.bsor", backend="rust")

            open_file.assert_not_called()
            make_bsor.assert_not_called()

        with mock.patch.object(bsor_bridge, "_run_bsor_tools", side_effect=RuntimeError("rust failed")), mock.patch(
            "builtins.open",
            mock.mock_open(read_data=b"python bsor"),
        ), mock.patch.object(bsor_bridge, "make_bsor", return_value=parsed):
            self.assertIs(bsor_bridge.load_bsor("fake.bsor", backend="auto"), parsed)

    def test_validate_bsor_rust_backend_does_not_fallback_to_python(self):
        parsed = SimpleNamespace(
            frames=[],
            notes=[],
            walls=[],
            pauses=[],
            user_data=[],
            info=SimpleNamespace(songHash="", difficulty="", mode=""),
        )
        with mock.patch.object(bsor_bridge, "bsor_tools_available", return_value=True), mock.patch.object(
            bsor_bridge,
            "_run_bsor_tools",
            side_effect=RuntimeError("rust validation failed"),
        ), mock.patch.object(bsor_bridge, "load_bsor", return_value=parsed) as load_bsor:
            with self.assertRaisesRegex(RuntimeError, "rust validation failed"):
                bsor_bridge.validate_bsor("fake.bsor", backend="rust")

            load_bsor.assert_not_called()

        with mock.patch.object(bsor_bridge, "bsor_tools_available", return_value=True), mock.patch.object(
            bsor_bridge,
            "_run_bsor_tools",
            side_effect=RuntimeError("rust validation failed"),
        ), mock.patch.object(bsor_bridge, "load_bsor", return_value=parsed) as load_bsor:
            summary = bsor_bridge.validate_bsor("fake.bsor", backend="auto")

        load_bsor.assert_called_once_with("fake.bsor", backend="python")
        self.assertEqual(summary["validation_backend"], "python")
        self.assertFalse(summary["rust_validation_ok"])
        self.assertIn("rust validation failed", summary["rust_validation_error"])


if __name__ == "__main__":
    unittest.main()
