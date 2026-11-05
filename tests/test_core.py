import tempfile
import unittest
from pathlib import Path

from guitar_chords import CHORDS
from guitar_chords.data import split_dataset
from guitar_chords.metrics import classification_metrics


class CoreTests(unittest.TestCase):
    def test_placeholder_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "placeholder"):
                split_dataset(Path(directory))

    def test_splits_are_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for chord in CHORDS:
                (root / chord).mkdir()
                for index in range(10):
                    (root / chord / f"{index}.png").touch()
            splits = split_dataset(root)
            self.assertEqual(splits, split_dataset(root))
            self.assertEqual([len(s) for s in splits], [77, 22, 11])
            self.assertEqual(len({p for group in splits for p, _ in group}), 110)

    def test_metrics(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["A", "Am"])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + .8) / 2)


if __name__ == "__main__":
    unittest.main()
