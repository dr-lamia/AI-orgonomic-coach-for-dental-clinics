"""Tests for posture utility helpers."""

import math
import unittest
from types import SimpleNamespace

from posture_utils import calculate_angle


class TestCalculateAngle(unittest.TestCase):
    def test_right_angle(self):
        a = SimpleNamespace(x=0.0, y=1.0)
        b = SimpleNamespace(x=0.0, y=0.0)
        c = SimpleNamespace(x=1.0, y=0.0)
        self.assertAlmostEqual(calculate_angle(a, b, c), 90.0, places=5)

    def test_straight_angle(self):
        a = (0.0, 0.0)
        b = (1.0, 0.0)
        c = (2.0, 0.0)
        self.assertAlmostEqual(calculate_angle(a, b, c), 180.0, places=5)

    def test_acute_angle(self):
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        c = (math.cos(math.pi / 3), math.sin(math.pi / 3))
        self.assertAlmostEqual(calculate_angle(a, b, c), 60.0, places=5)

    def test_iterable_support(self):
        def generator():
            yield 0.0
            yield 1.0

        a = generator()
        b = SimpleNamespace(x=0.0, y=0.0)
        c = (1.0, 0.0)
        self.assertAlmostEqual(calculate_angle(a, b, c), 90.0, places=5)


if __name__ == "__main__":
    unittest.main()
