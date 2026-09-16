#!/usr/bin/env python3
import unittest

from device import device_kind, select_device


class SelectDeviceTests(unittest.TestCase):
    def test_explicit_request_wins(self):
        self.assertEqual(select_device("cpu", cuda=True, mps=True), "cpu")
        self.assertEqual(select_device("cuda:1", cuda=False, mps=False), "cuda:1")
        self.assertEqual(select_device("mps", cuda=True, mps=True), "mps")

    def test_cuda_before_mps(self):
        self.assertEqual(select_device("", cuda=True, mps=True), "cuda:0")

    def test_mps_when_no_cuda(self):
        self.assertEqual(select_device("", cuda=False, mps=True), "mps")

    def test_cpu_fallback(self):
        self.assertEqual(select_device("", cuda=False, mps=False), "cpu")

    def test_whitespace_request_ignored(self):
        self.assertEqual(select_device("  ", cuda=False, mps=True), "mps")

    def test_device_kind(self):
        self.assertEqual(device_kind("cuda:1"), "cuda")
        self.assertEqual(device_kind("mps"), "mps")
        self.assertEqual(device_kind(""), "cpu")


if __name__ == "__main__":
    unittest.main()
