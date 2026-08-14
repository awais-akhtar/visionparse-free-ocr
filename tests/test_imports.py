import unittest


class ImportTests(unittest.TestCase):
    def test_public_api_imports_without_heavy_dependencies(self):
        import visionparse

        self.assertEqual(visionparse.__version__, "0.1.2")
        self.assertTrue(hasattr(visionparse, "extract_prices"))


if __name__ == "__main__":
    unittest.main()
