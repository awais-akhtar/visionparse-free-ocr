import json
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_prices_command(self):
        completed = subprocess.run(
            [sys.executable, "-m", "visionparse", "prices", "Burger", "£7.99"],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload[0]["raw"], "£7.99")


if __name__ == "__main__":
    unittest.main()

