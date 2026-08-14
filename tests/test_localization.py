import unittest

from visionparse.ocr.localization import (
    TextToken,
    group_lines_into_blocks,
    group_tokens_into_lines,
    render_aligned_text,
)


class LocalizationTests(unittest.TestCase):
    def test_groups_tokens_into_visual_lines(self):
        tokens = [
            TextToken("Burger", (10, 10, 70, 25)),
            TextToken("£7.99", (180, 11, 230, 25)),
            TextToken("Fries", (10, 45, 55, 60)),
            TextToken("£2.50", (180, 46, 230, 60)),
        ]

        lines = group_tokens_into_lines(tokens)

        self.assertEqual([line.text for line in lines], ["Burger £7.99", "Fries £2.50"])

    def test_render_aligned_text_preserves_columns(self):
        tokens = [
            TextToken("Burger", (10, 10, 70, 25)),
            TextToken("£7.99", (180, 11, 230, 25)),
            TextToken("Fries", (10, 45, 55, 60)),
            TextToken("£2.50", (180, 46, 230, 60)),
        ]

        text = render_aligned_text(group_tokens_into_lines(tokens), char_width=10)

        self.assertIn("Burger", text)
        self.assertIn("£7.99", text)
        self.assertEqual(text.splitlines()[0].index("£"), text.splitlines()[1].index("£"))

    def test_groups_lines_into_blocks(self):
        tokens = [
            TextToken("Hot", (10, 10, 40, 25)),
            TextToken("Dog", (45, 10, 80, 25)),
            TextToken("Burger", (10, 35, 70, 50)),
            TextToken("Drinks", (300, 200, 360, 215)),
        ]
        lines = group_tokens_into_lines(tokens)
        blocks = group_lines_into_blocks(lines)

        self.assertEqual(len(blocks), 2)


if __name__ == "__main__":
    unittest.main()

