import unittest

from visionparse.extraction.structured_text import clean_text, extract_menu_items, structure_text


class StructuredTextTests(unittest.TestCase):
    def test_extracts_menu_items_with_category(self):
        text = """
        Starters
        Samosa £3.50
        Chicken Pakora £5.99

        Mains:
        Lamb Karahi £12.95
        """

        items = extract_menu_items(text)

        self.assertEqual([item.name for item in items], ["Samosa", "Chicken Pakora", "Lamb Karahi"])
        self.assertEqual(items[0].category, "Starters")
        self.assertEqual(items[-1].category, "Mains")

    def test_structure_text(self):
        document = structure_text("Tea £1.50\nCoffee £2.00")

        self.assertEqual(len(document.lines), 2)
        self.assertEqual(len(document.prices), 2)
        self.assertEqual(len(document.items), 2)

    def test_clean_text_fixes_common_encoding_noise(self):
        self.assertEqual(clean_text("Fish Â£8.99"), "Fish £8.99")


if __name__ == "__main__":
    unittest.main()

