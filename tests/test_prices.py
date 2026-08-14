from decimal import Decimal
import unittest

from visionparse.extraction.prices import extract_prices


class PriceExtractionTests(unittest.TestCase):
    def test_extracts_currency_and_decimal_prices(self):
        prices = extract_prices("Burger £7.99, fries 2.50, lassi Rs. 450")

        self.assertEqual([price.raw for price in prices], ["£7.99", "2.50", "Rs. 450"])
        self.assertEqual(prices[0].currency, "£")
        self.assertEqual(prices[0].amount, Decimal("7.99"))
        self.assertEqual(prices[2].amount, Decimal("450"))

    def test_plain_numbers_are_opt_in(self):
        self.assertEqual(extract_prices("Burger 7"), [])
        self.assertEqual(extract_prices("Burger 7", allow_plain_numbers=True)[0].amount, Decimal("7"))

    def test_thousand_separator(self):
        price = extract_prices("Catering PKR 1,250.00")[0]
        self.assertEqual(price.currency, "PKR")
        self.assertEqual(price.amount, Decimal("1250.00"))


if __name__ == "__main__":
    unittest.main()

