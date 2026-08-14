import unittest

from visionparse.ocr.engine import BaseOCREngine, OCRResult
from visionparse.pipelines.document_pipeline import DocumentPipeline


class FakeOCR(BaseOCREngine):
    name = "fake"

    def read(self, image):
        return OCRResult(
            text="Burger £7.99 Fries £2.50",
            engine="fake",
            metadata={
                "tokens": [
                    {"text": "Burger", "box": [10, 10, 70, 25]},
                    {"text": "£7.99", "box": [180, 10, 230, 25]},
                    {"text": "Fries", "box": [10, 45, 55, 60]},
                    {"text": "£2.50", "box": [180, 45, 230, 60]},
                ]
            },
        )


class PipelineLayoutTests(unittest.TestCase):
    def test_pipeline_exposes_layout_text(self):
        pipeline = DocumentPipeline(ocr_engine=FakeOCR())
        result = pipeline.run(__file__)

        self.assertIn("Burger", result.layout_text)
        self.assertIn("£2.50", result.layout_text)


if __name__ == "__main__":
    unittest.main()
