import unittest

from visionparse.detection.yolo import available_packaged_assets, load_class_names, packaged_model_path


class ModelAssetTests(unittest.TestCase):
    def test_packaged_yolo_assets_are_available(self):
        assets = available_packaged_assets()

        self.assertIn("yolov3.cfg", assets)
        self.assertIn("coco.names", assets)
        self.assertTrue(packaged_model_path("yolov3.cfg").exists())

    def test_coco_names_can_be_loaded(self):
        names = load_class_names(packaged_model_path("coco.names"))

        self.assertIn("person", names)
        self.assertIn("car", names)


if __name__ == "__main__":
    unittest.main()
