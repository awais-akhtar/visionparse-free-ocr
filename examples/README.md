# Examples

The examples are written so they work with your own images and with the legacy research images if you still have them locally.

The public repo does not commit third-party or generated images. Put sample images in `examples/assets/` or point the scripts at `.visionparse_private_legacy/` while working locally.

```bash
python examples/free_ocr_layout.py path/to/menu.jpg
python examples/yolo_then_ocr.py path/to/menu.jpg --model models/best.pt
```

