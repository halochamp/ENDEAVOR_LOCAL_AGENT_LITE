# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp
"""No-server coverage for direct-VLM image message injection."""
from __future__ import annotations

from PIL import Image
from langchain_core.messages import HumanMessage

import react
from tools import read_image


def test_inject_pending_image_adds_openai_data_uri(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image)
    read_image._PENDING_IMAGES[:] = [str(image)]

    result = react._inject_pending_images([HumanMessage(content="describe this")])

    assert read_image._PENDING_IMAGES == []
    blocks = result[0].content
    assert blocks[0] == {"type": "text", "text": "describe this"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_inject_private_pending_snapshot_adds_data_uri():
    import config
    pending = config.DATA_DIR / "pending_images"
    pending.mkdir(parents=True, exist_ok=True)
    image = pending / "real-private-snapshot.png"
    Image.new("RGB", (2, 2), "white").save(image)
    read_image._PENDING_IMAGES[:] = [str(image)]

    result = react._inject_pending_images([HumanMessage(content="describe this")])
    assert result[0].content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert not image.exists()


def test_inject_pending_image_requeues_when_no_user_message(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image)
    read_image._PENDING_IMAGES[:] = [str(image)]

    assert react._inject_pending_images([]) == []
    assert read_image._PENDING_IMAGES == [str(image)]
    read_image._PENDING_IMAGES.clear()


def test_no_user_message_keeps_private_snapshot_on_disk():
    import config
    pending = config.DATA_DIR / "pending_images"
    pending.mkdir(parents=True, exist_ok=True)
    image = pending / "keep-for-next-call.png"
    Image.new("RGB", (2, 2), "white").save(image)
    read_image._PENDING_IMAGES[:] = [str(image)]

    assert react._inject_pending_images([]) == []
    assert image.exists()
    assert read_image._PENDING_IMAGES == [str(image)]
    read_image._PENDING_IMAGES.clear()
    image.unlink(missing_ok=True)


def test_unreadable_private_snapshot_is_cleaned_up(tmp_path, monkeypatch):
    import config
    pending = config.DATA_DIR / "pending_images"
    pending.mkdir(parents=True, exist_ok=True)
    image = pending / "broken.png"
    image.write_bytes(b"not an image")
    monkeypatch.setattr(read_image, "read_pending_snapshot_bytes", lambda _: None)
    monkeypatch.setattr(read_image, "read_safe_image_bytes", lambda _: None)

    assert react._image_to_data_uri(str(image)) is None
    assert not image.exists()
