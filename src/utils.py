import cv2


def resize_images(
    images: dict[str, cv2.typing.MatLike]
) -> dict[str, cv2.typing.MatLike]:
    """Resize the images to the common size (minimum of the two sizes)."""
    list_of_images = list(images.values())
    h, w = list_of_images[0].shape[:2]
    for img in list_of_images:
        h = min(h, img.shape[0])
        w = min(w, img.shape[1])
    return {name: cv2.resize(img, (w, h)) for name, img in images.items()}
