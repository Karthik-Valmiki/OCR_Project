from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Basic preprocessing for Phase - 1
     -Ensure RGB
     -Optional resizing hook
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image input")

    image = image.convert("RGB")

    # Resizing large images
    max_dimension = 1024
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension))

    return image
