from io import BytesIO

from PIL import Image
import io

from typing import Union


def bytes_to_image(binary_image: bytes) -> Image.Image:
    """Convert image from bytes to PIL RGB format

    Args:
        binary_image (bytes): The binary representation of the image

    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def image_to_bytes(image: Union[Image.Image, str]) -> BytesIO:
    """
    Convert PIL image to Bytes

    Args:
    image (Image): A PIL image instance

    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    # if image is str type
    image = Image.open(image)

    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=95)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image
