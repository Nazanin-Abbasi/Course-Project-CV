import cv2
import numpy as np


def read_png_to_np(filename):
    """
    Reads a PNG image file into a NumPy array and normalizes it to its original range.

    Args:
        filename (str): The path to the PNG file.

    Returns:
        np.ndarray: A single-channel NumPy array representing the depth frame.
    """
    # Read the PNG file (loads as 3-channel by default)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read the file: {filename}")

    # Convert to original scale (assuming the saved frame was normalized to 0-255)
    original_range = img.astype("float32") / 255.0  # Scale to 0-1
    original_depth = original_range * 1000  # Scale back to original depth range

    return original_depth.astype("float32")


# for i in range(56):
#     print(i)
#     formatted_number = f"{i:06d}"
#     # file = f"41ce0692a2/depth/{formatted_number}.png"
#     # a9cdfb67aa
#     file = f"a9cdfb67aa/depth/{formatted_number}.png"
#     f = read_png_to_np(file)

#     display_frame = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)
#     display_frame = display_frame.astype("uint8")

#     # Show the restored frame
#     cv2.imshow("Restored Frame", display_frame)

#     # Wait for a key press and then close the window
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
