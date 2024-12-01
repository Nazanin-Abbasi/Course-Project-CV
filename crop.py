import numpy as np
import cv2

# Global variables to store the polygon points
polygon_points = []


# Mouse callback function to capture the points of the polygon
def draw_polygon(event, x, y, flags, param):
    global polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add points to the polygon list when the left mouse button is clicked
        polygon_points.append((x, y))

    if event == cv2.EVENT_MOUSEMOVE:
        # Optionally, you could visualize the current drawing during the mouse movement
        if len(polygon_points) > 0:
            temp_img = param.copy()
            cv2.polylines(
                temp_img,
                [np.array(polygon_points)],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.imshow("Draw Polygon", temp_img)

    if event == cv2.EVENT_LBUTTONUP:
        # If the mouse button is released, close the polygon by connecting the last point to the first point
        polygon_points.append((x, y))
        cv2.polylines(
            param,
            [np.array(polygon_points)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imshow("Draw Polygon", param)
        print("Polygon points: ", polygon_points)


def crop_polygon_region(input_image_path, output_image_path):
    # Read the image
    image = cv2.imread(input_image_path)
    temp_image = image.copy()

    # Set the mouse callback function to capture the points
    cv2.imshow("Draw Polygon", temp_image)
    cv2.setMouseCallback("Draw Polygon", draw_polygon, param=temp_image)

    # Wait for the user to draw the polygon
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create a mask for the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if len(polygon_points) > 2:
        # Fill the polygon area with white (255) on the mask
        cv2.fillPoly(mask, [np.array(polygon_points)], color=255)

    # Apply the mask to the image to get the cropped region
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)

    # Optionally, show the cropped image
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
input_image_path = "ukulele2_frames/frame_121.png"  # Path to your input PNG image
output_image_path = (
    "ukulele2_frames/cropped_frame_121.png"  # Path to save the cropped image
)
crop_polygon_region(input_image_path, output_image_path)
