import numpy as np
import cv2

# Path to the .npy file
file_path = "Data/ukulele2/color_timesteps.npy"
file_path2 = "Data/ukulele2/color.npy"
# file_path = "Data/ukulele2/depth_timesteps.npy"
# Load the data
data = np.load(file_path)
data2 = np.load(file_path2)
print(data.shape, data2.shape)
assert False
# print(data)
# Loop through the frames and display them
for i, frame in enumerate(data):
    # Convert the frame to uint8 if needed for display
    frame_uint8 = frame.astype(np.uint8)

    # Display the frame
    cv2.imshow("Video", frame_uint8)

    # Wait for 30 ms to mimic a frame rate of approximately 30 FPS
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()
