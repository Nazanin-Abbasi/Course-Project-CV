import numpy as np
import cv2

filename = "Data/box/depth.npy"
filename = "Data/lama/depth.npy"
# filename = "Data/mold/depth.npy"
# filename = "Data/puzzle/depth.npy"
# filename = "Data/puzzle2/depth.npy"
# filename = "Data/shoe/depth.npy"
# filename = "Data/ukulele/depth.npy"
filename = "Data/ukulele2/depth.npy"

depth_frames = np.load(filename)
# print(depth_frames[100, 5])

# Define the depth range for lama
lower_bound = 500
upper_bound = 650

# Define the depth range for ukulele
lower_bound = 470
upper_bound = 850

# Apply the range filter
depth_frames[(depth_frames < lower_bound) | (depth_frames > upper_bound)] = 0

for i in range(depth_frames.shape[0]):
    # for lama
    # frame = depth_frames[i, 50:150, 100:200]

    # for ukulele
    frame = depth_frames[i, 30:200, 50:300]

    # Normalize the frame to the range [0, 255] and convert to uint8
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_normalized = frame_normalized.astype("uint8")

    # Resize for better visibility (optional)
    resized_frame = cv2.resize(
        frame_normalized, (400, 400), interpolation=cv2.INTER_NEAREST
    )

    # Display the frame
    # print(i)
    cv2.imshow("Animation", resized_frame)
    if cv2.waitKey(50) & 0xFF == 27:  # 50ms delay, press ESC to exit
        break
