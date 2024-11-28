import numpy as np
import cv2

filename = "Data/box/depth.npy"
filename = "Data/lama/depth.npy"
filename = "Data/mold/depth.npy"
filename = "Data/puzzle/depth.npy"
filename = "Data/ukulele2/depth.npy"
filename = "Data/shoe/depth.npy"
data = np.load(filename)

for i in range(data.shape[0]):
    frame = data[i]
    # Normalize the frame to the range [0, 255] and convert to uint8
    frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame_normalized = frame_normalized.astype("uint8")

    # Resize for better visibility (optional)
    resized_frame = cv2.resize(
        frame_normalized, (400, 400), interpolation=cv2.INTER_NEAREST
    )

    # Display the frame
    cv2.imshow("Animation", resized_frame)
    if cv2.waitKey(50) & 0xFF == 27:  # 50ms delay, press ESC to exit
        break
