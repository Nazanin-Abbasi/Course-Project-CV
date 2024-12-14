"""
This file is used to define a cleaner interface for OpenNI.
"""
import os
import subprocess
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from primesense import openni2
from skimage.io import imsave
from skimage.util import img_as_ubyte

# CHANGE THIS TO INDICATE THE DEFAULT PATH TO OPENNI
DEFAULT_OPENNI_PATH = "C:\\Program Files\\OpenNI2\\Tools"


class StreamType(Enum):
    """
    Enumerated type to describe the type of stream:

    * COLOUR_STREAM, representing the RGB video feed.
    * DEPTH_STREAM, representing the depth data.
    """
    COLOUR_STREAM = openni2.SENSOR_COLOR
    DEPTH_STREAM = openni2.SENSOR_DEPTH


def initialise(path_to_openni: str = DEFAULT_OPENNI_PATH):
    """
    Initialise OpenNI devices.

    :param path_to_openni: Path to the OpenNI installation.
    :return: None
    """
    openni2.initialize(path_to_openni)


def open_oni_file(filename: str) -> openni2.Device:
    """
    Read a specified *.oni file and return the created `Device`.
    :param filename: path to the *.oni file.
    :return: the `openni2.Device` object created by reading the file.
    """

    filename_as_bytes = bytes(filename, 'utf-8')
    return openni2.Device.open_file(filename_as_bytes)


def read_from_device(device: openni2.Device, stream_type: StreamType, freq: int = 0, ref_timestamps=None) -> [NDArray, NDArray]:
    """
    Read from an OpenNI file device.

    Read the stream of the specified type from the OpenNI `*.oni`
    file which is accessed using the provided device. The playback
    speed is set to `-1`, indicating that the frames only advance
    with the API calls. This allows us to manually go through
    each frame and convert it into a NumPy array.

    :param device: OpenNI `Device` instance created using a `*.oni` file.
    :param stream_type: indicates the type of stream to extract.
    :return: a NumPy array consisting of all frames of the stream,
             with the zero-axis representing time.
    """
    stream = device.create_stream(stream_type.value)
    stream_outputs = []

    playback_support = openni2.PlaybackSupport(device)

    playback_support.repeat = False
    playback_support.speed = -1

    # This could be used later to move to real-time data processing.
    # stream.register_new_frame_listener(
    #     lambda s: stream_outputs.append(convert_frame_to_numpy_array(s.read_frame(), stream_type))
    # )

    stream.start()
    playback_support.seek(stream, 0)

    # Store the timestamps
    timestamps = []

    # Iterate through all the frames, extract them and convert them to `NDArrays`.
    for i in range(playback_support.get_number_of_frames(stream)):
        frame = stream.read_frame()
        stream.get_horizontal_fov()
        if freq == 0 or i % freq == 0:
            timestamps.append(frame.timestamp)
            stream_outputs.append(convert_frame_to_numpy_array(frame, stream_type))

    if ref_timestamps is not None:
        new_stream_outputs = []
        timestamps = np.array(timestamps)
        for t in ref_timestamps:
            diff = np.abs(timestamps-t)
            new_stream_outputs.append(stream_outputs[np.argmin(diff)])

        stream_outputs = new_stream_outputs

    # Merge all frames into one NDArray
    return np.stack(stream_outputs, axis=0), np.array(timestamps)


def save_as_video_frames(stream_data: NDArray, stream_type: StreamType, output_directory: str,
                         save_as_video: bool = False):
    """
    Export video frames.

    Export the contents of a NDArray representing the data from a stream as frames
    that can be combined to create a movie. The frames are stored in the specified
    output folder, in a subfolder with the name of the stream type. The frames
    have filename `stream_type`_`i` where `i` is the frame number, padded with up
    to four zeros. To generate a video, these frames can be passed into `ffmpeg`.

    :param stream_data: NDArray containing the frames from the stream,
                        with the zero-axis being the time.
    :param stream_type: Indicate whether the stream is a depth stream or a
                        colour stream (used for normalisation).
    :param output_directory: Directory in which to save the frames.
    :param save_as_video: generate a video from the frames. Requires ffmpeg.
    :return: None
    """
    if stream_type == StreamType.DEPTH_STREAM:
        normalized_outputs = stream_data / stream_data.max()
    else:
        normalized_outputs = stream_data

    output_directory_for_stream = os.path.join(os.path.expanduser(output_directory), stream_type.name)

    if not os.path.isdir(output_directory_for_stream):
        os.makedirs(output_directory_for_stream)

    for i in range(len(stream_data)):
        output_filename = f"{stream_type.name}_{i:05d}.png"
        output_path = os.path.join(output_directory_for_stream, output_filename)

        frame = normalized_outputs[i]

        # if stream_type == StreamType.DEPTH_STREAM:
        frame = img_as_ubyte(frame)

        imsave(output_path, frame)

        # write(output_path, rate_in=frame_rate, data=normalized_stacked_array, pix_fmt="yuv420p")

    if save_as_video:
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", os.path.join(output_directory_for_stream, f"{stream_type.name}_%05d.png"),
            os.path.join(output_directory, f"{stream_type.name}.mp4"),
            "-pix_fmt", "yuv420p"
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"ffmpeg returned with error code {result.returncode}")
            print(result.stderr)


def convert_frame_to_numpy_array(frame: openni2.VideoFrame, stream_type: StreamType) -> NDArray:
    """
    Convert an OpenNI frame into a NumPy array.

    :param frame: The frame to convert.
    :param stream_type: Indicates whether the stream is a depth stream or a colour stream.
    :return: NumPy `NDArray` containing the data from the frame.
    """

    if stream_type == StreamType.DEPTH_STREAM:
        frame_data = frame.get_buffer_as_uint16()
        width = frame.width
        height = frame.height

        arr = np.array(frame_data).reshape((height, width))
        return arr

    frame_data = frame.get_buffer_as_triplet()
    width = frame.width
    height = frame.height

    arr = np.array(frame_data).reshape((height, width, 3))
    return arr

import os
from pathlib import Path
if __name__ == "__main__":
    initialise()
    directory = "Data\\"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if not os.path.isfile(f):
            continue
        device = open_oni_file(f)
        x, t = read_from_device(device, StreamType.DEPTH_STREAM, freq=4)
        device.close()
        device = open_oni_file(f)
        x2, t = read_from_device(device, StreamType.COLOUR_STREAM, ref_timestamps=t)

        # x, t = read_from_device(device, StreamType.COLOUR_STREAM, freq=4) # need to comment this part because can't open the same stream twice

        new_dir = f[:-4]
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        np.save(new_dir+"\\depth.npy", x)
        np.save(new_dir+"\\color.npy", x2)

