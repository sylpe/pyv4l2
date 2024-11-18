import cv2
import media_helpers
import mmap
import numpy as np
import time
import v4l2
from mediapipe import MediaPipe, MediaConfig

# Global frame counter
frame_counter = 0


def unpack_raw10(data, width=None, stride=None):
    """
    Unpack V4L2_PIX_FMT_Y10P format (and similar Raw10 Bayer Packed formats).

    Unpack 4 raw10-pixels stored on 5 bytes in memory with below pattern :
    Y'0[9:2]  |  Y'1[9:2]  |  Y'2[9:2]  |  Y'3[9:2]  |  Y'3[1:0] Y'2[1:0] Y'1[1:0] Y'0[1:0]
    """
    # Load data in numpy array
    data = np.frombuffer(data, dtype=np.uint8)

    # Optional : if stride is provided, crop to width (with respect to 10/8 ratio)
    if width is not None and stride is not None and int(width * 10 / 8) < stride:
        data = data.reshape(-1, stride)
        data = data[:, : int(width * 10 / 8)]

    # Reshape to 5 columns to prepare upcoming transformations
    data = data.reshape(-1, 5)

    # Prepare resulting array (4*uint16 instead of 5*uint8)
    data_unpacked = np.empty((data.shape[0], 4), dtype=np.uint16)

    # Each pixel is build with the concatenation of 2 bits coming from cell 5
    data_unpacked[:, 0:4] = data[:, 0:4]
    data_unpacked[:, 0] = data_unpacked[:, 0] << 2 | (data[:, 4] & 0x03)
    data_unpacked[:, 1] = data_unpacked[:, 1] << 2 | ((data[:, 4] & 0x0C) >> 2)
    data_unpacked[:, 2] = data_unpacked[:, 2] << 2 | ((data[:, 4] & 0x30) >> 4)
    data_unpacked[:, 3] = data_unpacked[:, 3] << 2 | ((data[:, 4] & 0xC0) >> 6)

    return data_unpacked


def save_frame_to_file(streamer: v4l2.CaptureStreamer):
    global frame_counter
    vbuf = streamer.dequeue()
    stride = streamer.strides[0]
    height = streamer.height
    width = streamer.width
    pix_fmt = v4l2.fourcc_to_str(streamer.format.v4l2_fourcc)
    filename = 'Frame_{}x{}_{}_{}.png'.format(width, height, pix_fmt, frame_counter)
    frame_counter += 1
    with mmap.mmap(
        streamer.fd, stride * height, mmap.MAP_SHARED, mmap.PROT_READ, offset=vbuf.offset
    ) as frame:
        if pix_fmt == 'YUYV':
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 2)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_YUV2BGR_YUYV)
        elif pix_fmt == 'GREY':
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, stride)
        elif pix_fmt == 'Y10 ':
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2)) << 6
        elif pix_fmt == 'Y10P':
            img_bgr = unpack_raw10(frame, width, stride).reshape(height, width) << 6
        elif pix_fmt == 'Y16 ':
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2))
        elif pix_fmt == 'BA81':  # BGGR8
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, stride)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerRG2BGR)
        elif pix_fmt == 'GBRG':  # GBRG8
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, stride)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGR2BGR)
        elif pix_fmt == 'GRBG':  # GRBG8
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, stride)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGB2BGR)
        elif pix_fmt == 'RGGB':  # RGGB8
            img_bgr = np.frombuffer(frame, dtype=np.uint8).reshape(height, stride)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerBG2BGR)
        elif pix_fmt == 'BG10':  # BGGR10
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2)) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerRG2BGR)
        elif pix_fmt == 'GB10':  # GBRG10
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2)) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGR2BGR)
        elif pix_fmt == 'BA10':  # GRBG10
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2)) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGB2BGR)
        elif pix_fmt == 'RG10':  # RGGB10
            img_bgr = np.frombuffer(frame, dtype=np.uint16).reshape(height, int(stride / 2)) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerBG2BGR)
        elif pix_fmt == 'pBAA':  # BGGR10P
            img_bgr = unpack_raw10(frame, width, stride).reshape(height, width) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerRG2BGR)
        elif pix_fmt == 'pGAA':  # GBRG10P
            img_bgr = unpack_raw10(frame, width, stride).reshape(height, width) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGR2BGR)
        elif pix_fmt == 'pgAA':  # GRBG10P
            img_bgr = unpack_raw10(frame, width, stride).reshape(height, width) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerGB2BGR)
        elif pix_fmt == 'pRAA':  # RGGB10P
            img_bgr = unpack_raw10(frame, width, stride).reshape(height, width) << 6
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BayerBG2BGR)
        else:
            raise Exception(
                f'Unsupported {pix_fmt} Pixel Format, please update save_frame_to_file() accordingly'
            )
    # Crop at width (in case stride != width)
    img_bgr = img_bgr[:, :width]
    cv2.imwrite(filename, img_bgr)
    streamer.queue(vbuf)


def get_format(media_conf):
    (width, height, _) = (
        media_conf.entities_conf[media_conf.get_src_entity().name].pads[0].get('fmt')
    )
    (_, _, pix_fmt) = media_conf.entities_conf[media_conf.get_dst_entities()[0].name].dev_fmt
    return (width, height, pix_fmt)


camera = 'vd56g3 10-0010'
pi4_media_topology = {
    'subdevs': [
        {
            'entity': camera,
            'pads': [{'pad': (0, 0), 'fmt': (0, 0, 0)}],
        }
    ],
    'devices': [{'entity': 'unicam-image', 'fmt': (0, 0, 0)}],
    'links': [{'src': (camera, 0), 'dst': ('unicam-image', 0)}],
}


# From given topology, produces a partial MediaConfig
media = media_helpers.find_media_node(camera)
partial_config = MediaConfig(camera, media=media, **pi4_media_topology)

# Generate a list of MediaConfig for each resolution/mbus_code of the source subdevice
all_media_configs = media_helpers.media_config_generator(partial_config)

# Iterate and stream over each Media Pipe configuration
for conf in all_media_configs:
    width, height, pix_fmt = get_format(conf)
    print(f'Streaming {width}x{height} - {pix_fmt} ...')
    # Initialize and configure media pipe
    media_pipe = MediaPipe(conf)
    media_pipe.setup_links()
    media_pipe.configure_subdevs()
    # Provide callback on new frame, then start stream
    frame_counter = 0
    media_pipe.start_stream(on_frame_cb=save_frame_to_file)
    time.sleep(1)
    media_pipe.stop_stream()
