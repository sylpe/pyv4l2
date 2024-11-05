import time
import v4l2
from mediapipe import MediaPipe, MediaConfig

configs = {
    'vd56g3_1120x1360_raw8': {
        'subdevs': [
            {
                'entity': 'vd56g3 10-0010',
                'pads': [{'pad': (0, 0), 'fmt': (1120, 1360, v4l2.BusFormat.SGRBG8_1X8)}],
            }
        ],
        'devices': [{'entity': 'unicam-image', 'fmt': (1120, 1360, v4l2.PixelFormats.SGRBG8)}],
        'links': [{'src': ('vd56g3 10-0010', 0), 'dst': ('unicam-image', 0)}],
    }
}


def get_frame(streamer: v4l2.CaptureStreamer):
    vbuf = streamer.dequeue()
    print(f'New Frame: {streamer.width}x{streamer.height}')
    streamer.queue(vbuf)


# Convert the configs dictionary to Configuration objects
media_configs = {name: MediaConfig(name, **config) for name, config in configs.items()}

# Initialize and Configure Media Pipe
media_pipe = MediaPipe(media_configs['vd56g3_1120x1360_raw8'])
media_pipe.setup_links()
media_pipe.configure_subdevs()

# Provide callback on new frame, then start stream
media_pipe.start_stream(on_frame_cb=get_frame)
time.sleep(10)
media_pipe.stop_stream()
