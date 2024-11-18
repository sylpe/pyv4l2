import glob
import os
import v4l2
from copy import deepcopy
from mediapipe import MediaConfig
from typing import List


def find_media_node(entity_name: str) -> v4l2.MediaDevice:
    for path in glob.glob('/dev/media*'):
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue

        try:
            media_device = v4l2.MediaDevice(path)
            entity = media_device.find_entity(entity_name)

            if entity:
                return media_device
        finally:
            os.close(fd)

    raise FileNotFoundError(f'Entity name "{entity_name}" not found in system media nodes')


def media_config_generator(media_config: MediaConfig) -> List[MediaConfig]:
    """
    Generate a list of MediaConfig objects for each supported resolution/mbus_code of the source subdevice.

    Args:
        media_config (MediaConfig): The initial MediaConfig object containing the media pipeline configuration.

    Returns:
        List[MediaConfig]: A list of MediaConfig objects, each corresponding to a different supported resolution/mbus_code.

    Raises:
        ValueError: If the source entity is not a subdevice or if any required entity is not found.
    """
    # Identify source and destination entities (consider that there's only 1 destination)
    source_entity = media_config.get_src_entity()
    dest_entity = media_config.get_dst_entities()[0]

    src_subdev = v4l2.SubDevice(source_entity.interface.dev_path)
    pad = 0  # Assuming pad index 0
    mbus_codes = src_subdev.get_formats(pad)

    viddev = v4l2.VideoDevice(dest_entity.interface.dev_path)
    if viddev.has_mplane_capture:
        buftype = v4l2.BufType.VIDEO_CAPTURE_MPLANE
    elif viddev.has_capture:
        buftype = v4l2.BufType.VIDEO_CAPTURE
    else:
        raise ValueError(f"viddev {dest_entity.name} isn't a capture device")

    # Generate a dict of MediaConfig for mbus_code / resolution
    media_configs = []
    for mbus_code in mbus_codes:
        pix_fmts = viddev.get_formats(buftype, mbus_code)
        frame_sizes = src_subdev.get_framesizes(pad, mbus_code)

        # For 1 mbus code, multiple pixel formats can be outputted by CSI Rx
        for pix_fmt in pix_fmts:
            for width, height in frame_sizes:
                new_media_config = deepcopy(media_config)

                # Update the NodeConfig for the source entity
                new_media_config.entities_conf[source_entity.name].pads = [
                    {
                        'pad': (pad, 0),  # Assuming stream index 0
                        'fmt': (width, height, mbus_code),
                    }
                ]

                # Update NodeConfig of other subdevices/devices
                for entity_name, node_conf in new_media_config.entities_conf.items():
                    if entity_name != source_entity.name:
                        if node_conf.pads is not None:
                            for pad_conf in node_conf.pads:
                                pad_conf['fmt'] = (width, height, mbus_code)
                        if node_conf.dev_fmt is not None:
                            node_conf.dev_fmt = (width, height, pix_fmt)

                media_configs.append(new_media_config)

    return media_configs
