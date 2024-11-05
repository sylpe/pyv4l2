import selectors
import threading
import v4l2
from typing import List, Dict, Optional, Callable, Any


class NodeConfig:
    def __init__(
        self,
        pads_conf: Optional[List[Dict[str, Any]]] = None,
        routes_conf: Optional[List[Dict[str, Any]]] = None,
        dev_fmt: Optional[Dict[str, Any]] = None,
    ):
        self.pads = pads_conf
        self.routes = routes_conf
        self.dev_fmt = dev_fmt


class MediaConfig:
    def __init__(
        self,
        name: str,
        subdevs: List[Dict[str, Any]],
        devices: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        media: v4l2.MediaDevice = v4l2.MediaDevice('/dev/media0'),
    ):
        self.name = name
        self.media = media
        self.md_entities: Dict[str, v4l2.MediaEntity] = {}
        self.md_links: List[v4l2.MediaLink] = []
        self.entities_conf: Dict[str, NodeConfig] = {}

        self._populate_entities(subdevs, devices)
        self._populate_links(links)

    def _populate_entities(
        self, subdevs: List[Dict[str, Any]], devices: List[Dict[str, Any]]
    ) -> None:
        for subdev in subdevs:
            entity = self.media.find_entity(subdev['entity'])
            if entity is None:
                raise ValueError(f'Failed to find subdev entity {subdev["entity"]}')
            node_conf = NodeConfig(pads_conf=subdev.get('pads'), routes_conf=subdev.get('routing'))
            self.md_entities[entity.name] = entity
            self.entities_conf[entity.name] = node_conf

        for device in devices:
            entity = self.media.find_entity(device['entity'])
            if entity is None:
                raise ValueError(f'Failed to find device entity {device["entity"]}')
            node_conf = NodeConfig(dev_fmt=device.get('fmt'))
            self.md_entities[entity.name] = entity
            self.entities_conf[entity.name] = node_conf

    def _populate_links(self, links: List[Dict[str, Any]]) -> None:
        for link in links:
            src_ent, src_pad_idx = link['src']
            sink_ent, sink_pad_idx = link['dst']

            if src_ent not in self.md_entities:
                raise ValueError(f'Unknown source entity {src_ent}')
            if sink_ent not in self.md_entities:
                raise ValueError(f'Unknown sink entity {sink_ent}')

            src_pad = self.md_entities[src_ent].pads[src_pad_idx]
            match_link = None

            for l in src_pad.links:
                if (
                    l.sink_pad.entity == self.md_entities[sink_ent]
                    and l.sink_pad.index == sink_pad_idx
                ):
                    match_link = l
                    break

            if match_link is None:
                raise ValueError(f'Failed to find link between {link["src"]} and {link["dst"]}')
            else:
                self.md_links.append(match_link)

    def get_src_entity(self) -> v4l2.MediaEntity:
        source_entities = set(link.source.entity for link in self.md_links)
        destination_entities = set(link.sink.entity for link in self.md_links)
        unique_sources = list(source_entities - destination_entities)

        # Only 1 source; it must be a subdevice
        if len(unique_sources) != 1:
            raise ValueError('Source entity is not unique')
        source_entity = unique_sources[0]
        if not source_entity.interface.is_subdev:
            raise ValueError(f'Source entity {source_entity.name} is not a subdevice')

        return source_entity

    def get_dst_entities(self) -> List[v4l2.MediaEntity]:
        source_entities = set(link.source.entity for link in self.md_links)
        destination_entities = set(link.sink.entity for link in self.md_links)
        unique_destinations = list(destination_entities - source_entities)

        # Possibly multiple destinations; It must be video devices
        for destination_entity in unique_destinations:
            if not destination_entity.interface.is_video:
                raise ValueError(f'Source entity {destination_entity.name} is not a video device')

        return unique_destinations


class MediaPipe:
    def __init__(self, config: MediaConfig, num_bufs: int = 4) -> None:
        self.config = config
        self.mediadevice = config.media
        self.subdevices: Dict[str, v4l2.SubDevice] = {}
        self.video_devices: Dict[str, v4l2.VideoDevice] = {}
        self.streamers: Dict[str, v4l2.CaptureStreamer] = {}
        self.num_bufs = num_bufs
        self.selector = selectors.DefaultSelector()
        self.stream_thread = None
        self.stop_flag = threading.Event()

        self._load_devices()

    def __repr__(self) -> str:
        subdev_names = ' -> '.join(self.subdevices.keys())
        video_dev_names = ' : '.join(self.video_devices.keys())
        return f'MediaPipe({subdev_names} => {video_dev_names})'

    def _load_devices(self) -> None:
        for name, entity in self.config.md_entities.items():
            if entity.interface.is_subdev:
                subdev = v4l2.SubDevice(entity.interface.dev_path)
                if not subdev:
                    raise ValueError(f'No subdev for entity {name}')
                self.subdevices[name] = subdev
            if entity.interface.is_video:
                dev = v4l2.VideoDevice(entity.interface.dev_path)
                if not dev:
                    raise ValueError(f'No device for entity {name}')
                self.video_devices[name] = dev

    def disable_all_media_links(self) -> None:
        for entity in self.mediadevice.entities:
            for link in entity.pad_links:
                if not link.is_immutable:
                    link.disable()

    def setup_links(self) -> None:
        self.disable_all_media_links()
        for link in self.config.md_links:
            if not link.is_immutable:
                link.enabled = True
                link.enable()

    def configure_subdevs(self) -> None:
        for entity_name, subdev in self.subdevices.items():
            node_conf = self.config.entities_conf[entity_name]

            if node_conf.pads is not None:
                self._configure_pads(subdev, node_conf.pads, entity_name)

            if node_conf.routes is not None:
                self._configure_routes(subdev, node_conf.routes, entity_name)

    def _configure_pads(
        self, subdev: v4l2.SubDevice, pads: List[Dict[str, Any]], entity_name: str
    ) -> None:
        for pad_conf in pads:
            pad_idx, stream_id = pad_conf['pad']
            w, h, mbus_fmt = pad_conf['fmt']

            try:
                subdev.set_format(pad_idx, stream_id, w, h, mbus_fmt)
            except Exception as e:
                print(
                    f'Failed to set format for {entity_name}:{pad_idx}/{stream_id}: {w}x{h}-{mbus_fmt}'
                )
                raise e

            if 'crop.bounds' in pad_conf:
                x, y, w, h = pad_conf['crop.bounds']
                subdev.set_selection(
                    v4l2.uapi.V4L2_SEL_TGT_CROP_BOUNDS,
                    v4l2.uapi.v4l2_rect(x, y, w, h),
                    pad_idx,
                    stream_id,
                )

            if 'crop' in pad_conf:
                x, y, w, h = pad_conf['crop']
                subdev.set_selection(
                    v4l2.uapi.V4L2_SEL_TGT_CROP, v4l2.uapi.v4l2_rect(x, y, w, h), pad_idx, stream_id
                )

            if 'ival' in pad_conf:
                assert len(pad_conf['ival']) == 2
                subdev.set_frame_interval(pad_idx, stream_id, pad_conf['ival'])

    def _configure_routes(
        self, subdev: v4l2.SubDevice, routes: List[Dict[str, Any]], entity_name: str
    ) -> None:
        route_list = []
        for route_conf in routes:
            sink_pad, sink_stream = route_conf['src']
            source_pad, source_stream = route_conf['dst']

            route = v4l2.Route()
            route.sink_pad = sink_pad
            route.sink_stream = sink_stream
            route.source_pad = source_pad
            route.source_stream = source_stream
            route.flags = v4l2.uapi.V4L2_SUBDEV_ROUTE_FL_ACTIVE

            route_list.append(route)

        try:
            subdev.set_routes(route_list)
        except Exception as e:
            print(f'Failed to set routes for {entity_name}')
            raise e

    def _stream_loop(self) -> None:
        while not self.stop_flag.is_set():
            events = self.selector.select()
            for key, _ in events:
                callback = key.data
                callback()

    def start_stream(self, on_frame_cb: Callable[[v4l2.CaptureStreamer], None]) -> None:
        for entity_name, video_dev in self.video_devices.items():
            node_conf = self.config.entities_conf[entity_name]
            width, height, pix_fmt = node_conf.dev_fmt

            streamer = video_dev.get_capture_streamer(v4l2.MemType.MMAP, width, height, pix_fmt)
            streamer.reserve_buffers(self.num_bufs)

            for i in range(self.num_bufs):
                streamer.queue(streamer.vbuffers[i])

            self.streamers[entity_name] = streamer
            self.selector.register(
                streamer.fd, selectors.EVENT_READ, lambda data=streamer: on_frame_cb(data)
            )
            streamer.stream_on()

        self.stop_flag.clear()
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.start()

    def stop_stream(self) -> None:
        self.stop_flag.set()
        if self.stream_thread is not None:
            self.stream_thread.join()
        for entity_name, streamer in self.streamers.items():
            self.selector.unregister(streamer.fd)
            streamer.stream_off()
        self.streamers.clear()
