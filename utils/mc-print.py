#!/usr/bin/python3

import v4l2
import v4l2.uapi
import argparse
import errno


def print_selection(subdev, pad):
    try:
        r = subdev.get_selection(v4l2.uapi.V4L2_SEL_TGT_CROP_BOUNDS, pad.index)
        print(f'    crop.bounds:({r.left},{r.top})/{r.width}✕{r.height}')
    except OSError as e:
        if e.errno != errno.ENOTTY:
            print(f'    crop.bounds:({e})')

    try:
        r = subdev.get_selection(v4l2.uapi.V4L2_SEL_TGT_CROP, pad.index)
        print(f'    crop:({r.left},{r.top})/{r.width}✕{r.height}')
    except OSError as e:
        if e.errno != errno.ENOTTY:
            print(f'    crop:({e})')


def print_routes(subdev):
    routes = [r for r in subdev.get_routes() if r.is_active]
    if len(routes) > 0:
        print('  Routing:')
        for r in routes:
            print('    {}/{} -> {}/{}'.format(r.sink_pad, r.sink_stream, r.source_pad, r.source_stream))

def print_videodev_pad(videodev):
    try:
        fmt = videodev.get_format()

        if fmt.type == v4l2.uapi.V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE:
            f = fmt.fmt.pix_mp
            fmt = f'{f.width}x{f.height}/{v4l2.fourcc_to_str(f.pixelformat)} numplanes:{f.num_planes}'
        elif fmt.type == v4l2.uapi.V4L2_BUF_TYPE_VIDEO_CAPTURE:
            f = fmt.fmt.pix
            fmt = f'{f.width}x{f.height}/{v4l2.fourcc_to_str(f.pixelformat)}'
        elif fmt.type == v4l2.BufType.META_CAPTURE.value or fmt.type == v4l2.BufType.META_OUTPUT.value:
            f = fmt.fmt.meta
            fmt = f'{f.buffersize}/{v4l2.fourcc_to_str(f.dataformat)}'
        else:
            print("XXXXX", fmt.type)

        print(f'    {fmt}')
    except OSError as e:
        if e.errno != errno.ENOTTY:
            print(f'    <{e}>')

def print_streams(subdev, pad, streams):
    for s in streams:
        try:
            fmt = subdev.get_format(pad.index, s)
            f = fmt.format

            try:
                bfmt = v4l2.BusFormat(f.code).name
            except ValueError:
                bfmt = f'0x{f.code:x}'

            print(f'    Stream{s} {f.width}✕{f.height}/{bfmt} field:{f.field} colorspace:{f.colorspace} quantization:{f.quantization} xfer:{f.xfer_func} flags:{f.flags}')
        except OSError as e:
            if e.errno != errno.ENOTTY:
                print(f'    Stream{s} <{e}>')

        try:
            ival = subdev.get_frame_interval(pad.index, s)
            print(f'            Interval {ival[0]}/{ival[1]} = {ival[1] / ival[0]} fps')
        except OSError as e:
            if e.errno != errno.ENOTTY:
                print(f'            Interval {e}')

        print_selection(subdev, pad)

def print_pads(ent, subdev, videodev):
    if subdev:
        routes = [r for r in subdev.get_routes() if r.is_active]
    else:
        routes = []

    for pad in ent.pads:
        links = list([l for l in pad.links if l.is_enabled])

        # Don't show external pads that have no enabled links
        if len(links) == 0 and not pad.is_internal:
            continue

        print(f'  Pad{pad.index} [{pad.flags.name}]')

        for link in links:
            remote_pad = link.source_pad if link.sink_pad == pad else link.sink_pad
            dir = '->' if pad.is_source else '<-'
            print(f'    {dir} \'{remote_pad.entity.name}\':{remote_pad.index} [{v4l2.MediaLinkFlag(link.flags).name}]')

        streams = set([r.source_stream for r in routes if r.source_pad == pad.index] + [r.sink_stream for r in routes if r.sink_pad == pad.index])

        if len(streams) == 0:
            streams = [ 0 ]

        streams = sorted(streams)

        if videodev:
            print_videodev_pad(videodev)

        if subdev:
            print_streams(subdev, pad, streams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='/dev/media0', help='media device')
    parser.add_argument('pattern', nargs='?', help='Entity pattern to show')
    args = parser.parse_args()

    md = v4l2.MediaDevice(args.device)

    if args.pattern:
        pat = args.pattern.lower()

        entities = [ent for ent in md.entities if pat in ent.name.lower() or pat in (ent.interface.dev_path if ent.interface else '')]
        print_queue = entities
        recurse = False
    else:
        entities = list(md.entities)

        # start with source-only subdevs (sensors)
        print_queue = sorted([ent for ent in entities if not any(p.is_sink and not p.is_internal for p in ent.pads)], key=lambda e: e.name)

        recurse = True

    flatten = lambda t: [item for sublist in t for item in sublist]

    printed = set()

    while len(print_queue) > 0:
        ent = print_queue.pop(0)

        if ent in printed:
            continue

        printed.add(ent)

        links = flatten([ p.links for p in ent.pads if p.is_source ])

        if recurse:
            print_queue += [ l.sink.entity for l in links ]

        pads_with_links = any([p for p in ent.pads if any([l for l in p.links if l.is_enabled])])
        if not pads_with_links:
            continue

        print(ent.name, ent.interface.dev_path if ent.interface else '')

        if ent.interface and ent.interface.is_subdev:
            subdev = v4l2.SubDevice(ent.interface.dev_path)
        else:
            subdev = None

        if ent.interface and ent.interface.is_video:
            videodev = v4l2.VideoDevice(ent.interface.dev_path)
        else:
            videodev = None

        print_pads(ent, subdev, videodev)

        if subdev:
            print_routes(subdev)

        print()


if __name__ == '__main__':
    main()
