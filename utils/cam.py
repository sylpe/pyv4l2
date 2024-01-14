#!/usr/bin/python3

from collections import deque
import argparse
import pprint
import selectors
import sys
import time
import types

from cam_helpers import *
from cam_pisp import *
import v4l2


class Context(object):
    use_ipython: bool
    user_script: types.ModuleType | None
    subdevices: dict
    streams: list
    md: v4l2.MediaDevice
    buf_type: str
    use_display: bool
    kms_committed: bool
    print_config: bool
    config_only: bool
    delay: int
    save: bool
    tx: list[str]
    run_ipython: types.LambdaType


def init_setup(ctx: Context):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='Configuration name')
    parser.add_argument('-c', '--config-only', action='store_true', default=False, help='configure only')
    parser.add_argument('-s', '--save', action='store_true', default=False, help='save frames to files')
    parser.add_argument('-d', '--display', action='store_true', default=False, help='show frames on screen')
    parser.add_argument('-x', '--tx', nargs='?', type=str, default=None, const='all', help='send frames to a server')
    parser.add_argument('-t', '--type', type=str, help='buffer type (drm/v4l2)')
    parser.add_argument('-p', '--print', action='store_true', default=False, help='print config dict')
    parser.add_argument('-i', '--ipython', action='store_true', default=False, help='IPython mode')
    parser.add_argument('-S', '--script', help='User script')
    parser.add_argument('-D', '--delay', type=int, help='Delay in secs after the initial KMS modeset')
    args = parser.parse_args()

    ctx.print_config = args.print
    ctx.config_only = args.config_only
    ctx.delay = args.delay
    ctx.save = args.save

    ctx.use_ipython = args.ipython

    if ctx.use_ipython:
        from cam_ipython import run_ipython
        ctx.run_ipython = run_ipython

    if args.script:
        import importlib.util
        spec = importlib.util.spec_from_file_location('userscript', args.script)
        assert(spec)
        user_mod = importlib.util.module_from_spec(spec)
        assert(spec.loader)
        spec.loader.exec_module(user_mod)
        ctx.user_script = user_mod
    else:
        ctx.user_script = None

    if args.tx:
        ctx.tx = args.tx.split(',')
        ctx.net_tx = NetTX()
    else:
        ctx.tx = None
        ctx.net_tx = None

    if args.type and args.type not in ['drm', 'v4l2']:
        print('Bad buffer type', args.type)
        sys.exit(-1)

    ctx.use_display = args.display

    if args.type:
        ctx.buf_type = args.type
    else:
        if args.display:
            ctx.buf_type = 'drm'
        else:
            ctx.buf_type = 'v4l2'

    config = read_config(args.config_name)

    if config['media']:
        print('Configure media entities')

        md = v4l2.MediaDevice(*config['media'])

        disable_all_links(md)

        setup_links(md, config)

        ctx.subdevices = configure_subdevs(md, config)
    else:
        md = None
        ctx.subdevices = None

    ctx.md = md

    ctx.streams = config['devices']


def init_viddevs(ctx: Context):
    streams = ctx.streams

    for stream in streams:
        if 'num_bufs' not in stream:
            stream['num_bufs'] = 5

        if ctx.md:
            vid_ent = ctx.md.find_entity(stream['entity'])
            assert(vid_ent)

            if not 'dev_path' in stream:
                stream['dev_path'] = vid_ent.interface.dev_path

            vd = v4l2.VideoDevice(vid_ent.interface.dev_path)
        else:
            dev_path = v4l2.VideoDevice.find_video_device(*stream['entity'])
            stream['dev_path'] = dev_path

            vd = v4l2.VideoDevice(dev_path)

        stream['dev'] = vd


def init_kms(ctx: Context):
    streams = ctx.streams

    if ctx.buf_type == 'drm' or ctx.use_display:
        import kms
        card = kms.Card()
    else:
        card = None

    if ctx.use_display:
        res = kms.ResourceManager(card)
        conn = res.reserve_connector()
        crtc = res.reserve_crtc(conn)
        card.disable_planes()
        mode = conn.get_default_mode()
        modeb = kms.Blob(card, mode)

    num_planes = sum(1 for stream in streams if ctx.use_display and stream.get('display', True))

    display_idx = 0

    for i, stream in enumerate(streams):
        stream['display'] = ctx.use_display and stream.get('display', True)
        stream['embedded'] = stream.get('embedded', False)

        stream['id'] = i

        stream['w'] = stream['fmt'][0]
        stream['h'] = stream['fmt'][1]
        stream['fourcc'] = stream['fmt'][2]

        stream['kms-buf-w'] = stream['w']
        stream['kms-buf-h'] = stream['h']

        if stream.get('dra-plane-hack', False):
            # Hack to reserve the unscaleable GFX plane
            res.reserve_generic_plane(crtc, kms.PixelFormat.RGB565)

        if not 'kms-fourcc' in stream:
            if stream['fourcc'] == v4l2.MetaFormat.GENERIC_8:
                stream['kms-fourcc'] = kms.PixelFormat.RGB565
            elif stream['fourcc'] == v4l2.MetaFormat.GENERIC_CSI2_12:
                stream['kms-fourcc'] = kms.PixelFormat.RGB565
            else:
                #kms_fourcc = v4l2.pixelformat_to_drm_fourcc(stream['fourcc'])
                #stream['kms-fourcc'] = kms.fourcc_to_pixelformat(kms_fourcc)
                # XXX
                stream['kms-fourcc'] = stream['fourcc']

        if ctx.buf_type == 'drm' and stream.get('embedded', False):
            divs = [16, 8, 4, 2, 1]
            for div in divs:
                w = stream['kms-buf-w'] // div
                if w % 2 == 0:
                    break

            h = stream['kms-buf-h'] * div

            stream['kms-buf-w'] = w
            stream['kms-buf-h'] = h

        if stream['display']:
            max_w = mode.hdisplay // (1 if num_planes == 1 else 2)
            max_h = mode.vdisplay // (1 if num_planes <= 2 else 2)

            stream['kms-src-w'] = min(stream['kms-buf-w'], max_w)
            stream['kms-src-h'] = min(stream['kms-buf-h'], max_h)
            stream['kms-src-x'] = (stream['kms-buf-w'] - stream['kms-src-w']) // 2
            stream['kms-src-y'] = (stream['kms-buf-h'] - stream['kms-src-h']) // 2

            stream['kms-dst-w']  =stream['kms-src-w']
            stream['kms-dst-h'] = stream['kms-src-h']

            if display_idx % 2 == 0:
                stream['kms-dst-x'] = 0
            else:
                stream['kms-dst-x'] = mode.hdisplay - stream['kms-dst-w']

            if display_idx // 2 == 0:
                stream['kms-dst-y'] = 0
            else:
                stream['kms-dst-y'] = mode.vdisplay - stream['kms-dst-h']

            display_idx += 1

            plane = res.reserve_generic_plane(crtc, stream['kms-fourcc'])
            assert(plane)
            stream['plane'] = plane

    for stream in streams:
        vd = stream['dev']

        mem_type = v4l2.MemType.DMABUF if ctx.buf_type == 'drm' else v4l2.MemType.MMAP

        if not stream.get('embedded', False):
            cap = vd.get_capture_streamer(mem_type, stream['w'], stream['h'], stream['fourcc'])
        else:
            bpp = embedded_fourcc_to_bytes_per_pixel(stream['fourcc'])
            size = stream['w'] * stream['h'] * bpp // 8
            cap = vd.get_meta_capture_streamer(mem_type, size, stream['fourcc'])

        stream['cap'] = cap


def setup(ctx: Context):
    streams = ctx.streams

    for stream in streams:
        cap = stream['cap']

        if ctx.buf_type == 'drm':
            # Allocate FBs
            fbs = []
            for i in range(stream['num_bufs']):
                fb = kms.DumbFramebuffer(card, stream['kms-buf-w'], stream['kms-buf-h'], stream['kms-fourcc'])
                fbs.append(fb)
            stream['fbs'] = fbs

        if stream['display']:
            assert(ctx.buf_type == 'drm')

            # Set fb0 to screen
            fb = stream['fbs'][0]
            plane = stream['plane']

            plane.set_props({
                'FB_ID': fb.id,
                'CRTC_ID': crtc.id,
                'SRC_X': stream['kms-src-x'] << 16,
                'SRC_Y': stream['kms-src-y'] << 16,
                'SRC_W': stream['kms-src-w'] << 16,
                'SRC_H': stream['kms-src-h'] << 16,
                'CRTC_X': stream['kms-dst-x'],
                'CRTC_Y': stream['kms-dst-y'],
                'CRTC_W': stream['kms-dst-w'],
                'CRTC_H': stream['kms-dst-h'],
            })

            stream['kms_old_fb'] = None
            stream['kms_fb'] = fb
            stream['kms_fb_queue'] = deque()

        if ctx.buf_type == 'drm':
            fds = [fb.fd(0) for fb in stream['fbs']]
            cap.reserve_buffers_dmabuf(fds)
        else:
            cap.reserve_buffers(stream['num_bufs'])

        first_buf = 1 if stream['display'] else 0

        # Queue the rest to the camera
        for i in range(first_buf, stream['num_bufs']):
            if stream['fourcc'] == v4l2.MetaFormat.RPI_FE_CFG:
                pisp_create_config(cap, cap.buffers[i])

            cap.queue(cap.buffers[i])

    if ctx.use_display:
        # Do the initial modeset
        req = kms.AtomicReq(card)
        req.add(conn, 'CRTC_ID', crtc.id)
        req.add(crtc, {'ACTIVE': 1,
                'MODE_ID': modeb.id})

        for stream in streams:
            if 'plane' in stream:
                req.add(stream['plane'], 'FB_ID', stream['kms_fb'].id)

        req.commit_sync(allow_modeset = True)

        if ctx.delay:
            print(f'Waiting for {ctx.delay} seconds')
            time.sleep(ctx.delay)

    for stream in streams:
        print(f'{stream["dev_path"]}: stream on')
        stream['cap'].stream_on()

    for stream in streams:
        stream['total_num_frames'] = 0
        stream['last_framenum'] = 0
        stream['last_timestamp'] = time.perf_counter()

    ctx.kms_committed = False

    if ctx.user_script:
        ctx.updater = ctx.user_script.Updater(subdevices)
    else:
        ctx.updater = None



def readvid(ctx: Context, stream):
    if ctx.updater:
        ctx.updater.update()

    stream['total_num_frames'] += 1

    # With IPython we have separate fps tracking
    if not ctx.use_ipython:
        ts = time.perf_counter()

        diff = ts - stream['last_timestamp']
        num_frames = stream['total_num_frames'] - stream['last_framenum']

        if stream['total_num_frames'] == 1:
            print('{}: first frame in {:.2f} s'
                  .format(stream['dev_path'], diff))

        if diff >= 5:
            print('{}: {} frames in {:.2f} s, {:.2f} fps'
                  .format(stream['dev_path'], num_frames, diff, num_frames / diff))

            stream['last_timestamp'] = ts
            stream['last_framenum'] = stream['total_num_frames']

    cap = stream['cap']
    vbuf = cap.dequeue()

    if ctx.buf_type == 'drm':
        fb = next((fb for fb in stream['fbs'] if fb.fd(0) == vbuf.fd), None)
        assert(fb != None)

    if ctx.save:
        save_fb_to_file(stream, ctx.buf_type == 'drm', fb if ctx.buf_type == 'drm' else vbuf)

    if stream['display']:
        stream['kms_fb_queue'].append(fb)

        if len(stream['kms_fb_queue']) >= stream['num_bufs'] - 1:
            print('WARNING fb_queue {}'.format(len(stream['kms_fb_queue'])))

        #print(f'Buf from {stream['dev_path']}: kms_fb_queue {len(stream['kms_fb_queue'])}, commit ongoing {kms_committed}')

        # XXX with a small delay we might get more planes to the commit
        if ctx.kms_committed == False:
            handle_pageflip(ctx)
    else:
        if ctx.tx and (ctx.tx == ['all'] or str(stream['id']) in ctx.tx):
            ctx.net_tx.tx(stream, vbuf, ctx.buf_type == 'drm')

        cap.queue(vbuf)


def readkey(ctx):
    streams = ctx.streams

    for stream in reversed(streams):
        print(f'{stream["dev_path"]}: stream off')
        stream['cap'].stream_off()
        #time.sleep(0.5)
        #print('DISABLED CAP')
        #time.sleep(1)

    print('Done')
    sys.stdin.readline()
    exit(0)


def handle_pageflip(ctx: Context):
    streams = ctx.streams

    ctx.kms_committed = False

    req = kms.AtomicReq(card)

    do_commit = False

    for stream in streams:
        if not stream['display']:
            continue

        #print(f'Page flip {stream['dev_path']}: kms_fb_queue {len(stream['kms_fb_queue'])}, new_fb {stream['kms_fb']}, old_fb {stream['kms_old_fb']}')

        cap = stream['cap']

        if stream['kms_old_fb']:
            assert(ctx.buf_type == 'drm')

            fb = stream['kms_old_fb']

            # XXX we should just track the vbufs in streams, instead of looking
            # for the vbuf based on the drm fb
            vbuf = next(vbuf for vbuf in cap.buffers if vbuf.fd == fb.fd(0))

            cap.queue(vbuf)
            stream['kms_old_fb'] = None

        if len(stream['kms_fb_queue']) == 0:
            continue

        stream['kms_old_fb'] = stream['kms_fb']

        fb = stream['kms_fb_queue'].popleft()
        stream['kms_fb'] = fb

        plane = stream['plane']

        req.add(plane, 'FB_ID', fb.id)

        do_commit = True

    if do_commit:
        req.commit(allow_modeset = False)
        ctx.kms_committed = True


def readdrm(ctx: Context):
    #print('EVENT');
    for ev in card.read_events():
        if ev.type == kms.DrmEventType.FLIP_COMPLETE:
            handle_pageflip(ctx)


def run(ctx: Context):
    sel = selectors.DefaultSelector()

    if not ctx.use_ipython:
        sel.register(sys.stdin, selectors.EVENT_READ, lambda: readkey(ctx))
    if ctx.use_display:
        sel.register(card.fd, selectors.EVENT_READ, lambda: readdrm(ctx))
    for stream in ctx.streams:
        sel.register(stream['cap'].fd, selectors.EVENT_READ, lambda data=stream: readvid(ctx, data))

    if not ctx.use_ipython:
        while True:
            events = sel.select()
            for key, _ in events:
                callback = key.data
                callback()
        sys.exit(0)
    else:
        ctx.run_ipython(ctx, sel)


if __name__ == "__main__":
    ctx = Context()

    init_setup(ctx)
    init_viddevs(ctx)
    init_kms(ctx)

    if ctx.print_config:
        for stream in ctx.streams:
            pprint.pprint(stream)

    if ctx.config_only:
        sys.exit(0)

    setup(ctx)
    run(ctx)
