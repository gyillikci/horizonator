#!/usr/bin/env python3
"""PeakFinder debug viewer: look at a solve the way the camera did.

    python3 skyfix.py PHOTO ... --view-json run.view.json
    python3 pf_debug/serve.py run.view.json [more.view.json ...]

then open http://127.0.0.1:8765 . For every step of the search (the
camera's own claim, each improvement of the running best, the final
fix) the page asks PeakFinder to render the scene from that pose and
draws three things over it: the photograph, the skyline the solver
extracted from it (yellow), and the DEM skyline the solver compared it
against (cyan). PeakFinder is only the backdrop; nothing flows back
into the solve.

PeakFinder's panorama module loads from peakfinder.com in the browser,
so this needs normal internet; without it the page still shows the
photo, both skylines and the cost map. Bound to localhost only.
"""

import argparse
import http.server
import io
import json
import os
import sys
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from assets import celnav  # noqa: E402

RUNS = []          # (name, view dict, resolved photo path or None)
PHOTO_CACHE = {}


def resolve_photo(view):
    """The view file records the absolute path on the machine that
    solved; fall back to the photo collection of this checkout."""
    p = view['photo']['path']
    if os.path.exists(p):
        return p
    for sub in ('peakfinder', 'theodolite', ''):
        q = celnav(sub, view['photo']['file'])
        if os.path.exists(q):
            return q
    return None


def photo_jpeg(i, max_w=1600):
    if i in PHOTO_CACHE:
        return PHOTO_CACHE[i]
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(RUNS[i][2])).convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=88)
    PHOTO_CACHE[i] = buf.getvalue()
    return PHOTO_CACHE[i]


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'index.html'), 'rb') as f:
                return self._send(f.read(), 'text/html; charset=utf-8')
        if path == '/runs':
            return self._send(json.dumps([
                dict(id=i, name=n, photo=v['photo']['file'],
                     status=v['result']['status'], has_photo=p is not None)
                for i, (n, v, p) in enumerate(RUNS)]).encode(),
                'application/json')
        parts = path.strip('/').split('/')
        if len(parts) == 2 and parts[1].isdigit() \
                and int(parts[1]) < len(RUNS):
            i = int(parts[1])
            if parts[0] == 'run':
                return self._send(json.dumps(RUNS[i][1]).encode(),
                                  'application/json')
            if parts[0] == 'photo' and RUNS[i][2]:
                return self._send(photo_jpeg(i), 'image/jpeg')
        self._send(b'not found', 'text/plain', 404)

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('views', nargs='+', help='files from skyfix --view-json')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    for v in a.views:
        with open(v) as f:
            view = json.load(f)
        ph = resolve_photo(view)
        name = os.path.basename(v).replace('.view.json', '')
        RUNS.append((name, view, ph))
        print('%-32s %3d steps  photo: %s' % (
            name, len(view['steps']), ph or 'NOT FOUND (skylines only)'))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    url = 'http://127.0.0.1:%d/' % a.port
    print('serving', url, '(ctrl-c to stop)')
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
