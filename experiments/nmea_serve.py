#!/usr/bin/env python3
"""nmea_serve: the chartplotter interface, standalone.

Serves NMEA 0183 sentences on TCP (default :10110, the de-facto
OpenCPN port) from either source:

    python3 nmea_serve.py --replay out/e5b.nmea [--rate 2.0]
        replay a recorded .nmea log (e.g. from e5b_live.py), one
        position pair per `rate` seconds — demo / bench mode

    python3 nmea_serve.py --live
        run the SkyNav passage simulation live (same loop as the
        station's Underway tab) and stream its sentences

To a navigator this makes the whole skyline stack look like a GPS
receiver: OpenCPN -> Options -> Connections -> Network TCP <host>
10110. Fix quality drops to 6 (estimated/DR) whenever a skyline-fix
attempt is INCONCLUSIVE, exactly like a receiver losing satellites.
"""

import argparse
import os
import socket
import sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLIENTS = []


def serve(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', port))
    srv.listen(8)
    print(f'NMEA server on tcp/{port}', flush=True)
    while True:
        c, addr = srv.accept()
        print('client', addr, flush=True)
        CLIENTS.append(c)


def broadcast(line):
    print(line, flush=True)
    dead = []
    for c in CLIENTS:
        try:
            c.sendall((line + '\r\n').encode())
        except OSError:
            dead.append(c)
    for c in dead:
        CLIENTS.remove(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=10110)
    ap.add_argument('--replay', help='.nmea log to replay')
    ap.add_argument('--live', action='store_true',
                    help='run the passage simulation live')
    ap.add_argument('--rate', type=float, default=2.0,
                    help='seconds per GGA/RMC pair in replay mode')
    a = ap.parse_args()
    threading.Thread(target=serve, args=(a.port,), daemon=True).start()
    time.sleep(0.3)
    if a.replay:
        lines = [ln.strip() for ln in open(a.replay) if ln.strip()]
        for i in range(0, len(lines), 2):
            for ln in lines[i:i + 2]:
                broadcast(ln)
            time.sleep(a.rate)
    elif a.live:
        import station
        station.nmea_broadcast = broadcast   # reroute into this server
        station.passage_thread(leg_seconds=a.rate)
    else:
        ap.error('pick --replay FILE or --live')


if __name__ == '__main__':
    main()
