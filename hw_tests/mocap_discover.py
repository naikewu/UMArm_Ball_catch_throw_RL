"""Last read-only attempt to find a NatNet server before declaring the stream down.

Two things the narrower probes could not rule out.  First, an **intermittent**
stream: a 4 s listen that sees nothing is weak evidence against a source that
bursts.  Second, a server at a **different address**: the workspace's
192.168.1.100 is a constant lifted from the RS485 repo, and a Motive host that
moved would look exactly like a Motive host that stopped.

So this listens for a full half minute on the standard group and, in parallel,
sends one NAT_PING to the directed broadcast and to every neighbour the ARP
cache knows on the camera network.  A NatNet server answers a ping with a
server-info packet naming itself, so a reply identifies the host without any
guesswork about which of them is running Motive.

192.168.1.10 -- the Kinova controller -- is excluded by address, since this
session must not contact it.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time

MULTICAST = "239.255.42.99"
DATA_PORT = 1511
CMD_PORT = 1510
EXCLUDE = {"192.168.1.10"}


def ping_sweep(client_ip: str, targets: list[str], seconds: float) -> list:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    replies = []
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind((client_ip, 0))
        s.settimeout(0.25)
        payload = b"probe\0" + bytes(250) + bytes([4, 1, 0, 0])
        pkt = struct.pack("<HH", 0, len(payload)) + payload
        end = time.monotonic() + seconds
        next_send = 0.0
        while time.monotonic() < end:
            if time.monotonic() >= next_send:
                for t in targets:
                    if t in EXCLUDE:
                        continue
                    try:
                        s.sendto(pkt, (t, CMD_PORT))
                    except OSError:
                        pass
                next_send = time.monotonic() + 2.0
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            name = data[4:260].split(b"\0")[0].decode("utf-8", "replace")
            replies.append((addr, name, len(data)))
            print("  PING REPLY from %s: app=%r (%d bytes)"
                  % (addr, name, len(data)))
    finally:
        s.close()
    return replies


def long_listen(client_ip: str, seconds: float) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    n = 0
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", DATA_PORT))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(MULTICAST) + socket.inet_aton(client_ip))
        s.settimeout(0.5)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            n += 1
            if n == 1:
                print("  DATA from %s, %d bytes, head %s"
                      % (addr, len(data), data[:8].hex()))
    finally:
        s.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", default="192.168.1.120")
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()

    targets = ["192.168.1.255", "192.168.1.100"]
    print("listening %g s on %s:%d and pinging %s"
          % (a.seconds, MULTICAST, DATA_PORT, targets))

    box = {}
    th = threading.Thread(
        target=lambda: box.update(n=long_listen(a.client, a.seconds)),
        daemon=True)
    th.start()
    replies = ping_sweep(a.client, targets, a.seconds)
    th.join(a.seconds + 5.0)

    n = box.get("n", 0)
    print()
    print("RESULT: %d data datagrams in %g s; %d ping replies"
          % (n, a.seconds, len(replies)))
    if n == 0 and not replies:
        print("  No NatNet server answered and no data arrived.  Motive's"
              " streaming engine is not reaching this NIC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
