"""Widest read-only look for a NatNet stream on this NIC.

``mocap_wire_probe.py`` checked the one group and the one port this workspace
expects.  This checks the assumption itself, because a Motive project can be
re-pointed at a different multicast group or data port without anything in a
client repo noticing, and because "nothing arrived" is only evidence of a
misconfigured client if the client was listening where the server was talking.

Two independent passes:

* a **multicast/unicast sweep** -- every plausible NatNet data port bound at
  once, with the standard group joined on every local IPv4 interface, so a join
  that attached to the wrong NIC cannot hide the traffic;
* a **promiscuous IP capture** on the camera-network NIC (``SIO_RCVALL``), which
  needs Administrator and is skipped without it.  It sees every IP packet the
  adapter accepts regardless of port or group, so it is the pass that can say
  "Motive is emitting nothing" rather than "we did not find it".

Nothing is transmitted except one NAT_PING per command port, and nothing is
written.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import socket
import struct
import sys
import time

MULTICAST = "239.255.42.99"
DATA_PORTS = (1511, 1512, 1513)
CMD_PORTS = (1510,)


def local_ipv4s() -> list[str]:
    out = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip not in out:
                out.append(ip)
    except OSError:
        pass
    return out


def sweep(seconds: float, client_ip: str) -> dict:
    """Bind every candidate data port; join the group on every interface."""
    socks = []
    joined: dict[int, list[str]] = {}
    for port in DATA_PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
        except OSError as exc:
            print("  bind :%d failed %r" % (port, exc))
            s.close()
            continue
        ok = []
        for ip in dict.fromkeys([client_ip] + local_ipv4s() + ["0.0.0.0"]):
            try:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(MULTICAST) + socket.inet_aton(ip))
                ok.append(ip)
            except OSError:
                pass
        joined[port] = ok
        s.setblocking(False)
        socks.append((port, s))
    print("  listening on ports %s; group %s joined on %s"
          % ([p for p, _ in socks], MULTICAST, joined))

    counts: dict = collections.Counter()
    sample: dict = {}
    end = time.monotonic() + seconds
    try:
        import select
        while time.monotonic() < end:
            rl, _, _ = select.select([s for _, s in socks], [], [], 0.25)
            for s in rl:
                try:
                    data, addr = s.recvfrom(65535)
                except OSError:
                    continue
                port = s.getsockname()[1]
                key = (addr[0], port)
                counts[key] += 1
                sample.setdefault(key, (len(data), data[:8].hex()))
    finally:
        for _, s in socks:
            s.close()
    return {"counts": dict(counts), "sample": sample}


def promiscuous(client_ip: str, seconds: float) -> dict | None:
    """Every IP packet this NIC accepts.  Needs Administrator; else ``None``."""
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        is_admin = False
    if not is_admin:
        print("  promiscuous capture SKIPPED: not Administrator")
        return None
    SIO_RCVALL = 0x98000001
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
    try:
        s.bind((client_ip, 0))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        s.ioctl(SIO_RCVALL, 1)
        s.settimeout(0.25)
        seen: dict = collections.Counter()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                pkt = s.recvfrom(65535)[0]
            except socket.timeout:
                continue
            if len(pkt) < 20:
                continue
            ihl = (pkt[0] & 0x0F) * 4
            proto = pkt[9]
            src = socket.inet_ntoa(pkt[12:16])
            dst = socket.inet_ntoa(pkt[16:20])
            if proto == 17 and len(pkt) >= ihl + 8:
                sp, dp = struct.unpack("!HH", pkt[ihl:ihl + 4])
                seen[(src, sp, dst, dp)] += 1
            else:
                seen[(src, proto, dst, "non-udp")] += 1
        try:
            s.ioctl(SIO_RCVALL, 0)
        except OSError:
            pass
        return dict(seen)
    except OSError as exc:
        print("  promiscuous capture failed: %r" % (exc,))
        return None
    finally:
        s.close()


def ping_cmd_ports(server: str, client_ip: str) -> None:
    for port in CMD_PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            s.bind((client_ip, 0))
            s.settimeout(1.5)
            payload = b"probe\0" + bytes(250) + bytes([4, 1, 0, 0])
            s.sendto(struct.pack("<HH", 0, len(payload)) + payload,
                     (server, port))
            try:
                data, addr = s.recvfrom(65535)
                print("  cmd :%d replied from %s, %d bytes"
                      % (port, addr, len(data)))
            except socket.timeout:
                print("  cmd :%d no reply" % port)
        finally:
            s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", default="192.168.1.120")
    ap.add_argument("--server", default="192.168.1.100")
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()

    print("PASS 1 -- command-port ping")
    ping_cmd_ports(a.server, a.client)
    print("PASS 2 -- multicast/unicast port sweep")
    res = sweep(a.seconds, a.client)
    if res["counts"]:
        for (src, port), n in sorted(res["counts"].items()):
            ln, head = res["sample"][(src, port)]
            print("  %s -> :%d  %d datagrams (first %d B, head %s)"
                  % (src, port, n, ln, head))
    else:
        print("  nothing on any candidate data port")
    print("PASS 3 -- promiscuous IP capture on %s" % a.client)
    seen = promiscuous(a.client, a.seconds)
    if seen:
        for key, n in sorted(seen.items(), key=lambda kv: -kv[1])[:25]:
            print("  %s  x%d" % (key, n))
    elif seen == {}:
        print("  no IP packets at all on this NIC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
