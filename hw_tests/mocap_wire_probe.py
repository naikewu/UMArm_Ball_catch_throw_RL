"""Diagnostic below the SDK -- is anything arriving on the wire at all?

The census printed zero frames while ``NatNetClient.run()`` returned true, and
those two facts are compatible with three quite different faults: Motive is not
streaming, the multicast join attached to the wrong interface, or the SDK
received bytes and failed to parse them.  A raw socket separates them, since it
counts datagrams without interpreting one byte of their contents.

Three listeners are tried in turn, all read-only:

* the multicast data group 239.255.42.99:1511 joined on the named interface;
* the same port bound unicast, which is where a Motive configured for unicast
  sends;
* the command port 1510, which answers a NAT_PING with a server-info packet and
  therefore proves the host is running Motive rather than merely answering ICMP.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

MULTICAST = "239.255.42.99"
DATA_PORT = 1511
CMD_PORT = 1510
NAT_PING = 0
NAT_PINGRESPONSE = 1


def listen(sock: socket.socket, seconds: float, label: str) -> tuple[int, int]:
    sock.settimeout(0.25)
    end = time.monotonic() + seconds
    n = nbytes = 0
    first = None
    while time.monotonic() < end:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError as exc:
            print("  %s: recv error %r" % (label, exc))
            break
        n += 1
        nbytes += len(data)
        if first is None:
            first = (addr, len(data), data[:4].hex())
    if first:
        print("  %s: %d datagrams, %d bytes; first from %s len=%d head=%s"
              % (label, n, nbytes, first[0], first[1], first[2]))
    else:
        print("  %s: NOTHING (0 datagrams in %.1f s)" % (label, seconds))
    return n, nbytes


def try_multicast(local_ip: str, seconds: float) -> int:
    print("multicast %s:%d joined on %s" % (MULTICAST, DATA_PORT, local_ip))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", DATA_PORT))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(MULTICAST) + socket.inet_aton(local_ip))
        return listen(s, seconds, "mcast")[0]
    finally:
        s.close()


def try_unicast(local_ip: str, seconds: float) -> int:
    print("unicast %s:%d" % (local_ip, DATA_PORT))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((local_ip, DATA_PORT))
        return listen(s, seconds, "unicast")[0]
    except OSError as exc:
        print("  unicast: bind failed %r" % (exc,))
        return 0
    finally:
        s.close()


def try_ping(server: str, local_ip: str, timeout: float) -> bool:
    """NAT_PING on the command port; a reply proves Motive is up and answering."""
    print("NAT_PING -> %s:%d from %s" % (server, CMD_PORT, local_ip))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((local_ip, 0))
        s.settimeout(timeout)
        # NatNet command packet: uint16 id, uint16 size, payload.  The SDK sends
        # its own name plus a four-byte version in the connect payload; a bare
        # ping needs neither.
        payload = b"PythonSample\0" + bytes(256 - 13) + bytes([4, 1, 0, 0])
        pkt = struct.pack("<HH", NAT_PING, len(payload)) + payload
        s.sendto(pkt, (server, CMD_PORT))
        try:
            data, addr = s.recvfrom(65535)
        except socket.timeout:
            print("  ping: NO REPLY in %.1f s" % timeout)
            return False
        msg_id = struct.unpack("<H", data[0:2])[0]
        name = data[4:4 + 256].split(b"\0")[0].decode("utf-8", "replace")
        ver = tuple(data[4 + 256:4 + 260]) if len(data) >= 264 else ()
        nnver = tuple(data[4 + 260:4 + 264]) if len(data) >= 268 else ()
        print("  ping: reply from %s id=%d app=%r app_ver=%s natnet_ver=%s"
              % (addr, msg_id, name, ver, nnver))
        return msg_id == NAT_PINGRESPONSE
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--client", default="192.168.1.120")
    ap.add_argument("--server", default="192.168.1.100")
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()

    ok_ping = try_ping(a.server, a.client, 2.0)
    n_mc = try_multicast(a.client, a.seconds)
    n_uc = 0
    if n_mc == 0:
        n_uc = try_unicast(a.client, a.seconds)

    print()
    print("VERDICT: ping=%s multicast=%d unicast=%d"
          % ("ok" if ok_ping else "no-reply", n_mc, n_uc))
    if n_mc == 0 and n_uc == 0:
        print("  No data datagrams reached this NIC.  Either Motive's streaming"
              " engine is off, it streams a different multicast group/port, or"
              " an inbound rule is dropping UDP %d." % DATA_PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
