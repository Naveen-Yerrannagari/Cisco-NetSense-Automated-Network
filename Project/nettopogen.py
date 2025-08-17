#!/usr/bin/env python3
"""
Cisco VIP 2025 Network Tool - merged script with .pkt-like export and extended Day-1/Day-2 simulation
Usage:
    python network_tool.py --configs ./configs --switches ./switches --draw --analyze --simulate-day1 --simulate-day2 --export-pkt
"""

import argparse
import ipaddress
import os
import re
import threading
import time
import json
import zipfile
import queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx

# -------------------------
# Utility & Data Structures
# -------------------------
@dataclass
class Interface:
    name: str
    ip: Optional[str] = None
    mask: Optional[str] = None
    mtu: Optional[int] = None
    bandwidth_kbps: Optional[int] = None
    description: Optional[str] = None
    vlan: Optional[int] = None  # for subinterfaces encapsulation dot1Q
    is_subif: bool = False

@dataclass
class Device:
    name: str
    dtype: str  # 'router' | 'switch' | 'host'
    interfaces: Dict[str, Interface] = field(default_factory=dict)

@dataclass
class Link:
    a: str
    b: str
    bandwidth_mbps: float = 100.0
    mtu: Optional[int] = None
    label: Optional[str] = None

# -------------------------
# Parsing
# -------------------------
HOSTNAME_RE = re.compile(r"^hostname\s+(\S+)", re.MULTILINE)
# Capture interface blocks including subinterfaces like GigabitEthernet0/0.10
INTF_BLOCK_RE = re.compile(r"^interface\s+([A-Za-z]+[A-Za-z0-9/\.]+)\s*\n(.*?)(?=^\S|\Z)", re.MULTILINE | re.DOTALL)
IP_RE = re.compile(r"ip address\s+(\d+\.\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+\.\d+)")
BW_RE = re.compile(r"bandwidth\s+(\d+)")  # in kbps on Cisco
MTU_RE = re.compile(r"mtu\s+(\d+)")
DESC_RE = re.compile(r"description\s+(.+)")
ENCAP_DOT1Q_RE = re.compile(r"encapsulation\s+dot1Q\s+(\d+)", re.IGNORECASE)

def parse_router_config(text: str) -> Device:
    hostname = HOSTNAME_RE.search(text)
    name = hostname.group(1) if hostname else "UNKNOWN"
    dev = Device(name=name, dtype="router", interfaces={})
    for m in INTF_BLOCK_RE.finditer(text):
        iname = m.group(1).strip()
        body = m.group(2)
        iface = Interface(name=iname)
        ipm = IP_RE.search(body)
        if ipm:
            iface.ip, iface.mask = ipm.group(1), ipm.group(2)
        bwm = BW_RE.search(body)
        if bwm:
            iface.bandwidth_kbps = int(bwm.group(1))
        mtum = MTU_RE.search(body)
        if mtum:
            iface.mtu = int(mtum.group(1))
        descm = DESC_RE.search(body)
        if descm:
            iface.description = descm.group(1).strip()
        encm = ENCAP_DOT1Q_RE.search(body)
        if encm:
            try:
                iface.vlan = int(encm.group(1))
                iface.is_subif = True
            except Exception:
                pass
        # Defaults if bandwidth not set
        if iface.bandwidth_kbps is None:
            lname = iname.lower()
            if lname.startswith("gigabitethernet"):
                iface.bandwidth_kbps = 100000  # 100 Mbps default if unset
            elif lname.startswith("fastethernet"):
                iface.bandwidth_kbps = 10000   # 10 Mbps
            elif lname.startswith("serial"):
                iface.bandwidth_kbps = 1544    # ~T1
            else:
                iface.bandwidth_kbps = 100000
        dev.interfaces[iname] = iface
    return dev

def load_devices(config_dir="configs", switch_dir="switches") -> Dict[str, Device]:
    devices: Dict[str, Device] = {}
    if os.path.isdir(config_dir):
        for f in os.listdir(config_dir):
            if f.lower().endswith(".txt") or f.lower().endswith(".cfg") or f.lower().endswith(".dump"):
                with open(os.path.join(config_dir, f), "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                d = parse_router_config(text)
                # Avoid overwriting switch placeholder - routers take precedence if same name
                devices[d.name] = d
    # Switch configs (optional) - very light parsing for hostname only
    if os.path.isdir(switch_dir):
        for f in os.listdir(switch_dir):
            if f.lower().endswith(".txt") or f.lower().endswith(".cfg"):
                with open(os.path.join(switch_dir, f), "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                hostname = HOSTNAME_RE.search(text)
                name = hostname.group(1) if hostname else f.replace(".txt", "").replace(".cfg", "")
                # only add if not already present (don't override routers)
                if name not in devices:
                    devices[name] = Device(name=name, dtype="switch", interfaces={})
    return devices

# -------------------------
# Topology Inference
# -------------------------
def ip_to_network(ip: str, mask: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(f"{ip}/{mask}", strict=False)

def build_graph(devices: Dict[str, Device], links_csv: Optional[str] = "links.csv") -> Tuple[nx.Graph, List[Link]]:
    G = nx.Graph()
    for name, dev in devices.items():
        G.add_node(name, dtype=dev.dtype)
    links: List[Link] = []
    if links_csv and os.path.exists(links_csv):
        import csv
        with open(links_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                links.append(Link(
                    a=row["endpointA"].strip(),
                    b=row["endpointB"].strip(),
                    bandwidth_mbps=float(row.get("bandwidth_mbps", 100)),
                    mtu=int(row["mtu"]) if row.get("mtu") else None,
                    label=row.get("label") or None
                ))
    else:
        subnet_map = defaultdict(list)  # network -> [(device, interface)]
        for d in devices.values():
            for iface in d.interfaces.values():
                if iface.ip and iface.mask:
                    try:
                        net = ip_to_network(iface.ip, iface.mask)
                        subnet_map[str(net)].append((d.name, iface))
                    except Exception:
                        pass
        # For each subnet: if exactly 2 router interfaces -> link them. If >2, assume LAN/switch.
        for net, members in subnet_map.items():
            if len(members) == 2:
                (d1, i1), (d2, i2) = members
                bw_mbps = min(i1.bandwidth_kbps, i2.bandwidth_kbps) / 1000.0
                mtu = None
                if i1.mtu and i2.mtu:
                    mtu = min(i1.mtu, i2.mtu)
                links.append(Link(a=d1, b=d2, bandwidth_mbps=bw_mbps, mtu=mtu, label=f"ptp {net}"))
            elif len(members) > 2:
                # create or reuse a pseudo switch node for the LAN/VLAN
                sw_node = f"SW_{net.replace('/', '_')}"
                if sw_node not in G.nodes:
                    G.add_node(sw_node, dtype="switch")
                for (dname, iface) in members:
                    bw_mbps = iface.bandwidth_kbps / 1000.0
                    links.append(Link(a=dname, b=sw_node, bandwidth_mbps=bw_mbps, mtu=iface.mtu, label=f"lan {net}"))
    # Add edges
    for lk in links:
        G.add_edge(lk.a, lk.b, bandwidth_mbps=lk.bandwidth_mbps, mtu=lk.mtu, label=lk.label)
    return G, links

# -------------------------
# Validations
# -------------------------
def validate(devices: Dict[str, Device], G: nx.Graph, links: List[Link], endpoints_csv: Optional[str] = "endpoints.csv") -> str:
    report = []
    report.append("= VALIDATION REPORT =")
    # Duplicate IPs & overlapping subnets
    ip_map = {}
    dup_ips = []
    subnets = []
    for d in devices.values():
        for iface in d.interfaces.values():
            if iface.ip and iface.mask:
                key = iface.ip
                if key in ip_map:
                    dup_ips.append((key, ip_map[key], (d.name, iface.name)))
                else:
                    ip_map[key] = (d.name, iface.name)
                try:
                    subnets.append((d.name, iface.name, ip_to_network(iface.ip, iface.mask)))
                except Exception:
                    pass
    if dup_ips:
        report.append("⚠ Duplicate IP addresses found:")
        for ip, a, b in dup_ips:
            report.append(f"  - {ip} used by {a} and {b}")
    else:
        report.append("✅ No duplicate IPs detected.")
    # Overlapping subnets (basic pairwise check)
    overlaps = []
    for i in range(len(subnets)):
        for j in range(i+1, len(subnets)):
            n1 = subnets[i][2]
            n2 = subnets[j][2]
            if n1.overlaps(n2) and n1 != n2:
                overlaps.append((subnets[i][:2], subnets[j][:2], str(n1), str(n2)))
    if overlaps:
        report.append("⚠ Overlapping subnets detected:")
        for a, b, n1, n2 in overlaps:
            report.append(f"  - {a}({n1}) overlaps {b}({n2})")
    else:
        report.append("✅ No overlapping subnets detected.")
    # Heuristic: Non-standard gateway (router LAN IP not .1)
    for d in devices.values():
        for iface in d.interfaces.values():
            if iface.ip and iface.mask:
                try:
                    net = ip_to_network(iface.ip, iface.mask)
                except Exception:
                    continue
                host_part = int(str(iface.ip).split(".")[-1])
                if net.prefixlen >= 24:  # simple heuristic for /24 or smaller
                    if host_part != 1:
                        report.append(f"ℹ Gateway heuristic: {d.name} {iface.name} has IP {iface.ip} on {net}. Not .1 — check default gateway settings on hosts.")
    # Subinterface/VLAN labeling vs encapsulation
    for d in devices.values():
        for iface in d.interfaces.values():
            if iface.is_subif and iface.vlan:
                if not (iface.description and str(iface.vlan) in iface.description):
                    report.append(f"⚠ {d.name} {iface.name}: encapsulation dot1Q {iface.vlan} but description doesn't mention VLAN {iface.vlan}. Consider labeling correctly.")
            if iface.is_subif and iface.ip and not iface.vlan:
                report.append(f"⚠ {d.name} {iface.name}: subinterface has IP but no 'encapsulation dot1Q' detected.")
    # Potential loops (graph cycles)
    try:
        if not nx.is_tree(G):
            cycles = list(nx.cycle_basis(G))
            if cycles:
                report.append("⚠ Potential loops (cycles) detected in topology:")
                for cyc in cycles[:10]:
                    report.append(f"  - Cycle: {' -> '.join(cyc)}")
            else:
                report.append("ℹ Graph is not a tree, but no simple cycles found.")
        else:
            report.append("✅ No cycles; topology is a tree.")
    except Exception:
        report.append("ℹ Could not determine cycles (graph analysis error).")
    # Missing switch configs (inferred L2)
    for n, attrs in G.nodes(data=True):
        if attrs.get("dtype") == "switch":
            if n.startswith("SW_"):
                report.append(f"⚠ Missing switch configuration for inferred L2 node: {n}. Provide switch config in switches/ to validate VLANs, STP, etc.")
    # Optional PC endpoint sanity checks
    if endpoints_csv and os.path.exists(endpoints_csv):
        import csv
        with open(endpoints_csv, newline="") as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                ip = row.get("host_ip")
                gw = row.get("gateway_ip")
                if ip and gw:
                    try:
                        subnet = row.get("subnet") or f"{ip}/24"
                        net = ipaddress.ip_network(subnet, strict=False)
                        if ipaddress.ip_address(gw) not in net:
                            report.append(f"⚠ Endpoint {row.get('host_name')} gateway {gw} not in same subnet as {ip}.")
                    except Exception:
                        pass
    return "\n".join(report)

# -------------------------
# Load & Capacity Analysis
# -------------------------
def load_balancing_recommendations(G, traffic_demands=None):
    """
    Analyze network links and traffic to detect overloaded links or unreachable paths.
    Args:
        G : networkx.Graph - topology graph
        traffic_demands : list of tuples (src, dst, demand_mbps)
    Returns:
        recommendations : list of strings
    """
    recommendations = []
    traffic_demands = traffic_demands or [
        ("R1", "Switch1", 194),
        ("R1", "R2", 550),
        ("R2", "Switch2", 152),
        ("R3", "R2", 761),
        ("Switch2", "R3", 465),
        ("Switch1", "Switch2", 231),
    ]

    for src, dst, demand in traffic_demands:
        if src not in G.nodes or dst not in G.nodes:
            recommendations.append(f"❌ No path between {src} and {dst}, demand {demand} Mbps dropped.")
            continue
        try:
            path = nx.shortest_path(G, src, dst)
            # Check if any link along path is overloaded
            for u, v in zip(path[:-1], path[1:]):
                capacity = G[u][v].get("bandwidth", 100)
                if demand > capacity:
                    recommendations.append(f"🔥 Link {u}–{v} is OVERLOADED ({demand}/{capacity} Mbps). Add capacity or reroute flows.")
        except nx.NetworkXNoPath:
            recommendations.append(f"❌ No path between {src} and {dst}, demand {demand} Mbps dropped.")

    return recommendations
def load_analysis(G: nx.Graph, links: List[Link], traffic_csv: Optional[str]) -> str:
    report = []
    report.append("= LOAD ANALYSIS =")
    if not (traffic_csv and os.path.exists(traffic_csv)):
        report.append("ℹ No traffic_demands.csv provided or file not found. Skipping load computation.")
        return "\n".join(report)
    import csv

    def resolve_subnet_node(subnet_str: str) -> Optional[str]:
        if subnet_str in G.nodes:
            return subnet_str
        sw_guess = f"SW_{subnet_str}"
        if sw_guess in G.nodes:
            return sw_guess
        return None

    # Precompute edge capacities
    edge_capacity = {}
    for (u, v, data) in G.edges(data=True):
        cap = data.get("bandwidth_mbps", 100.0)
        edge_capacity[(u, v)] = cap
        edge_capacity[(v, u)] = cap

    edge_load = defaultdict(float)
    rows = []
    with open(traffic_csv, newline="") as fh:
        rdr = csv.DictReader(fh)
        headers = rdr.fieldnames or []
        # Check required headers
        if not all(k in headers for k in ("src_subnet", "dst_subnet", "mbps")):
            report.append("⚠ traffic_demands.csv missing required headers: 'src_subnet', 'dst_subnet', 'mbps'")
            return "\n".join(report)
        for row in rdr:
            rows.append(row)

    for row in rows:
        try:
            src_subnet = row.get("src_subnet")
            dst_subnet = row.get("dst_subnet")
            mbps_str = row.get("mbps", "0")
            if not src_subnet or not dst_subnet:
                report.append(f"⚠ Skipping incomplete demand row: {row}")
                continue
            src = resolve_subnet_node(src_subnet.strip())
            dst = resolve_subnet_node(dst_subnet.strip())
            mbps = float(mbps_str)
            if src is None or dst is None or not G.has_node(src) or not G.has_node(dst):
                report.append(f"ℹ Skipping demand {row} (unresolved nodes). Provide SW_{src_subnet} or links.csv with your switch names.")
                continue
            path = nx.shortest_path(G, src, dst, weight=None)
        except (ValueError, KeyError) as e:
            report.append(f"⚠ Error parsing demand row {row}: {e}")
            continue
        except nx.NetworkXNoPath:
            report.append(f"⚠ No path between {src} and {dst}.")
            continue

        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_load[(u, v)] += mbps
            edge_load[(v, u)] += mbps

    overloads = []
    for (u, v), load in edge_load.items():
        cap = edge_capacity.get((u, v), 100.0)
        if load > cap:
            overloads.append((u, v, load, cap))

    if not overloads:
        report.append("✅ No overloaded links for provided demands.")
    else:
        report.append("⚠ Overloaded links:")
        for u, v, load, cap in overloads:
            report.append(f"  - {u} <-> {v}: load {load:.2f} Mbps > capacity {cap:.2f} Mbps")
            # Try alternate path suggestion
            G2 = G.copy()
            if G2.has_edge(u, v):
                G2.remove_edge(u, v)
                try:
                    alt_path = nx.shortest_path(G2, u, v)
                    report.append(f"    Suggestion: reroute via: {' -> '.join(alt_path)}")
                except nx.NetworkXNoPath:
                    report.append("    No alternate path; consider capacity upgrade or redundancy.")

    try:
        diam = nx.diameter(G)
    except Exception:
        diam = 0

    if diam >= 4:
        report.append("ℹ Topology diameter is high; consider BGP for inter-domain scaling; OSPF is fine intra-domain.")
    else:
        report.append("ℹ OSPF appears adequate; consider BGP if connecting to external AS or policy-heavy paths.")

    return "\n".join(report)

# -------------------------
# Drawing
# -------------------------
def draw_topology(G: nx.Graph, out_png="output/network_topology.png"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=7)
    node_colors = []
    for n, attrs in G.nodes(data=True):
        if attrs.get("dtype") == "router":
            node_colors.append("lightblue")
        elif attrs.get("dtype") == "switch":
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightgray")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
    nx.draw_networkx_edges(G, pos)
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    edge_labels = {(u, v): f"{data.get('bandwidth_mbps','?')}Mbps" for (u, v, data) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Network Topology")
    plt.axis('off')
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# -------------------------
# Export / "pkt-like" archive generation helpers
# -------------------------
def generate_startup_config(dev: Device) -> str:
    """Create a Cisco-style startup-config text for a Device."""
    lines = []
    lines.append(f"hostname {dev.name}")
    lines.append("no ip http server")
    lines.append("no ip http secure-server")
    lines.append("!")
    # Interfaces
    for ifname, iface in dev.interfaces.items():
        lines.append(f"interface {ifname}")
        if iface.description:
            lines.append(f" description {iface.description}")
        if iface.is_subif:
            if iface.vlan:
                lines.append(f" encapsulation dot1Q {iface.vlan}")
        if iface.ip and iface.mask:
            lines.append(f" ip address {iface.ip} {iface.mask}")
        if iface.mtu:
            lines.append(f" mtu {iface.mtu}")
        if iface.bandwidth_kbps:
            lines.append(f" bandwidth {iface.bandwidth_kbps}")
        lines.append(" no shutdown")
        lines.append("!")
    lines.append("ip routing")
    return "\n".join(lines) + "\n"

def export_pkt_like_archive(devices: Dict[str, Device], G: nx.Graph, links: List[Link],
                            out_file: str = "output/network_project.pkt"):
    """
    Create a ZIP archive with .pkt extension containing:
      - startup configs per device (<hostname>_startup.cfg)
      - topology.json
      - links.csv
      - generated topology image (if available)
      - README_import_instructions.txt
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    topo_png = "output/network_topology.png"
    if not os.path.exists(topo_png):
        try:
            draw_topology(G, out_png=topo_png)
        except Exception:
            topo_png = None
    topo = {"nodes": [], "links": []}
    for n, data in G.nodes(data=True):
        topo["nodes"].append({
            "name": n,
            "dtype": data.get("dtype", "unknown"),
            "attrs": {k: v for k, v in data.items() if k != "dtype"}
        })
    for u, v, d in G.edges(data=True):
        topo["links"].append({
            "a": u, "b": v,
            "bandwidth_mbps": float(d.get("bandwidth_mbps", 0)),
            "mtu": d.get("mtu"),
            "label": d.get("label")
        })
    with zipfile.ZipFile(out_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # startup configs
        for name, dev in devices.items():
            cfg = generate_startup_config(dev)
            zf.writestr(f"configs/{name}_startup.cfg", cfg)
        # topology json
        zf.writestr("topology/topology.json", json.dumps(topo, indent=2))
        # links.csv
        csv_lines = ["endpointA,endpointB,bandwidth_mbps,mtu,label"]
        for lk in links:
            csv_lines.append(f"{lk.a},{lk.b},{lk.bandwidth_mbps},{lk.mtu or ''},{lk.label or ''}")
        zf.writestr("topology/links.csv", "\n".join(csv_lines))
        # include inputs if exist
        for fname in ("endpoints.csv", "traffic_demands.csv", "links.csv"):
            if os.path.exists(fname):
                with open(fname, "rb") as fh:
                    zf.writestr(f"inputs/{os.path.basename(fname)}", fh.read())
        # topology image
        if topo_png and os.path.exists(topo_png):
            zf.write(topo_png, arcname="topology/network_topology.png")
        # README
        readme = """
This archive is a convenience package containing device startup-configs and topology data.
How to use:
1) For Packet Tracer:
   - Open Packet Tracer and create devices matching the hostnames listed in configs/.
   - For each device, open the CLI and paste the contents of configs/<hostname>_startup.cfg
     (or use the device's 'import config' feature if available).
   - Connect interfaces as described in topology/topology.json or topology/links.csv.
2) For GNS3 / automation:
   - Use topology/topology.json and configs/* to programmatically build the lab or
     convert configs into templates.
Notes:
- This .pkt file is a ZIP-style archive for convenience. It is NOT a native Packet Tracer file.
- If you want an actual Packet Tracer activity file, please use Packet Tracer GUI to build the topology,
  paste the startup configs into each device, then save the activity as a .pkt from Packet Tracer.
"""
        zf.writestr("README_import_instructions.txt", readme.strip())
    print(f"Exported archive to {out_file} (ZIP with .pkt extension).")

# -------------------------------
# Router Simulator Class (Day-1/Day-2 Simulation)
# -------------------------------
class RouterSimulator(threading.Thread):
    def __init__(self, name, graph, queues):
        super().__init__()
        self.name = name
        self.graph = graph
        self.queues = queues
        self.routing_table = {}
        self.mac_table = {}
        self.events = []
        self.running = True
        self.paused = False
        self.failed = False
        self.failed_links = set()

    def run(self):
        while self.running:
            if self.paused or self.failed:
                time.sleep(0.1)
                continue
            try:
                msg = self.queues[self.name].get(timeout=0.1)
                src = msg.get("src")
                if (self.name, src) not in self.failed_links and (src, self.name) not in self.failed_links:
                    self.process_message(msg)
            except queue.Empty:
                pass
            self.simulate_arp()
            self.simulate_ospf()
            time.sleep(0.1)

    def process_message(self, msg):
        self.events.append(f"[{time.time():.1f}] Received {msg}")
        if msg.get("type") == "ARP":
            src_mac = msg.get("src_mac")
            if src_mac:
                self.mac_table[src_mac] = msg["src"]

    def simulate_arp(self):
        for neighbor in self.graph.neighbors(self.name):
            if (self.name, neighbor) not in self.failed_links:
                self.queues[neighbor].put({"type": "ARP", "src": self.name, "src_mac": f"MAC_{self.name}"})
                self.events.append(f"[{time.time():.1f}] Sent ARP to {neighbor}")

    def simulate_ospf(self):
        for neighbor in self.graph.neighbors(self.name):
            if (self.name, neighbor) not in self.failed_links:
                self.routing_table[neighbor] = neighbor
                self.events.append(f"[{time.time():.1f}] OSPF adjacency with {neighbor}")

    def pause(self):
        self.paused = True
        self.events.append(f"[{time.time():.1f}] PAUSED")

    def resume(self):
        self.paused = False
        self.events.append(f"[{time.time():.1f}] RESUMED")

    def stop(self):
        self.running = False
        self.events.append(f"[{time.time():.1f}] STOPPED")

    def fail_device(self):
        self.failed = True
        self.events.append(f"[{time.time():.1f}] DEVICE FAILURE")

    def recover_device(self):
        self.failed = False
        self.events.append(f"[{time.time():.1f}] DEVICE RECOVERED")

    def fail_link(self, neighbor):
        self.failed_links.add((self.name, neighbor))
        self.events.append(f"[{time.time():.1f}] LINK {self.name}-{neighbor} DOWN")

    def recover_link(self, neighbor):
        if (self.name, neighbor) in self.failed_links:
            self.failed_links.remove((self.name, neighbor))
        self.events.append(f"[{time.time():.1f}] LINK {self.name}-{neighbor} UP")

# -------------------------------
# Demo Topology Builder (Fallback)
# -------------------------------
def build_demo_topology():
    G = nx.Graph()
    nodes = ["R1", "R2", "R3", "Switch1", "Switch2"]
    for n in nodes:
        G.add_node(n)
    G.add_edge("R1", "R2", bandwidth=100)
    G.add_edge("R2", "R3", bandwidth=100)
    G.add_edge("R1", "Switch1", bandwidth=100)
    G.add_edge("R2", "Switch2", bandwidth=100)
    G.add_edge("R3", "Switch2", bandwidth=100)
    return G

# -------------------------------
# Day-1 Simulation
# -------------------------------
def run_day1_simulation(G, duration=10.0):
    print("=== Running Day-1 Simulation ===")
    queues = {node: queue.Queue() for node in G.nodes}
    routers = []
    for node in G.nodes:
        r = RouterSimulator(node, G, queues)
        r.start()
        routers.append(r)

    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.5)

    # Demo pause/resume first router
    first_router = routers[0]
    print("\nPausing first router for 2 seconds...")
    first_router.pause()
    time.sleep(2)
    first_router.resume()

    # Stop routers
    for r in routers:
        r.stop()
    for r in routers:
        r.join()

    os.makedirs("output", exist_ok=True)
    for r in routers:
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', r.name)
        with open(f"output/{safe_name}_day1_log.txt", "w") as fh:
            fh.write("\n".join(r.events))

    print("Day-1 simulation finished. Logs saved in 'output/' folder.")
# -------------------------------
# Day-2 Simulation
# -------------------------------

def run_day2_simulation(G, duration=15.0, fault_events=None):
    print("=== Running Day-2 Simulation with Fault Injection ===")
    queues = {node: queue.Queue() for node in G.nodes}
    routers = {node: RouterSimulator(node, G, queues) for node in G.nodes}

    for r in routers.values():
        r.start()

    fault_events = fault_events or []
    start_time = time.time()

    while time.time() - start_time < duration:
        t = time.time() - start_time
        for event in fault_events:
            event_time, event_type, target, neighbor = event
            if abs(t - event_time) < 0.2:
                if event_type == "device":
                    routers[target].fail_device()
                elif event_type == "link" and neighbor:
                    routers[target].fail_link(neighbor)
        time.sleep(0.1)

    # Recovery phase
    for r in routers.values():
        r.recover_device()
        for neighbor in G.neighbors(r.name):
            r.recover_link(neighbor)

    for r in routers.values():
        r.stop()
    for r in routers.values():
        r.join()

    os.makedirs("output", exist_ok=True)
    for r in routers.values():
        safe_name = re.sub(r'[^A-Za-z0-9_]', '_', r.name)
        with open(f"output/{safe_name}_day2_log.txt", "w") as fh:
            fh.write("\n".join(r.events))

    print("Day-2 simulation finished. Logs saved in 'output/' folder.")

# -------------------------
# Detect Missing Components
# -------------------------
def detect_missing_components(config_folder, expected_devices):
    missing = []
    if not os.path.isdir(config_folder):
        return expected_devices[:]
    available_files = os.listdir(config_folder)
    available_devices = [os.path.splitext(f)[0] for f in available_files]
    for device in expected_devices:
        if device not in available_devices:
            missing.append(device)
    return missing

# -------------------------
# CLI and Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Cisco VIP 2025 Network Tool")
    ap.add_argument("--configs", default="configs", help="Router configs folder")
    ap.add_argument("--switches", default="switches", help="Switch configs folder (optional)")
    ap.add_argument("--links", default="links.csv", help="links.csv (optional)")
    ap.add_argument("--endpoints", default="endpoints.csv", help="endpoints.csv (optional)")
    ap.add_argument("--traffic", default="traffic_demands.csv", help="traffic_demands.csv (optional)")
    
    ap.add_argument("--draw", action="store_true", help="Draw topology PNG")
    ap.add_argument("--analyze", action="store_true", help="Run validations and load analysis")

    ap.add_argument("--export-pkt", action="store_true", dest="export_pkt",
                    help="Export a .pkt-like ZIP archive with configs + topology JSON")

    ap.add_argument("--recommend-lb", action="store_true", dest="recommend_lb",
                    help="Print Load Balancing Recommendations")

    ap.add_argument("--check-missing", action="store_true", help="Check for missing components")

    ap.add_argument("--simulate-day1", action="store_true", help="Run Day-1 router simulation")
    ap.add_argument("--simulate-day2", action="store_true", help="Run Day-2 simulation with fault injection")
    
    args = ap.parse_args()

    devices = load_devices(args.configs, args.switches)

    if not devices:
        print("No devices parsed from configs. Using demo topology and nodes.")
        G = build_demo_topology()
    else:
        G, links = build_graph(devices, args.links)

    if args.draw:
        draw_topology(G)
        print("Topology image saved to output/network_topology.png")

    if args.analyze and devices:
        os.makedirs("output", exist_ok=True)
        vr = validate(devices, G, links, args.endpoints)
        with open("output/validation_report.txt", "w", encoding="utf-8") as fh:
            fh.write(vr)
        la = load_analysis(G, links, args.traffic)
        with open("output/load_analysis.txt", "w", encoding="utf-8") as fh:
            fh.write(la)
        print("Validation and load analysis written to output/*.txt")

    if args.export_pkt and devices:
        export_pkt_like_archive(devices, G, links, out_file="output/network_project.pkt")
        print("Exported Packet Tracer-like archive to output/network_project.pkt")

    if args.recommend_lb:
        recs = load_balancing_recommendations(G)
        print("\n=== Load Balancing Recommendations ===")
        if recs:
            for r in recs:
                print(r)
        else:
            print("No load balancing issues detected.")

    if args.check_missing:
        expected = ["R1", "R2", "R3", "RouterC", "Switch1", "Switch2"]
        missing = detect_missing_components(args.configs, expected)
        if missing:
            print("\n= Missing Components Detected =")
            for dev in missing:
                print(f"⚠️ Config for {dev} is missing!")
            os.makedirs("output", exist_ok=True)
            with open("output/missing_components.txt", "w", encoding="utf-8") as fh:
                for dev in missing:
                    fh.write(f"{dev}\n")
        else:
            print("\nAll expected device configs are present.")

    if args.simulate_day1:
        if 'G' not in locals() or not G.nodes:
            G = build_demo_topology()
        run_day1_simulation(G, duration=10)

    if args.simulate_day2:
        if 'G' not in locals() or not G.nodes:
            G = build_demo_topology()
        # Example fault events: (time_offset_sec, type, target, neighbor)
        faults = [
            (3, "device", "R2", None),
            (5, "link", "R1", "Switch1"),
        ]
        run_day2_simulation(G, duration=15, fault_events=faults)
    
if __name__ == "__main__":
    main()
