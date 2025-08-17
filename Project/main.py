import argparse
import networkx as nx
import threading
import queue
import time
import os

# -------------------------------
# Router Simulator Class
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
# Build Demo Topology
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

    # Save logs
    os.makedirs("output", exist_ok=True)
    for r in routers:
        with open(f"output/{r.name}_day1_log.txt", "w") as fh:
            fh.write("\n".join(r.events))

    print("Day-1 simulation finished. Logs saved in 'output/' folder.")


# -------------------------------
# Day-2 Simulation with Fault Injection
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

    # Stop routers
    for r in routers.values():
        r.stop()
    for r in routers.values():
        r.join()

    # Save logs
    os.makedirs("output", exist_ok=True)
    for r in routers.values():
        with open(f"output/{r.name}_day2_log.txt", "w") as fh:
            fh.write("\n".join(r.events))

    print("Day-2 simulation finished. Logs saved in 'output/' folder.")


# -------------------------------
# Main Function
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Network Simulation Tool")
    ap.add_argument("--simulate", action="store_true", help="Run Day-1 simulation")
    ap.add_argument("--simulate-day2", action="store_true", help="Run Day-2 simulation with fault injection")
    args = ap.parse_args()

    G = build_demo_topology()

    if args.simulate:
        run_day1_simulation(G, duration=10)

    if args.simulate_day2:
        # Example fault events: (time_offset_sec, type, target, neighbor)
        faults = [
            (3, "device", "R2", None),
            (5, "link", "R1", "Switch1"),
        ]
        run_day2_simulation(G, duration=10, fault_events=faults)


if __name__ == "__main__":
    main()
