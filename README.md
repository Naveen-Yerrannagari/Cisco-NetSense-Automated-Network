# NetSense-Automated-Network-Insight-Resilience-Cisco-Internship-
NetSense automates enterprise network management by generating instant topologies, detecting configuration errors, optimizing traffic flow, and streamlining troubleshooting. It reduces downtime, boosts security, and delivers real-time visibility, making networks resilient and efficient with minimal manual effort.
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-Educational-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🌐 Overview
The **Cisco VIP 2025 Network Tool** is an automated solution for generating, validating, simulating, and exporting network topologies from Cisco router and switch configuration files.  
It allows network engineers and students to quickly **reconstruct network environments**, **analyze traffic loads**, **identify configuration issues**, and **simulate fault-tolerant scenarios**.

The tool produces output compatible with **Cisco Packet Tracer**, enabling seamless manual integration.

---

## ✨ Features

### 🏗 Automatic Network Topology Generation
- Parses Cisco router and switch configs to create a hierarchical network topology.
- Identifies node roles and link bandwidth.

### ⚙ Configuration Validation & Optimization
- Detects duplicate IPs, overlapping subnets, VLAN issues, gateway inconsistencies, MTU mismatches, missing configs, and potential loops.

### 📊 Load and Traffic Analysis
- Compares traffic demands with link capacities.
- Provides **load balancing recommendations** for scalable networks.

### 🖥 Day-1 & Day-2 Network Simulation
- Multithreaded simulation with ARP and OSPF protocol emulation.
- Supports link/device failure injection and pause/resume for testing fault tolerance.

### 💾 Export Packet Tracer-Compatible Files
- ZIP archive containing startup configs, topology JSON, CSV link lists, diagrams, and usage instructions.

---
## 🗂 File Structure

```text
Cisco-VIP-2025-Network-Tool/
│
├── configs/                # Router configuration files (Cisco IOS format)
│   ├── R1.cfg
│   ├── R2.cfg
│   └── ...
│
├── switches/               # Switch configuration files (optional)
│   ├── S1.cfg
│   └── ...
│
├── topology/               # Optional manual topology input files
│   ├── links.csv           # Manual link definitions
│   ├── endpoints.csv       # Endpoint IP and gateway definitions
│   └── traffic_demands.csv # Traffic load demands
│
├── output/                 # Generated outputs
│   ├── network_topology.png       # Visual topology diagram
│   ├── validation_report.txt      # Configuration validation report
│   ├── load_analysis.txt          # Load and traffic analysis
│   ├── <device>_day1_log.txt     # Day-1 simulation logs
│   ├── <device>_day2_log.txt     # Day-2 simulation logs
│   └── network_project.pkt        # Packet Tracer-compatible ZIP archive
│
├── nettopogen.py           # Main Python script
├── README.md               # Project documentation
└── requirements.txt        # Optional: Python dependencies
```
## 🛠 Requirements

- Python 3.7+
- Libraries: `networkx`, `matplotlib`

Install dependencies:

```bash
pip install networkx matplotlib
```
# 🚀 Usage
Run the tool:

```bash
python nettopogen.py [options]
```
# ⚡ Common Options
Flag	Description
--configs	Directory with router configs (default: configs)
--switches	Directory with switch configs (optional)
--links	CSV with manual link definitions (optional)
--endpoints	CSV with endpoint definitions (optional)
--traffic	CSV with traffic demands (default: traffic_demands.csv)
--draw	Generate network topology PNG
--analyze	Run configuration validation & load analysis
--export-pkt	Export ZIP archive for Packet Tracer
--recommend-lb	Generate load balancing recommendations
--check-missing	Check for missing config files
--simulate-day1	Run Day-1 simulation
--simulate-day2	Run Day-2 simulation with faults
![Options](Project/screenshots/options.png) 

# 📌 Example Usage
Generate topology, analyze configs, run simulation, and export:

``` bash
python nettopogen.py --configs ./configs --draw --analyze --simulate-day1 --export-pkt
```

Run load balancing recommendations with custom traffic file:

```bash
python nettopogen.py --recommend-lb --traffic my_traffic.csv
```

# 📁 Input File Formats
Router/Switch Configs: Standard Cisco IOS text files (--configs & --switches)

links.csv: Optional CSV specifying manual links:

``` css
endpointA,endpointB,bandwidth_mbps,mtu,label
R1,R2,100,1500,ptp
```
endpoints.csv: Optional CSV with endpoint IP and gateway info

traffic_demands.csv: Optional CSV describing traffic load:

```swift
src_subnet,dst_subnet,mbps
192.168.10.0/24,192.168.20.0/24,50
```
# 📂 Output Files
**Topology image:** `output/network_topology.png` ![Network Topology](Project/output/topology.png)  

- **Validation report:** [validation_report.txt](Project/output/validation_report.txt)  
- **Load analysis:** [load_analysis.txt](Project/output/load_analysis.txt)

- **Simulation logs:**
  - [R1 Day-1 Log](Project/output/R1_day1_log.txt)
  - [R2 Day-1 Log](Project/output/R2_day1_log.txt)
  - [R1 Day-2 Log](Project/output/R1_day2_log.txt)
  - [R2 Day-2 Log](Project/output/R2_day2_log.txt)

- **Packet Tracer archive:** [network_project.pkt](Project/output/network_project.pkt)

# 🖱 How to Import into Cisco Packet Tracer
Extract network_project.pkt ZIP archive

Open Packet Tracer

Manually create devices from configs/

Paste startup configs into device CLI using _startup.cfg

Connect devices using topology/links.csv or visualize with network_topology.png

Save your Packet Tracer .pkt file

# 🌟 Project Highlights
Automatic topology reconstruction from raw Cisco configs

Comprehensive validation: IP conflicts, VLANs, routing anomalies

Traffic-based load analysis with actionable recommendations
![Load Balancing](Project/screenshots/Load_Balancing.png) 
Day-1/Day-2 scenario emulation with fault injection
![Simulations](Project/screenshots/Simulation.png)
![Generated Log Files](Project/screenshots/Log_files.png) 
Bridges automation with manual import for Packet Tracer
# 🤝 Contributing
Contributions are welcome! Submit issues or pull requests via the repository.

# 📜 License
For educational and internal use as part of the Cisco Virtual Internship Program 2025.
Redistribution or commercial use is subject to Cisco’s terms.

Thank you for using the Cisco VIP 2025 Network Tool! 🚀

