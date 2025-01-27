import networkx as nx
from .packet import packet
from .node import node, comms_node, bus_node, generator_node, power_node
from .edge import edge, power_edge
from pandapower import networks as pn
from pandapower import pandapowerNet
import numpy as np
from pandapower import runpp
from networkx.algorithms.flow   import maxflow
import math
from rich.progress import Progress
from pandapower import contingency
import pandapower as pp

class graph:
    def __init__(self):
        self.nodes: dict[int, node] = {}
        self.edges: dict[tuple[int,int],edge] = {}
        self.degree: dict[int,int] = {}
        self.graph: nx.graph = None
        
    def add_edge(self, source_node_id: int, destination_node_id: int, capacity: float = 1):
        self.nodes[source_node_id].add_neighbor(self.nodes[destination_node_id])
        self.nodes[destination_node_id].add_neighbor(self.nodes[source_node_id])
        self.graph.add_edge(source_node_id, destination_node_id, capacity=capacity, weight=capacity)
        self.edges[(source_node_id, destination_node_id)] = capacity
        self.degree[source_node_id] = len(self.nodes[source_node_id].neighbors)
        self.degree[destination_node_id] = len(self.nodes[destination_node_id].neighbors)
        
    def remove_edge(self, source_node_id: int, destination_node_id: int):
        self.nodes[source_node_id].remove_neighbor(self.nodes[destination_node_id])
        self.nodes[destination_node_id].remove_neighbor(self.nodes[source_node_id])
        self.graph.remove_edge(source_node_id, destination_node_id)
        del self.edges[(source_node_id, destination_node_id)]
        self.degree[source_node_id] = len(self.nodes[source_node_id].neighbors)
        self.degree[destination_node_id] = len(self.nodes[destination_node_id].neighbors)
        
    def shortest_path_length(self, source_node: node, destination_node: node)->float:
        try:
            return nx.shortest_path_length(self.graph, source_node.node_id, destination_node.node_id)
        except:
            return float("inf")
        
    def number_of_nodes(self) -> int:
        return len(self.nodes)
    
    def get_node(self, node_id: int) -> comms_node | bus_node | generator_node:
        return self.nodes[node_id]
    
    def remove_node(self, node_id: int):
        del self.nodes[node_id]
        self.graph.remove_node(node_id)
        del self.degree[node_id]
        for neighbor_id in self.nodes[node_id].neighbors:
            self.remove_edge(node_id, neighbor_id)

class power_graph(graph):
    def __init__(self, case: str):
        super().__init__()
        self.nodes: dict[int, bus_node | generator_node] = {}
        self.edges: dict[tuple[int,int], power_edge] = {}
        self.graph: nx.graph = nx.Graph()
        self.net: pandapowerNet = None
        self.buses_topological_consequences: dict[int,float] = {}
        self.buses_electrical_consequences: dict[int,float] = {}
        self.buses_consequences: dict[int,float] = {}
        self.buses_fail_p: dict[int,float] = {}
        self.lines_fail_p: dict[tuple[int,int],float] = {}
        self.buses_risks: dict[int,float] = {}
        self.lines_consequences: dict[tuple[int,int],float] = {}
        self.lines_risks: dict[tuple[int,int],float] = {}
        self.lines_triped: list[tuple[int,int]] = []
        self.power_flow = None
        self.case: str = case
        self.flows: dict[tuple[int,int],tuple] = {}
        self.create_graph()
    
    def add_node(self, node: bus_node | generator_node):
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)
        self.degree[node.node_id] = 0
        
    def calculate_transmission_line_capacity(self, edge)-> float:
        return abs(self.net._ppc["internal"]["Ybus"][edge[0],edge[1]])
    
    def degrees(self):
        for node_id in self.nodes:
            self.degree[node_id] = self.graph.degree[node_id]

    def create_graph(self, except_edges: list[tuple[int,int]] = []):
        with Progress() as progress:
            task = progress.add_task("[blue]Initializing power graph...", total=3)

            self.lines_triped += except_edges
            self.net: pandapowerNet = pn.__getattribute__(self.case)()
            
            self.generators = self.net.gen.bus.to_list() + self.net.ext_grid.bus.to_list()
            self.loads = self.net.load.bus.to_list()
            self.buses = self.net.bus

            self.source_nodes = [value for value in self.generators if value not in self.loads]
            self.sink_nodes =  [value for value in self.loads if value not in self.generators]

            self.line_edges = {(self.net.line.from_bus.at[i], self.net.line.to_bus.at[i]): self.net.line.loc[i].to_dict() for i in self.net.line.index}
            self.trafo_edges = {(self.net.trafo.hv_bus.at[i], self.net.trafo.lv_bus.at[i]): self.net.trafo.loc[i].to_dict() for i in self.net.trafo.index}
            self.buses_info = {bus: self.net.bus.loc[bus].to_dict() for bus in self.net.bus.index}
            
            task_1 = progress.add_task("[cyan]Setting buses...", total=len(self.net.bus.index))
            task_2 = progress.add_task("[cyan]Setting lines...", total=len(self.line_edges))
            task_3 = progress.add_task("[cyan]Setting trafos...", total=len(self.trafo_edges))

            for bus_id in self.net.bus.index:
                if bus_id in self.generators:
                    self.add_node(generator_node(bus_id))    
                else:
                    self.add_node(bus_node(bus_id))
                progress.update(task_1, advance=1)
            progress.update(task, advance=1)
            
            for edge in self.line_edges:
                self.add_edge(edge[0], edge[1])
                self.edges[edge] = power_edge(edge[0], edge[1], False)
                progress.update(task_2, advance=1)
            progress.update(task, advance=1)

            for edge in self.trafo_edges:
                self.add_edge(edge[0], edge[1])
                self.edges[edge] = power_edge(edge[0], edge[1], True)
                progress.update(task_3, advance=1)
            progress.update(task, advance=1)
            
            self.generation = self.net.gen
            self.degrees()
            self.run_power_flow()

            for key, edge in self.edges.items():
                edge.weight = self.calculate_transmission_line_capacity(key)
                self.graph[key[0]][key[1]]['capacity'] = edge.weight

        
    def update_graph(self, vulnerable_line: power_edge):
        self.remove_edge(vulnerable_line.start_node_id,vulnerable_line.end_node_id)
        if len(self.edges) == 0:
            return True
        return False
    
    def calculate_buses_risk(self) -> None:
        with Progress() as progress:
            task = progress.add_task("[cyan]Calculating buses risk...", total=len(self.nodes))
            for bus_id, node in self.nodes.items():
                self.buses_topological_consequences[bus_id] = self.topological_consequence(node)
                self.buses_electrical_consequences[bus_id] = self.electrical_consequence(node)
                self.buses_consequences[bus_id] = self.bus_consequence(node)
                self.buses_fail_p[bus_id] = self.bus_fail_p(node)
                self.buses_risks[bus_id] = self.bus_risk(node)
                progress.update(task, advance=1)
        progress.stop()   
            
    def calculate_maximum_flows(self) -> None:
        with Progress() as progress:
            m = self.source_nodes
            n = self.sink_nodes
            task = progress.add_task("[cyan]Calculating maximum flows...", total=len(m)*len(n))
            for u in m:
                for v in n:
                    if u != v:
                        self.flows[(u,v)] = maxflow.maximum_flow(self.graph, u, v, capacity="capacity")
                    progress.update(task, advance=1)
            progress.stop()
                    
    def calculate_lines_risk(self) -> None:
        self.calculate_maximum_flows()
        with Progress() as progress:
            task = progress.add_task("[cyan]Calculating line risks...", total=len(self.edges.values()))
            self.line_consequences: dict[tuple[int,int]] = {}
            self.lines_risks: dict[tuple[int,int]] = {}
            for edge_tuple, edge in self.edges.items():
                self.lines_consequences[edge_tuple] = self.line_consequence(edge, self.source_nodes, self.sink_nodes)
                self.lines_fail_p[edge_tuple] = self.line_fail_p(edge)
                self.lines_risks[edge_tuple] = self.line_risk(edge)
                progress.update(task, advance=1)
            progress.stop()
    
    def run_power_flow(self) -> None:
        try:
            runpp(self.net,init="results", enforce_q_lims=True, run_control=True)
            for key, edge in self.edges.items():
                if not edge.is_trafo:
                    line = self.net.line[( self.net.line['from_bus'] == edge.start_node_id) & ( self.net.line['to_bus'] == edge.end_node_id)]
                    max_flow = line.loc[line.index[0], 'max_i_ka']
                    flow = self.net.res_line.loc[line.index[0], 'i_ka']
                else:
                    line = self.net.trafo[( self.net.trafo["hv_bus"] == edge.start_node_id) & ( self.net.trafo["lv_bus"] == edge.end_node_id)]
                    Sn = line.loc[line.index[0], 'sn_mva']
                    Vkv = max(line.loc[line.index[0], 'vn_hv_kv'],line.loc[line.index[0], 'vn_lv_kv'])
                    max_flow = Sn * line.loc[line.index[0], 'max_loading_percent'] / 100
                    flow = Sn / ( 3**(1/2) * Vkv )
                edge.L = flow*10*2.5
                edge.L_max = max_flow
                edge.L_maxlimit = max_flow * 1.4
        except Exception as e:
            print(f"Power flow did not converge, error {e}")
            self.net.converged = False
    
    def topological_consequence(self, node: power_node)->float:
        # if max(self.degree.values()) == 0:
        #     self.net.converged = False
        #     return 0
        return self.nodes[node.node_id].degree / max(self.degree.values())
    
    def electrical_consequence(self, node: power_node)->float:
        return abs(self.net.res_bus.at[node.node_id, 'p_mw']) / max(abs(self.net.res_bus['p_mw']))
    
    def bus_consequence(self, node: power_node)->float:
        return node.N1 * self.topological_consequence(node) + node.N2 * self.electrical_consequence(node)    
    
    def bus_fail_p(self, node: bus_node | generator_node)->float:
        if type(node) == bus_node:
            return node.pi_bus_power_overload_probability()
        else:
            return node.p_bus_failure_probability()

    def bus_risk(self, node: bus_node | generator_node)->float:
        if type(node) == bus_node:
            return self.bus_consequence(node) * node.pi_bus_power_overload_probability()
        else:
            return self.bus_consequence(node) * node.p_bus_failure_probability()
    
    def line_consequence(self, line: power_edge, m: list, n: list):
        sum_f_uv_ij = 0
        sum_f_uv_max = 0
        for u in m:
            for v in n:
                if u != v:
                    f_uv_ij = self.flows[(u,v)][1][line.start_node_id][line.end_node_id]
                    f_uv_max = self.flows[(u,v)][0]
                    sum_f_uv_ij += f_uv_ij
                    sum_f_uv_max += f_uv_max
        if sum_f_uv_max == 0:
            return 0
        line.consequence = sum_f_uv_ij / sum_f_uv_max
        return line.consequence

    def line_fail_p(self, line: power_edge)->float:
        return line.pij_line_overload_probability()
    
    def line_risk(self, line: power_edge)->float:
        return line.consequence * line.pij_line_overload_probability()
    
class comms_graph(graph):
    def __init__(self, n: int = 10, weight: float = 1):
        super().__init__()
        self.graph: nx.graph = nx.Graph()
        self.nodes: dict[int, comms_node] = {}
        self.type: str = type
        self.centre_node: comms_node = None
        self.nodes_with_packets: dict[int,list[packet]] = {}
        self.current_packets: list[packet] = []
        self.create_scale_free_graph(n, weight)
            
    def create_scale_free_graph(self, n: int, weight: float):
        graph = nx.scale_free_graph(n, seed=41)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        for node in graph.nodes:
            self.add_node(node, False)
        for edge in graph.edges:
            self.add_edge(edge[0], edge[1], weight)
    
    def add_message(self, message: packet):
        if message.current_node_id not in self.nodes_with_packets:
            self.nodes_with_packets[message.current_node_id] = []
        self.nodes_with_packets[message.current_node_id].append(message)
        self.current_packets.append(message)
        
    def update_message(self, message: packet, current_node: comms_node, next_node: comms_node):
        if current_node.node_id != message.destination_node_id:
            self.nodes_with_packets[current_node.node_id].remove(message)
            self.current_packets.remove(message)
            if len(self.nodes_with_packets[current_node.node_id]) == 0:
                del self.nodes_with_packets[current_node.node_id]
            if next_node.node_id not in self.nodes_with_packets:
                self.nodes_with_packets[next_node.node_id] = []
            self.nodes_with_packets[next_node.node_id].append(message)
            self.current_packets.append(message)
        
    def add_node(self, node_id: int, is_control_centre: bool):
        self.nodes[node_id] = comms_node(node_id, is_control_centre)
        self.graph.add_node(node_id)
        self.degree[node_id] = 0

    def move_forward(self, message: packet) -> comms_node:
        current_node = self.get_node(message.current_node_id)
        next_node = self.get_next_node(message)
        current_node.send_packet(message, next_node)
        message.path.append(next_node)
        message.current_node_id = next_node.node_id
        return next_node
        
    def get_next_node(self, message: packet) -> comms_node:
        current_node = self.get_node(message.current_node_id)
        destination_node = self.get_node(message.destination_node_id)
        if destination_node.node_id in current_node.neighbors:
            return destination_node
        else:
            lista = list(current_node.neighbors.values())
            return max(lista, key=lambda destination_node: self.P(current_node, destination_node, destination_node))
    
    def P(self, current_node: comms_node, destination_node: comms_node, control_centre_node: comms_node)->float:
        x = math.exp(-destination_node.B * self.H(destination_node, control_centre_node))
        y = sum(math.exp(-neighbor.B * self.H(neighbor, control_centre_node)) for neighbor in current_node.neighbors.values())
        return x/y
    
    def H(self, current_node: comms_node, control_centre_node: comms_node)->float:
        d = self.shortest_path_length(current_node, control_centre_node)
        return current_node.hd * d + current_node.hc * len(current_node.queue)

    
    

    
