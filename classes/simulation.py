from . import plot
from .graph import power_graph, comms_graph
from .packet import packet
from .edge import communications_edge, power_edge
from .node import power_node, comms_node
import numpy as np
import networkx as nx
import random
import pandapower as pp

class simulation:
    def __init__(self):
        self.comms_graph: comms_graph = None
        self.power_graph: power_graph = None
        self.comms_power_edges: dict = {}
        self.power_comms_edges: dict = {}
        self.current_buses_with_failures: list[int] = []
        self.results: dict[str, any] = {}
        self.i: int = 0
        self.K: int = 7 
        self.alpha: float = 0.3
        self.T: int = 0
        self.vulnerable_line = None
        self.tij = None
        self.is_system_overloaded = False
        
        self.bus_consequences_hist = []
        self.topological_consequences_hist = []
        self.electrical_consequences_hist = []
        self.line_risks_hist = []
        self.line_consequences_hist = []
        self.line_probs_hist = []
        self.lines_triped = []
        self.risks_hist = []
        
        self.original_power_graph: power_graph = None
        self.original_comms_graph : comms_graph = None
    
    def join_graphs(self, case: str) -> dict:
        if self.comms_graph.number_of_nodes() <= self.power_graph.number_of_nodes():
            return None
        if case == "d2d":
            self.degree_to_degree_join()
        elif case == "random":
            self.random_join()
    
    def degree_to_degree_join(self) -> None:
        power_degree_tuples = [(key, value) for key, value in self.power_graph.degree.items()]
        sorted_power_degrees = sorted(power_degree_tuples, key=lambda x: x[1], reverse=True)
        comms_degree_tuples = [(key, value) for key, value in self.comms_graph.degree.items()]
        sorted_comm_degrees = sorted(comms_degree_tuples, key=lambda x: x[1], reverse=True)
        for i, power_node in enumerate(sorted_power_degrees):
            if i < len(sorted_comm_degrees):
                self.power_comms_edges[power_node[0]] = sorted_comm_degrees[i][0]
                self.comms_power_edges[sorted_comm_degrees[i][0]] = power_node[0]
        centre_node_id = sorted_comm_degrees[0][0]
        self.comms_graph.nodes[centre_node_id].is_control_centre = True
        self.comms_graph.centre_node = self.comms_graph.nodes[centre_node_id]
        self.original_comms_graph.nodes[centre_node_id].is_control_centre = True
        self.original_comms_graph.centre_node = self.comms_graph.nodes[centre_node_id]
        
        
    def random_join(self) -> None:
        comm_nodes = list(self.comms_graph.nodes)
        for power_node in self.power_graph.nodes:
            comm_node = random.choice(comm_nodes)
            self.power_comms_edges[power_node] = comm_node
            self.comms_power_edges[comm_node] = power_node
            comm_nodes.remove(comm_node)
        centre_node_id = random.choice(comm_nodes)
        self.comms_graph.nodes[centre_node_id].is_control_centre = True
        self.comms_graph.centre_node = self.comms_graph.nodes[centre_node_id]
    
    def run_fail_protocol(self, lines_tripped) -> dict:
        self.risk_assessment()
        self.n_1_contingency(lines_tripped, 'lines')
        self.get_new_operating_point()      
        
        while True and self.power_graph.net.converged:

            if self.does_power_flow_converge() == False or self.is_system_overloaded == True:
                break

            self.risk_assessment()
            self.vulnerable_line = self.rank_risks()
            overloaded_edges = pp.overloaded_lines(self.power_graph.net,100)

            if len(overloaded_edges) == 0:
                self.n_1_contingency(self.vulnerable_line, 'line')
                self.get_new_operating_point()
                continue

            self.vulnerable_line = None
            
            for overloaded_index in overloaded_edges:
                line = self.power_graph.net.line.loc[overloaded_index]
                overloaded_line = self.power_graph.edges[(line["from_bus"],line["to_bus"])]
                self.tij = self.calculate_tij(overloaded_line)
                print("tij: ",self.tij)

                bus = self.get_bus_from_tripped_line(overloaded_line)
                self.current_buses_with_failures += [bus]
                self.start_fail_protocol(bus)
                
                solved = False
                while self.T < self.tij and not solved:
                    solved = self.step_fail_protocol()
                    self.capture_vars(f"state {self.i}")

                if solved == False:
                    self.n_1_contingency(overloaded_line, 'line')
                    self.get_new_operating_point()
                    self.is_system_overloaded = True
                else:
                    self.T = 0
                    self.capture_vars(f"state {self.i}")
                    self.is_system_overloaded = True

            
            
    def start_fail_protocol(self, power_node_fail_id: str):
        comms_node_id_started = self.power_comms_edges[power_node_fail_id]
        comms_node_id_end = self.comms_graph.centre_node.node_id
        comms_node_started = self.comms_graph.get_node(comms_node_id_started)
        message = packet(comms_node_id_started, comms_node_id_end, power_node_fail_id)
        comms_node_started.receive_packet(message)
        self.comms_graph.add_message(message)
        if message.is_delivered:
            self.current_buses_with_failures.remove(message.power_node_fail_id)
        self.capture_vars("start", start_coms_node_id = comms_node_id_started, end_coms_node_id = comms_node_id_end)
        self.T += 1

    def step_fail_protocol(self):
        for message in self.comms_graph.current_packets.copy():
            current_node = self.comms_graph.get_node(message.current_node_id)
            next_node = self.comms_graph.move_forward(message)
            if next_node is not None:
                self.comms_graph.update_message(message, current_node, next_node)
                self.capture_vars("move", current_node=current_node, next_node=next_node)
            if message.is_delivered:
                self.T += 1
                del self.comms_graph.nodes_with_packets[next_node.node_id]
                self.comms_graph.current_packets.remove(message)
                control_signal = next_node.process_packet(message)
                self.current_buses_with_failures.remove(control_signal)
                self.capture_vars("delivered", current_node=current_node, next_node=next_node, control_signal=control_signal)
                return True
            self.i += 1
            self.T += 1

    def capture_vars(self, case: str, **kwargs) -> None:
        self.results[self.i] = {}
        self.results[self.i]["vulnerable"] = (self.vulnerable_line.start_node_id,self.vulnerable_line.end_node_id) if self.vulnerable_line != None else None
        self.results[self.i]["lines_triped"] = self.power_graph.lines_triped.copy()
        self.results[self.i]["buses_failed"] = self.current_buses_with_failures.copy()
        self.results[self.i]["action"] = case
        self.results[self.i]["queues"] = {source: [packet.to_dict()["source_node_id"] for packet in self.comms_graph.get_node(source).queue] for source in list(self.comms_graph.nodes_with_packets.keys())}
        self.results[self.i]["nodes_with_packets"] = list(self.comms_graph.nodes_with_packets.keys())
        self.results[self.i]["current_packets"] = [packet.to_dict() for packet in self.comms_graph.current_packets]
        self.results[self.i]["t"] = self.T
        self.results[self.i]["tij"] = self.tij
        
        if case == "start":
            self.results[self.i]["started_node"] = kwargs["start_coms_node_id"]
            self.results[self.i]["destination_node"] = kwargs["end_coms_node_id"]
                
        elif case == "move":
            self.results[self.i]["current_node"] = kwargs["current_node"].node_id
            self.results[self.i]["next_node"] = kwargs["next_node"].node_id
            
        elif case == "delivered":
            self.results[self.i]["bus_repaired"] = kwargs["control_signal"]
            self.results[self.i]["current_node"] = kwargs["current_node"].node_id
            self.results[self.i]["next_node"] = kwargs["next_node"].node_id
        
        elif case == "tripped":
            self.results[self.i]["line_tripped"] = kwargs["key"]
            
        self.results["bus_consequences"] = self.power_graph.buses_consequences.copy()
        self.results["topological_consequences"] = self.power_graph.buses_topological_consequences.copy()
        self.results["electrical_consequences_"] = self.power_graph.buses_electrical_consequences.copy()
        self.results["line_consequences"] = self.power_graph.lines_consequences.copy()
        self.results["bus_fail_probability"] = self.power_graph.buses_fail_p.copy()
        self.results["line_fail_probability"] = self.power_graph.lines_fail_p.copy()
        self.results["line_risks"] = self.power_graph.lines_risks.copy()
        self.results["lines_triped"] = self.power_graph.lines_triped.copy()
        self.results["bus_risks"] = self.power_graph.buses_risks.copy()
        self.results["net"] = self.power_graph.net.copy()
        
        self.i += 1
            
    def risk_assessment(self):
        self.power_graph.calculate_buses_risk()
        self.power_graph.calculate_lines_risk()
        self.topological_consequences_hist.append(self.power_graph.buses_topological_consequences.copy())
        self.electrical_consequences_hist.append(self.power_graph.buses_electrical_consequences.copy())
        self.bus_consequences_hist.append(self.power_graph.buses_consequences.copy())
        self.line_consequences_hist.append(self.power_graph.lines_consequences.copy())
        self.line_probs_hist.append(self.power_graph.lines_fail_p.copy())
        self.line_risks_hist.append(self.power_graph.lines_risks.copy())
        self.risks_hist.append(self.power_graph.buses_risks.copy())
        self.capture_vars(f"state {self.i}")
    
    def get_new_operating_point(self):
        self.power_graph.run_power_flow()
        self.capture_vars(f"state {self.i}")
        
    def n_1_contingency(self, line: power_edge | list[power_edge], contingency_case: str):
        net = self.power_graph.net
        if contingency_case == 'lines':
            for line in line:
                self.n_1_contingency(line, 'line')
        elif contingency_case == 'line':
            if line.is_trafo:
                net.trafo.loc[(net.trafo.hv_bus == line.start_node_id ) & (net.trafo.lv_bus == line.end_node_id), 'in_service'] = False
            else:
                net.line.loc[(net.line.from_bus == line.start_node_id ) & (net.line.to_bus == line.end_node_id), 'in_service'] = False
            print("tripped",line.start_node_id,line.end_node_id)
            print("converged",self.power_graph.net.converged)
            self.power_graph.lines_triped.append((line.start_node_id,line.end_node_id))
            self.capture_vars("tripped", key=(line.start_node_id,line.end_node_id))
            self.lines_triped.append((line.start_node_id,line.end_node_id))
            self.is_system_overloaded = self.power_graph.update_graph(line)

    def rank_risks(self):
        rank = sorted(self.power_graph.lines_risks.items(), key=lambda x: x[1], reverse=True)
        line = self.power_graph.edges[(rank[0][0][0],rank[0][0][1])]
        self.vulnerable_line = line
        return line

    # def is_system_overloaded(self) -> bool:
    #     for line_risk in self.power_graph.lines_risks.values():
    #         if line_risk > 0:
    #             first = False
    #     first = True
    #     net = self.power_graph.net
    #     if not first:
    #         for line in net.res_line.index:
    #             if net.res_line.at[line, 'p_from_mw'] != 0 or net.res_line.at[line, 'p_to_mw'] != 0:
    #                 return False 
    #         print("System Overloaded")
    #     return True
    
    def calculate_total_shedding():
        """
        Función que calcula la cantidad total de carga que se ha perdido.
        """
        pass
    
    def does_power_flow_converge(self)->bool:
        return self.power_graph.net.converged
    
    def get_bus_from_tripped_line(self, line: power_edge) -> int:
        bus = line.start_node_id
        print(f"Vulnerable bus: {bus}")
        return bus
    
    def calculate_tij(self, vulnerable_line: power_edge) -> float:
        print(f"{vulnerable_line.start_node_id},{vulnerable_line.end_node_id}: {vulnerable_line.L}, {vulnerable_line.L_max}")
        return self.K / ( abs( vulnerable_line.L / vulnerable_line.L_max ) ** self.alpha - 1 )
    
    def data_input(self, case, n, weight, join_case):
        self.power_graph = power_graph(case)
        self.comms_graph = comms_graph(n, weight)
        
        self.original_power_graph = power_graph(case)
        self.original_comms_graph = comms_graph(n, weight)
        
        self.comms_power_edges = self.join_graphs(join_case)
    
    def cascading_event_simulation(self, path: str, case: str, n: int, weight: float, join_case: str, tripped_lines: list[tuple[int,int]]):
        """
        Función que simula un evento de cascada en la red de potencia.
        """
        self.data_input(case, n, weight, join_case)
        self.get_new_operating_point()
        plot.plot_net(self.power_graph, path, case)
        lines_tripped = []
        for key, line in self.power_graph.edges.items():
            if key in tripped_lines or (key[1],key[0]) in tripped_lines:
                lines_tripped.append(line)
        self.run_fail_protocol(lines_tripped)
        return self.get_results()
        
    def set_example_graphs(self):
        # Not working
        
        self.power_graph = power_graph("example")
        self.comms_graph = comms_graph("example")
        
        power_graph_nodes = ["A", "B", "C", "D", "E"]
        communications_graph_nodes = ["a", "b", "c", "d", "e", "f", "g", "h", "CC"]
        
        for node in power_graph_nodes:
            power_graph.add_node(node)
            
        for node in communications_graph_nodes:
            comms_graph.add_node(node, False)
        
        comms_graph.centre_node = comms_graph.get_node("CC")
        comms_graph.get_node("CC").is_control_centre = True
        
        power_edges = [("B", "A"), ("C", "B"), ("D", "C"), ("E", "C"), ("E", "A"), ("C", "A"), ("D", "B")]
        communications_edges = [("a", "b"), ("c", "d"), ("e", "h"), ("h", "g"), ("g", "d"), ("d", "f"), ("f", "CC"), ("b", "CC"), ("a", "g"), ("c", "f")]
        power_coms_edges = {"power-comms": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, "comms-power": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}}
        
        for edge in power_edges:
            power_graph.add_edge(edge[0], edge[1])
            
        for edge in communications_edges:
            comms_graph.add_edge(edge[0], edge[1])
            
        graphs_edges = []
        for key, value in power_coms_edges["power-comms"].items():
            graphs_edges.append((key, value))
        
        results = self.run_fail_protocol(power_graph, comms_graph, power_coms_edges)
        
        figs = plot.plot_coms_interaction(comms_graph, power_graph, graphs_edges, results)
        
        return figs
    
    def get_results(self):
        graphs_edges = []
        for key, value in self.power_comms_edges.items():
            graphs_edges.append((key, value))
        hist = {"bus_consequences": self.bus_consequences_hist,
                "topological_consequences": self.topological_consequences_hist,
                "electrical_consequences": self.electrical_consequences_hist,
                "line_risks": self.line_risks_hist,
                "line_consequences": self.line_consequences_hist,
                "line_probs": self.line_probs_hist,
                "risks": self.risks_hist}
        return self.results, self.original_power_graph, self.original_comms_graph, graphs_edges, hist
        
        
    
    
    
    