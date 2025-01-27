from .packet import packet


class node:
    def __init__(self, node_id: int):
        self.node_id: int = node_id
        self.neighbors: dict[int,list[power_node|comms_node]] = {}
        self.degree: int = 0
        
    def add_neighbor(self, destination_node: 'node'):
        self.neighbors[destination_node.node_id] = destination_node
        self.degree += 1
        
    def remove_neighbor(self, destination_node: 'node'):
        del self.neighbors[destination_node.node_id]
        self.degree -= 1
         
class power_node(node):
    def __init__(self, node_id: int):
        super().__init__(node_id)
        self.node_id = node_id
        self.N1 = 0.5 # Proporción de la consecuencia topológica
        self.N2 = 0.5 # Proporción de la consecuencia eléctrica

class bus_node(power_node):
    def __init__(self, node_id: int):
        super().__init__(node_id)
        self.pi_avg = 0      # Average over load probability of node i
        self.p = 0           # Total power injection for node i
        self.p_min = 0       # Minimum security setting
        self.p_max = 0       # Maximum security setting
        self.p_maxlimit = 0  # Maximum upper limit
        
    def pi_bus_power_overload_probability(self)->float:
        if self.p_min <= self.p < self.p_max:
            probability = self.pi_avg
        elif self.p_max <= self.p < self.p_maxlimit:
            probability = ( ((1-self.pi_avg) * self.p) + (self.pi_avg * self.p_maxlimit) - (self.p_max) ) / (self.p_maxlimit - self.p_max)
        else:
            probability = 1
        return probability
    
class generator_node(power_node):
    def __init__(self, node_id: int):
        super().__init__(node_id)
        self.F_min = 0       # Minimum frequency margin
        self.F_max = 0       # Maximum frequency margin
        self.F_minlimit = 0  # Minimum frequency lower limit
        self.F_maxlimit = 0  # Maximum frequency upper limit
        self.pf_avg = 0      # Average frequency violation probability of generator i
        self.F = 0           # Generator frequency
        self.U_min = 0       # Minimum voltage margin
        self.U_max = 0       # Maximum voltage margin
        self.U_minlimit = 0  # Minimum voltage lower limit
        self.U_maxlimit = 0  # Maximum voltage upper limit
        self.pu_avg = 0      # Average voltage violation probability of generator i
        self.U = 0           # Generator voltage
        
    def pf_frecuency_violation_probability(self)->float:
        if self.F_min <= self.F < self.F_max:
            probability = self.pf_avg
        elif self.F_max <= self.F < self.F_maxlimit:
            probability = ( ((1-self.pf_avg) * self.F) + (self.pf_avg * self.F_maxlimit) - (self.F_max) ) / (self.F_maxlimit - self.F_max)
        elif self.F_minlimit <= self.F < self.F_min:
            probability = ( ((self.pf_avg-1) * self.F) + (self.F_min) - (self.pf_avg * self.F_minlimit) ) / (self.F_min - self.F_minlimit)
        else:
            probability = 1
        return probability

    def pu_voltage_violation_probability(self)->float:
        if self.U_min <= self.U < self.U_max:
            probability = self.pu_avg
        elif self.U_max <= self.U < self.U_maxlimit:
            probability = ( ((1-self.pu_avg) * self.U) + (self.pu_avg * self.U_maxlimit) - (self.U_max) ) / (self.U_maxlimit - self.U_max)
        elif self.U_minlimit <= self.U < self.U_min:
            probability = ( ((self.pu_avg-1) * self.U) + (self.U_min) - (self.pu_avg * self.U_minlimit) ) / (self.U_min - self.U_minlimit)
        else:
            probability = 1
        return probability

    def p_bus_failure_probability(self)->float:
        return max(self.pf_frecuency_violation_probability(), self.pu_voltage_violation_probability())

class comms_node(node):
    def __init__(self, node_id: int, is_control_centre: bool):
        super().__init__(node_id)
        self.is_control_centre: bool = is_control_centre
        self.queue: list[packet] = []
        self.hd = 0.75                     
        self.hc = 1 - self.hd              
        self.B = 20               

    def receive_packet(self, message: packet):
        self.queue.append(message)
        if message.destination_node_id == self.node_id:
            message.is_delivered = True
        
    def send_packet(self, message: packet, destination_node: 'comms_node'):
        destination_node.receive_packet(message)
        self.queue.remove(message)
        
    def process_packet(self, message: packet):
        self.queue.remove(message)
        control_signal = message.power_node_fail_id
        return control_signal