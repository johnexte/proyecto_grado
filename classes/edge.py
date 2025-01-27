class edge():
    def __init__(self, start_node_id: int, end_node_id: int, weight: float = 1):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.weight = weight

class power_edge(edge):
    def __init__(self, start_node_id: int, end_node_id: int, is_trafo: bool, weight: float = 1):
        super().__init__(start_node_id, end_node_id, weight)
        self.is_trafo: bool = is_trafo 
        self.L = 0           # Line flow
        self.L_min = 0       # Minimum security setting
        self.L_max = 0       # Maximum security setting
        self.L_minlimit = 0  # Minimum limit of the line capacity
        self.L_maxlimit = 0  # Maximum limit of the line capacity
        self.consequence = 0
        self.overload_probability = 0
        self.risk = 0
              
    def pij_line_overload_probability(self)->float:
        if self.L_min <= self.L < self.L_max:
            probability = 0
        elif self.L_max <= self.L < self.L_maxlimit:
            probability = ( ((1-0) * self.L) + (0 * self.L_maxlimit) - (self.L_max) ) / (self.L_maxlimit - self.L_max)
        else:
            probability = 1
        return probability

class communications_edge(edge):
    def __init__(self, start_node_id: int, end_node_id: int, weight: float):
        super().__init__(start_node_id, end_node_id, weight)
