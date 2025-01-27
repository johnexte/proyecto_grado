class packet:
    def __init__(self, source_node_id: int, destination_node_id: int, power_node_fail_id: int):  
        self.source_node_id: int = source_node_id
        self.destination_node_id: int = destination_node_id
        self.path: list = []
        self.current_node_id: int = source_node_id
        self.is_delivered: bool = False
        self.power_node_fail_id: int = power_node_fail_id
             
    def to_dict(self):
        return {"source_node_id": self.source_node_id, 
                "destination_node_id": self.destination_node_id, 
                "path": [node.node_id for node in self.path], 
                "current_node_id": self.current_node_id, 
                "is_delivered": self.is_delivered, 
                "power_node_fail_id": self.power_node_fail_id}
   