### -----------------------
### Graph Structure for SCM
### -----------------------

class SCM:
    def __init__(self, nodes):
        self.nodes = {node.name: node for node in nodes}

    @property
    def edges(self):
        return [
            (parent, node.name)
            for node in self.nodes.values()
            for parent in node.parents
        ]
