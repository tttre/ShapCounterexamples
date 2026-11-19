### ------------------------
### Nodal Structure for SCM
### ------------------------

class Node:
    def __init__(self, name, parents=()):
        self.name = name
        self.parents = tuple(parents)
