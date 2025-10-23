class Edge:
    """超图的边，边可以包含多个顶点"""

    STR_EDGE_TYPE_ISSUE_DIS = 'issue relation'
    STR_EDGE_TYPE_REPORT_RELATION = 'report'
    STR_EDGE_TYPE_COMMENTER_RELATION = 'comment'
    STR_EDGE_TYPE_SIMILAR_RELATION = 'similar'
    # STR_EDGE_TYPE_FIX_RELATION = 'fix'
    STR_EDGE_TYPE_CREATE_RELATION = 'create'
    STR_EDGE_TYPE_REMOVE_RELATION = 'remove'
    STR_EDGE_TYPE_MODIFY_RELATION = 'modify'


    def __init__(self, key, edgeType, description, weight=0):
        self.id = key
        self.connectedTo = []
        self.weight = weight
        self.type = edgeType
        self.description = description

    def add_nodes(self, nodes):
        for node_id in nodes:
            if node_id not in self.connectedTo:
                self.connectedTo.append(node_id)

    def __str__(self):
        return "node id:" + str(self.id) + " type:" + self.type + "  description:" + self.description
