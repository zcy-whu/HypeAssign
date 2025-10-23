class Node:
    
    STR_NODE_TYPE_COMMITTER = 'committer'
    STR_NODE_TYPE_ISSUE = 'issue'
    STR_NODE_TYPE_FILE = 'file'
    STR_NODE_TYPE_REPORTER = 'reporter'
    STR_NODE_TYPE_COMMENTER = 'commenter'
    STR_NODE_TYPE_DEVELOPER = 'developer'

    STR_NODE_TYPE_WORD = 'word'

    STR_NODE_TYPE_ORGANIZATION = 'organization'
    STR_NODE_TYPE_REPO = 'repo'

    def __init__(self, key, nodeType, contentKey, description):
        self.id = key  
        self.connectedTo = []  
        self.in_degree = 0  
        self.out_degree = 0  
        self.type = nodeType  
        self.contentKey = contentKey  
        self.description = description  

    def add_edge(self, edge_id):
        
        if edge_id not in self.connectedTo:
            
            self.connectedTo.append(edge_id)

    def __str__(self):
        return "node id:" + str(self.id) + " type:" + self.type + "  description:" + self.description

    def get_id(self):
        return self.id
