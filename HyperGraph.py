import gc
from datetime import datetime

from Edge import Edge
from Node import Node
import numpy as np


class HyperGraph:

    def __init__(self):
        self.node_list = {}
        self.node_content_list = {}
        self.num_nodes = 0
        self.node_id_counter = 0
        self.node_id_map = {}
        self.node_type = {}

        self.edge_list = {}
        self.edge_content_list = {}
        self.num_edges = 0
        self.edge_id_counter = 0
        self.edge_id_map = {}

        self.DV = None
        self.DE = None
        self.H = None
        self.W = None
        self.A = None

    def add_node(self, nodeType, contentKey, description):

        node = self.get_node_by_content(nodeType=nodeType, contentKey=contentKey)
        if node is None:
            new_node = Node(key=self.node_id_counter, contentKey=contentKey, nodeType=nodeType, description=description)
            self.node_list[self.node_id_counter] = new_node
            self.node_id_map[self.num_nodes] = self.node_id_counter
            self.node_content_list[(nodeType, contentKey)] = new_node
            if nodeType not in self.node_type.keys():
                self.node_type[nodeType] = [new_node]
            else:
                self.node_type[nodeType].append(new_node)

            self.num_nodes += 1
            self.node_id_counter += 1
            return new_node
        else:
            return node

    def get_node_by_key(self, n):
        return self.node_list.get(n, None)

    def get_node_by_content(self, nodeType, contentKey):
        return self.node_content_list.get((nodeType, contentKey), None)

    def add_edge(self, nodes, edgeType, description, weight, nodeObjects=None, queryBeforeAdd=False):

        nodes.sort()

        if queryBeforeAdd:
            edge = self.get_edge_by_content(edgeType, nodes)
            if edge is not None:
                return edge

        edge = Edge(key=self.edge_id_counter, edgeType=edgeType, description=description, weight=weight)
        edge.add_nodes(nodes)
        self.edge_list[self.edge_id_counter] = edge
        self.edge_id_map[self.num_edges] = self.edge_id_counter
        self.num_edges += 1
        self.edge_id_counter += 1
        self.edge_content_list[(edgeType, tuple(nodes))] = edge

        if nodeObjects is not None:
            for node in nodeObjects:
                node.add_edge(edge.id)
        else:
            for node in nodes:
                node = self.get_node_by_key(node)
                node.add_edge(edge.id)
        return edge

    def get_edge_by_key(self, n):
        return self.edge_list.get(n, None)

    def get_edge_by_content(self, edgeType, nodes):
        return self.edge_content_list.get((edgeType, tuple(nodes)), None)

    def get_nodes(self):
        return self.node_list.keys()


    def add_node_to_edge(self, n, edge_id):
        node = self.get_node_by_key(n)
        edge = self.get_edge_by_key(edge_id)
        if isinstance(node, Node):
            node.add_edge(edge_id)

            self.edge_content_list.pop((edge.type, tuple(edge.connectedTo)))
            edge.connectedTo.append(node.id)
            edge.connectedTo.sort()
            self.edge_content_list[(edge.type, tuple(edge.connectedTo))] = edge

    def remove_node_by_key(self, n):

        node = self.get_node_by_key(n)
        edges = node.connectedTo
        deleteEdgeIdList = []
        for edge_id in edges:
            edge = self.get_edge_by_key(edge_id)
            if edge.connectedTo.__len__() > 2:
                self.edge_content_list.pop((edge.type, tuple(edge.connectedTo)))
                edge.connectedTo.remove(node.id)
                self.edge_content_list[(edge.type, tuple(edge.connectedTo))] = edge
            else:
                for node_id_temp in edge.connectedTo:
                    if node_id_temp != node.id:
                        node_temp = self.get_node_by_key(node_id_temp)
                        node_temp.connectedTo.remove(edge_id)
                deleteEdgeIdList.append(edge_id)
        for edge_id in deleteEdgeIdList:
            edge = self.edge_list.pop(edge_id)
            nodes = list(edge.connectedTo)
            nodes.sort()
            nodes = tuple(nodes)
            self.edge_content_list.pop((edge.type, nodes))
            edgeIdList = []
            for i in range(0, self.num_edges):
                edgeIdList.append(self.edge_id_map[i])
            edgeIdList.remove(edge_id)
            self.edge_id_map.clear()
            for index, edge_res_id in enumerate(edgeIdList):
                self.edge_id_map[index] = edge_res_id
            self.num_edges -= 1
        self.node_list.pop(n)
        self.node_content_list.pop((node.type, node.contentKey))
        nodeIdList = []
        for i in range(0, self.num_nodes):
            nodeIdList.append(self.node_id_map[i])
        nodeIdList.remove(n)
        self.node_id_map.clear()
        for index, node_res_id in enumerate(nodeIdList):
            self.node_id_map[index] = node_res_id
        self.num_nodes -= 1

    def updateMatrix(self):

        inverseNodeMap = {k: v for v, k in self.node_id_map.items()}
        self.W = np.zeros((self.num_edges, 1))
        self.H = np.zeros((self.num_nodes, self.num_edges))
        self.DE = np.zeros((self.num_edges, 1))
        self.DV = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(0, self.num_edges):
            edge_1 = self.get_edge_by_key(self.edge_id_map[i])
            self.W[i] = edge_1.weight
            self.DE[i] = edge_1.connectedTo.__len__()
            for node_id in edge_1.connectedTo:
                node_matrix_id = inverseNodeMap[node_id]
                self.H[node_matrix_id][i] = 1
                self.DV[node_matrix_id][node_matrix_id] += edge_1.weight

        DV_inv = np.linalg.inv(self.DV + 1e-6 * np.eye(self.num_nodes))
        DV_sqrt = np.sqrt(DV_inv)

        A = np.dot(DV_sqrt, self.H)
        DE_inv = np.array(list(map(lambda x: 1 / x, self.DE)))
        W_DE_Inv = np.multiply(self.W, DE_inv)

        startTime = datetime.now()

        for index, la in enumerate(W_DE_Inv):
            A[:, index] = A[:, index] * la
        print("cost time", datetime.now() - startTime)

        A = np.dot(A, self.H.T)
        A = np.dot(A, DV_sqrt)

        del self.W
        del self.H
        gc.collect()
        self.W = None
        self.H = None

        self.A = A
