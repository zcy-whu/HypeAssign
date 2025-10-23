import pickle
from queue import Queue

from HyperGraph import HyperGraph
from Edge import Edge
from Node import Node
import pandas as pd
from tqdm import tqdm

class HyperGraphHelper:

    @staticmethod
    def createTrainDataGraph_zcy(args, train_data, train_data_create, train_data_modified, train_data_removed,
                                 train_data_report_closed, train_data_comment, train_data_similar, train_fixers,
                                 val_data, val_data_create, val_data_modified, val_data_removed, val_data_report,
                                 val_data_comment, val_data_similar, val_fixers,
                                 test_data, test_data_create, test_data_modified, test_data_removed,
                                 test_data_report, test_data_comment, test_data_similar, test_fixers): 

        graph = HyperGraph()
       
        issueList = list(set(list(train_data['issue_node_id']) + list(val_data['issue_node_id']) + list(test_data['issue_node_id'])))
        for issue in issueList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_ISSUE, contentKey=issue, description=f"issue:{issue}")

        fixerList = [item for sublist in train_fixers.values() for item in sublist] + [item for sublist in val_fixers.values() for item in sublist] + [item for sublist in test_fixers.values() for item in sublist]
        devList = list(set(fixerList + list(train_data_create['developer']) + list(train_data_modified['developer']) + list(train_data_removed['developer'])
                           + list(train_data_report_closed['developer']) + list(train_data_comment['developer'])
                           + list(val_data_create['developer']) + list(val_data_modified['developer']) + list(val_data_removed['developer'])
                           + list(val_data_report['developer']) + list(val_data_comment['developer'])
                           + list(test_data_create['developer']) + list(test_data_modified['developer']) + list(test_data_removed['developer'])
                           + list(test_data_report['developer']) + list(test_data_comment['developer'])))
        for dev in devList:  
            graph.add_node(nodeType=Node.STR_NODE_TYPE_DEVELOPER, contentKey=dev, description=f"developer:{dev}")
        
        fileList = list(set(list(train_data_similar['file']) + list(train_data_create['file']) + list(train_data_modified['file']) + list(train_data_removed['file'])
                            + list(val_data_similar['file']) + list(val_data_create['file']) + list(val_data_modified['file']) + list(val_data_removed['file'])
                            + list(test_data_similar['file']) + list(test_data_create['file']) + list(test_data_modified['file']) + list(test_data_removed['file'])))

        for file in fileList:
            graph.add_node(nodeType=Node.STR_NODE_TYPE_FILE, contentKey=file, description=f"file:{file}")

        
        for issue in issueList:
            issue_node = graph.get_node_by_content(Node.STR_NODE_TYPE_ISSUE, issue) 

            # issue-reporter hyperedge
            report_df = pd.concat([train_data_report_closed, val_data_report, test_data_report])
            reporters = list(report_df[report_df['issue'] == issue]['developer'])
            reporter_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, reporter).id for reporter in reporters]
            reporter_nodesList.append(issue_node.id)

            weight = sum(list(report_df[report_df['issue'] == issue]['weight']))
            graph.add_edge(nodes=reporter_nodesList, edgeType=Edge.STR_EDGE_TYPE_REPORT_RELATION, weight= args.report * weight, description=f" report relation between issue {issue} and reporters")

            # issue-commenter hyperedge
            comment_df = pd.concat([train_data_comment, val_data_comment, test_data_comment])
            if issue in list(comment_df['issue']):
                commenters = list(comment_df[comment_df['issue'] == issue]['developer'])
                commenters_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, commenter).id for commenter in commenters]
                commenters_nodesList.append(issue_node.id)

                weight = sum(list(comment_df[comment_df['issue'] == issue]['weight']))
                graph.add_edge(nodes=commenters_nodesList, edgeType=Edge.STR_EDGE_TYPE_COMMENTER_RELATION, weight= args.comment * weight, description=f" commenter relation between issue {issue} and commenters")

            # issue-file fix hyperedge
            similar_df = train_data_similar
            if issue in list(similar_df['issue']):
                similar_files = list(similar_df[similar_df['issue'] == issue]['file'])
                similar_files_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_FILE, similar_file).id for similar_file in similar_files]
                similar_files_nodesList.append(issue_node.id)

                weight = sum(list(similar_df[similar_df['issue'] == issue]['weight']))
                graph.add_edge(nodes=similar_files_nodesList, edgeType=Edge.STR_EDGE_TYPE_SIMILAR_RELATION, weight= args.similar * weight, description=f" similar relation between issue {issue} and files")

        # dev-file hyperedge
        for dev in devList:
            dev_node = graph.get_node_by_content(Node.STR_NODE_TYPE_DEVELOPER, dev)

            # create hyperedge
            create_df = train_data_create
            if dev in list(create_df['developer']):
                create_files = list(create_df[create_df['developer'] == dev]['file'])
                create_files_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_FILE, create_file).id for create_file in create_files]
                create_files_nodesList.append(dev_node.id)

                weight = sum(list(create_df[create_df['developer'] == dev]['weight']))
                graph.add_edge(nodes=create_files_nodesList, edgeType=Edge.STR_EDGE_TYPE_CREATE_RELATION, weight= args.create * weight, description=f" create relation between developer {dev} and files")

            # remove hyperedge
            removed_df = train_data_removed
            if dev in list(removed_df['developer']):
                remove_files = list(removed_df[removed_df['developer'] == dev]['file'])
                remove_files_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_FILE, remove_file).id for remove_file in remove_files]
                remove_files_nodesList.append(dev_node.id)

                weight = sum(list(removed_df[removed_df['developer'] == dev]['weight']))
                graph.add_edge(nodes=remove_files_nodesList, edgeType=Edge.STR_EDGE_TYPE_REMOVE_RELATION, weight= args.remove * weight, description=f" remove relation between developer {dev} and files")

            # modify hyperedge
            modified_df = train_data_modified
            if dev in list(modified_df['developer']):
                modify_files = list(modified_df[modified_df['developer'] == dev]['file'])
                modify_files_nodesList = [graph.get_node_by_content(Node.STR_NODE_TYPE_FILE, modify_file).id for modify_file in modify_files]
                modify_files_nodesList.append(dev_node.id)

                weight = sum(list(modified_df[modified_df['developer'] == dev]['weight']))
                graph.add_edge(nodes=modify_files_nodesList, edgeType=Edge.STR_EDGE_TYPE_MODIFY_RELATION, weight= args.modify * weight, description=f" modify relation between developer {dev} and files")

        return graph

    @staticmethod
    def SaveAndGetGraph_zcy(args, chunk_id, train_data, train_data_create, train_data_modified,
                            train_data_removed, train_data_report_closed, train_data_comment, train_data_similar,
                            train_fixers, pathDict,
                            val_data, val_data_create, val_data_modified, val_data_removed,
                            val_data_report, val_data_comment, val_data_similar, val_fixers,
                            test_data, test_data_create, test_data_modified, test_data_removed,
                            test_data_report, test_data_comment, test_data_similar, test_fixers,
                            issueCreatedTimeMap):
        graph = HyperGraphHelper.createTrainDataGraph_zcy(args, train_data, train_data_create, train_data_modified,
                                                          train_data_removed, train_data_report_closed, train_data_comment,
                                                          train_data_similar, train_fixers,
                                                          val_data, val_data_create, val_data_modified,
                                                          val_data_removed, val_data_report, val_data_comment,
                                                          val_data_similar, val_fixers,
                                                          test_data, test_data_create, test_data_modified,
                                                          test_data_removed, test_data_report, test_data_comment,
                                                          test_data_similar, test_fixers)  

        graph.updateMatrix()
        return graph
