import pickle
import numpy as np
import pandas as pd
import torch
import sys


def val_tes_data(chunks, added_relations_dict, modified_relations_dict, removed_relations_dict, report_relations_dict,
                 comment_relations_dict, issue_chunks, similar_relations_dict, issue_fixer, issue_nodes_dict):
    test_created_file_relations = added_relations_dict[chunks]
    test_modified_file_relations = modified_relations_dict[chunks]
    test_removed_file_relations = removed_relations_dict[chunks]
    test_created_file_relations_df = pd.DataFrame(data=test_created_file_relations,
                                                  columns=['developer', 'file', 'weight'])
    test_modified_file_relations_df = pd.DataFrame(data=test_modified_file_relations,
                                                   columns=['developer', 'file', 'weight'])
    test_removed_file_relations_df = pd.DataFrame(data=test_removed_file_relations,
                                                  columns=['developer', 'file', 'weight'])
    # developer-issue
    test_report_issue_relations = report_relations_dict[chunks]
    test_comment_relations = comment_relations_dict[chunks]
    test_report_issue_relations_df = pd.DataFrame(data=test_report_issue_relations, columns=['developer', 'issue'])
    test_report_issue_relations_df['weight'] = 1.0
    test_comment_relations_df = pd.DataFrame(data=test_comment_relations, columns=['developer', 'issue'])
    test_comment_relations_df['weight'] = 1.0
    test_data = issue_chunks[chunks]
    # issue-file
    test_issue_similar_file_relation = similar_relations_dict[chunks]
    test_issue_similar_file_relation_df = pd.DataFrame(data=test_issue_similar_file_relation,
                                                       columns=['issue', 'file', 'weight'])

    test_fixers = {}
    test_issue_node_id = []
    for _, issue in test_data.iterrows():
        test_fixers[issue['_id']] = issue_fixer[issue_nodes_dict[issue['_id']]]
        test_issue_node_id.append(issue_nodes_dict[issue['_id']])
    test_data['issue_node_id'] = test_issue_node_id

    return (test_data, test_created_file_relations_df, test_modified_file_relations_df, test_removed_file_relations_df,
            test_report_issue_relations_df, test_comment_relations_df, test_issue_similar_file_relation_df, test_fixers)


def getOriginData(project, chunks, datafolder=f'./data/'):
    with open(f"{datafolder}{project}/preprocessed_issue_chunks", 'rb') as f:
        issue_chunks = pickle.load(f)
    with open(f"{datafolder}{project}/added_relations_dict", 'rb') as f:
        added_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/modified_relations_dict", 'rb') as f:
        modified_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/removed_relations_dict", 'rb') as f:
        removed_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/report_closed_relations_dict", 'rb') as f:
        report_closed_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/report_relations_dict", 'rb') as f:
        report_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/comment_relations_dict", 'rb') as f:
        comment_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/issue_similar_file_relation", 'rb') as f:
        similar_relations_dict = pickle.load(f)
    with open(f"{datafolder}{project}/issue_nodes_dict.pkl", 'rb') as f:
        issue_nodes_dict = pickle.load(f)
    with open(f"{datafolder}{project}/file_nodes_dict", 'rb') as f:
        file_nodes_dict = pickle.load(f)
    with open(f"{datafolder}{project}/ten_commits_preprocessed_codes", 'rb') as f:
        preprocessed_codes = pickle.load(f)
    with open(f"{datafolder}{project}/final_version_commit", 'rb') as f:
        final_version_commit = pickle.load(f)
    with open(f"{datafolder}{project}/issue_fixer", 'rb') as f:
        issue_fixer = pickle.load(f)
    with open(f"{datafolder}{project}/issue_locations", 'rb') as f:
        pathDict = pickle.load(f)

    train_data = []
    train_data_create = []
    train_data_modified = []
    train_data_removed = []
    train_data_report_closed = []
    train_data_comment = []
    train_data_similar = []
    for chunk_index in range(chunks-1):
        # developer-file
        dev_created_file_relations = added_relations_dict[chunk_index]
        dev_modified_file_relations = modified_relations_dict[chunk_index]
        dev_removed_file_relations = removed_relations_dict[chunk_index]
        dev_created_file_relations_df = pd.DataFrame(data=dev_created_file_relations, columns=['developer', 'file', 'weight'])
        dev_modified_file_relations_df = pd.DataFrame(data=dev_modified_file_relations, columns=['developer', 'file', 'weight'])
        dev_removed_file_relations_df = pd.DataFrame(data=dev_removed_file_relations, columns=['developer', 'file', 'weight'])
        # developer-issue
        dev_report_closed_issue_relations = report_closed_relations_dict[chunk_index]
        dev_comment_relations = comment_relations_dict[chunk_index]
        dev_report_closed_issue_relations_df = pd.DataFrame(data=dev_report_closed_issue_relations, columns=['developer', 'issue'])
        dev_report_closed_issue_relations_df['weight'] = 1.0
        dev_comment_relations_df = pd.DataFrame(data=dev_comment_relations, columns=['developer', 'issue'])
        dev_comment_relations_df['weight'] = 1.0
        # issue-file
        issue_similar_file_relation = similar_relations_dict[chunk_index]
        issue_similar_file_relation_df = pd.DataFrame(data=issue_similar_file_relation, columns=['issue', 'file', 'weight'])

        train_data_create.append(dev_created_file_relations_df)
        train_data_modified.append(dev_modified_file_relations_df)
        train_data_removed.append(dev_removed_file_relations_df)
        train_data_report_closed.append(dev_report_closed_issue_relations_df)
        train_data_comment.append(dev_comment_relations_df)
        train_data_similar.append(issue_similar_file_relation_df)
        train_data.append(issue_chunks[chunk_index])

    train_data_create = pd.concat(train_data_create, ignore_index=True)
    train_data_create = train_data_create.drop_duplicates()  
    train_data_modified = pd.concat(train_data_modified, ignore_index=True)
    train_data_modified = train_data_modified.drop_duplicates()
    train_data_removed = pd.concat(train_data_removed, ignore_index=True)
    train_data_removed = train_data_removed.drop_duplicates()
    train_data_report_closed = pd.concat(train_data_report_closed, ignore_index=True)
    train_data_report_closed = train_data_report_closed.drop_duplicates()
    train_data_comment = pd.concat(train_data_comment, ignore_index=True)
    train_data_comment = train_data_comment.drop_duplicates()
    train_data_similar = pd.concat(train_data_similar, ignore_index=True)
    train_data_similar = train_data_similar.drop_duplicates()
    train_data = pd.concat(train_data, ignore_index=True)

    train_fixers = {}
    issue_node_id = []
    for _, issue in train_data.iterrows():
        train_fixers[issue['_id']] = issue_fixer[issue_nodes_dict[issue['_id']]]
        issue_node_id.append(issue_nodes_dict[issue['_id']])
    train_data['issue_node_id'] = issue_node_id

    (val_data, val_created_file_relations_df, val_modified_file_relations_df, val_removed_file_relations_df,
     val_report_issue_relations_df, val_comment_relations_df, val_issue_similar_file_relation_df, val_fixers) \
        = val_tes_data(chunks-1, added_relations_dict, modified_relations_dict, removed_relations_dict, report_relations_dict,
                 comment_relations_dict, issue_chunks, similar_relations_dict, issue_fixer, issue_nodes_dict)

    (test_data, test_created_file_relations_df, test_modified_file_relations_df, test_removed_file_relations_df,
     test_report_issue_relations_df, test_comment_relations_df, test_issue_similar_file_relation_df, test_fixers) \
        = val_tes_data(chunks, added_relations_dict, modified_relations_dict, removed_relations_dict,
                       report_relations_dict,
                       comment_relations_dict, issue_chunks, similar_relations_dict, issue_fixer, issue_nodes_dict)


    return (train_data, train_data_create, train_data_modified, train_data_removed, train_data_report_closed, train_data_comment, train_data_similar, train_fixers, pathDict,
            val_data, val_created_file_relations_df, val_modified_file_relations_df, val_removed_file_relations_df, val_report_issue_relations_df, val_comment_relations_df, val_issue_similar_file_relation_df, val_fixers,
            test_data, test_created_file_relations_df, test_modified_file_relations_df, test_removed_file_relations_df, test_report_issue_relations_df, test_comment_relations_df, test_issue_similar_file_relation_df, test_fixers)


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret

def accuracy_at_one_multi_label(scores, labels, K=10):
    _, predicted_indices = torch.max(scores, dim=1)

    correct_predictions = 0
    for predicted_idx, positive_indices in zip(predicted_indices, labels):
        if predicted_idx.item() in positive_indices:
            correct_predictions += 1
    accuracy = correct_predictions / scores.shape[0]

    reciprocal_rank = []
    at_k = [0] * K  
    num_samples = scores.shape[0]
    for i in range(num_samples):
        candidate_score = {dev_id: score for dev_id, score in enumerate(scores[i])}
        sorted_candidate_score = dict(sorted(candidate_score.items(), key=lambda item: item[1], reverse=True))
        ranked_candidates = list(sorted_candidate_score.keys())
        ranked_candidates_index = {dev_id: rank_index for rank_index, dev_id in enumerate(ranked_candidates)}
        relevant_indices = labels[i]
        if len(relevant_indices) > 0:  
            top_rank = min([ranked_candidates_index[dev_id] for dev_id in relevant_indices])
            reciprocal_rank.append(1 / (top_rank+1))
            if (top_rank + 1) <= K:
                at_k[top_rank] += 1

    mrr = sum(reciprocal_rank) / num_samples
    hit_k = [sum(at_k[:k + 1]) / num_samples for k in range(K)]

    return accuracy, mrr, hit_k  

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass