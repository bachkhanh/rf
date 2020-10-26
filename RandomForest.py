import pandas as pd
import numpy as np


def get_gini_index(subsets, labels):
    n_instances = sum((len(subset) for subset in subsets))
    gini = 0.0
    for subset in subsets:
        subset_size = len(subset)
        temp = 0.0
        if subset_size > 0:
            for label in labels:
                subset_label = subset[subset.iloc[:, -1] == label]
                proportion = len(subset_label)/subset_size
                temp += proportion * proportion
            gini += (1 - temp) * (subset_size / n_instances)
    return gini


def split_tree_by_value(data, feature, value):
    subset_left = data[data[feature] < value]
    subset_right = data[data[feature] >= value]
    return subset_left, subset_right


def get_node_value(data):
    labels = list(set(data.iloc[:, -1]))
    b_feature = 0
    b_value = 0
    b_score = 10
    b_subsets = None
    temp1 = data.iloc[:, :len(data.columns) - 1]
    temp2 = data.iloc[:, -1]
    sample_data_features = temp1.sample(n=3, axis=1)
    sample_data = pd.concat([sample_data_features, temp2], axis=1)
    for feature in sample_data_features.columns:
        for __, row in sample_data.iterrows():
            subsets = split_tree_by_value(data, feature, row[feature])
            gini = get_gini_index(subsets, labels)
            if gini < b_score:
                print(feature, gini)
                b_score = gini
                b_feature = feature
                b_value = row.iloc[feature]
                b_subsets = subsets
    print('node created')
    return {'feature': b_feature, 'value': b_value, 'subsets': b_subsets}


def get_terminal_node_label(subset):
    return subset.iloc[:, -1].value_counts().idxmax()


def split_decision_tree(node, max_depth, min_sample_split, current_depth):
    subset_left, subset_right = node['subsets']
    del(node['subsets'])
    if subset_left.empty or subset_right.empty:
        node['left'] = node['right'] = get_terminal_node_label(subset_left.append(subset_right, ignore_index=True))
        return
    if current_depth == max_depth:
        node['left'] = get_terminal_node_label(subset_left)
        node['right'] = get_terminal_node_label(subset_right)
        return
    if len(subset_left) < min_sample_split:
        node['left'] = get_terminal_node_label(subset_left)
    else:
        node['left'] = get_node_value(subset_left)
        split_decision_tree(node['left'], max_depth, min_sample_split, current_depth + 1)
    if len(subset_right) < min_sample_split:
        node['right'] = get_terminal_node_label(subset_right)
    else:
        node['right'] = get_node_value(subset_right)
        split_decision_tree(node['right'], max_depth, min_sample_split, current_depth + 1)
    return node


def get_decision_tree(train_data, max_depth, min_sample_split):
    root = get_node_value(train_data)
    tree = split_decision_tree(root, max_depth, min_sample_split, 1)
    return tree


def get_rd_decision_tree(train_data, max_depth, min_sample_split):
    root = get_node_value(train_data)
    tree = split_decision_tree(root, max_depth, min_sample_split, 1)
    return tree


def predict(data_row, tree):
    if data_row[tree['feature']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(data_row, tree['left'])
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(data_row, tree['right'])
        else:
            return tree['right']


def decision_tree(train_data, test_data, max_depth, min_sample_split=5):
    tree = get_decision_tree(train_data, max_depth, min_sample_split)
    true_predictions, false_predictions = [0, 0]
    for __, data_row in test_data.iterrows():
        # print('Label=%d, Predict=%d' % (data_row.iloc[-1], predict(data_row, tree)))
        if data_row.iloc[-1] == predict(data_row, tree):
            true_predictions += 1
        else:
            false_predictions += 1
    print('True Predictions = %d, False Predictions = %d' % (true_predictions, false_predictions))


def random_forest(n_trees, train_data, test_data, max_depth, min_sample_split=5):
    rd_forest = []
    true_predictions, false_predictions = [0, 0]
    for i in range(n_trees):
        sampled_train_data = train_data.sample(n=len(train_data), replace=True)
        random_tree = get_rd_decision_tree(sampled_train_data, max_depth, min_sample_split)
        rd_forest.append(random_tree)
        print('Tree added')
    for __, data_row in test_data.iterrows():
        predict_list = [predict(data_row, tree) for tree in rd_forest]
        result_df = pd.DataFrame({'Result': predict_list})
        if result_df.value_counts().idxmax() == data_row.iloc[-1]:
            true_predictions += 1
        else:
            false_predictions += 1
    print('True Predictions = %d, False Predictions = %d' % (true_predictions, false_predictions))


if __name__ == '__main__':
    df = pd.read_csv("data_banknote_authentication.csv", header=None)
    msk = np.random.rand(len(df)) < 0.7
    rd_features = np.random.rand(len(df)) < 0.7
    train_data_df = df[msk]
    test_data_df = df[~msk]

    # print(train_data_df.head(5))
    # print(test_data_df.head(5))

    random_forest(n_trees=3, train_data=train_data_df, test_data=test_data_df, max_depth=3)
    # decision_tree(train_data_df, test_data_df, 3)

    # test_tree = get_decision_tree(df, 4, 100)
    # print(test_tree)

    # df_data = df.iloc[:, :len(df.columns) - 1]
    # print(df_data)
    # for i in range(10):
    #     print(df_data.sample(n=3, axis=1, replace=True))
