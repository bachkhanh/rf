import pandas as pd
import numpy as np

df = pd.DataFrame({'A': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': [0, 0, 0, 0, 1, 1, 1, 0]})


def get_terminal_node_label(subset):
    # list_of_labels = subset.iloc[:, -1]
    return subset.iloc[:, -1]. value_counts(). idxmax()


if __name__ == '__main__':
    # test = set(df.iloc[:, -1])
    # print(df)
    # print(test)
    # print(get_terminal_node_label(df))
    List = [1, 3,
            4, 7]
    List[::] = [value for value in List if value != 3]
    print(List)
