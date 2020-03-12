import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_train_test_annotations(documents):
    count_dict = {}
    for ann_list in documents['annotations'].to_numpy():
        for start, end, label in ann_list:
            if label not in count_dict.keys():
                count_dict[label] = 1
            else:
                count_dict[label] += 1
    return count_dict


def plot_distribution(documents):

    train_docs = documents[documents.test == False]
    test_docs = documents[documents.test == True]

    c = documents.groupby(['test', 'corpus']).count()
    print(c)
    c.to_excel('train_and_test_corpus_distribution.xlsx')

    def get_ditribution(docs):
        counts = docs.groupby(['corpus']).count()['test']
        print(counts)

        count_dict = {}
        for c in counts.iteritems():
            for corpus in c[0].strip('][').split(' '):
                if corpus not in count_dict.keys():
                    count_dict[corpus] = 1 * c[1]
                else:
                    count_dict[corpus] += 1 * c[1]

        print(count_dict)
        return count_dict

    print('TRAIN DATA' )
    print()
    train_count_dict = get_ditribution(train_docs)
    print()
    print('TEST DATA')
    print()
    test_count_dict = get_ditribution(test_docs)
    print()

    x = np.arange(len(train_count_dict.keys()))
    width = 0.3

    fig, ax = plt.subplots()


    rects1 = ax.bar(x - width / 2, [value for (key, value) in sorted(train_count_dict.items())], width, label='Train')
    rects2 = ax.bar(x + width / 2, [value for (key, value) in sorted(test_count_dict.items())], width, label='Test')

    plt.xticks(x, sorted(train_count_dict.keys()))
    ax.legend()
    plt.show()
    return train_count_dict, test_count_dict



def plot_annotations(annotations):
    count_dict = {}
    counts = annotations.groupby(['type', 'corpus']).count()['doc']
    print(counts)
    counts.to_excel('annotations_distribution.xlsx')

def plot_training(score_df):

    iterations = range(len(score_df))
    span_precision = score_df['span_precision']
    span_recall = score_df['span_recall']
    span_f1 = score_df['span_f1']

    type_precision = score_df['type_precision']
    type_recall = score_df['type_recall']
    type_f1 = score_df['type_f1']

    fig, ax = plt.subplots()
    ax.plot(iterations, span_precision, label = 'Span Precision')
    ax.plot(iterations, span_recall, label = 'Span Recall')
    ax.plot(iterations, span_f1, label='Span F1')
    plt.xlabel('Number of Iterations')
    plt.title('Span Matching Results')
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(iterations, type_precision, label='Type Precision')
    ax2.plot(iterations, type_recall, label='Type Recall')
    ax2.plot(iterations, type_f1, label='Type F1')
    plt.xlabel('Number of Iterations')
    plt.title('Type Matching Results')
    ax2.legend()

    plt.show()


def plot_5_models_average():
    scores = []
    for i in range(5):
        scores += [pd.read_excel('all_types_model_' + str(i) + 'fold.xlsx')]

    scores = np.array([score.to_numpy() for score in scores])
    scores = scores.sum(axis = 0)
    scores = scores / 5.0
    df_scores = pd.DataFrame(scores, columns = ['index', 'span_precision', 'span_recall', 'span_f1', 'type_precision', 'type_recall', 'type_f1'])
    plot_training(df_scores)



plot_training(pd.read_excel('all_types_model_2fold.xlsx'))
plot_training(pd.read_excel('all_types_model_on_all_data.xlsx'))

