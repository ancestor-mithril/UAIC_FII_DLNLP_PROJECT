from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def get_labels():
    return [line.rstrip() for line in open("./data/scorers/techniques_subtask3.txt", encoding='utf-8').readlines()]


def read_input_file(output_path):
    """
    Read a csv file with three columns TAB separated:
       - first column is the id of the article
       - second column is the id of the paragraph in the article
       - third column is the comma-separated list of techniques of the example
    """
    ret = {}
    for line in open(output_path, encoding='utf-8').readlines():
        cols = line.rstrip().split('\t')
        prediction_id = f"{cols[0]}_{cols[1]}"
        if len(cols) == 3:
            labels = cols[2].split(",")
        else:
            labels = []
        ret[prediction_id] = labels
    return ret


def extract_matching_lists(predicted, gold):
    """
    Extract the list of values from the two dictionaries ensuring that elements with the same key are in the same position.
    We extract based on predicted assuming that we predict for few articles but we have a big gold file with all articles.
    """
    pred_values, gold_values = ([], [])
    for k in predicted.keys():
        pred_values.append(predicted[k])
        gold_values.append(gold[k])
    return pred_values, gold_values


def evaluate(predicted, gold, classes):
    """
    :param predicted: dict with predictions
    :param gold: dict with gold labels
    :param classes: list of classes
    :return: the micro-F1 and macro-F1 score
    """
    predicted, gold = extract_matching_lists(predicted, gold)
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    predicted = mlb.transform(predicted)
    gold = mlb.transform(gold)

    micro_f1 = f1_score(gold, predicted, average="micro", zero_division=1)
    macro_f1 = f1_score(gold, predicted, average="macro", zero_division=1)
    return micro_f1, macro_f1,


def score(predicted_path, gold_path):
    predicted = read_input_file(predicted_path)
    gold = read_input_file(gold_path)
    labels = get_labels()
    micro_f1, macro_f1 = evaluate(predicted, gold, labels)
    print(f"micro-F1={micro_f1:.5f}\tmacro-F1={macro_f1:.5f}")
    return micro_f1, macro_f1


def main():
    print(get_labels())
    predicted = "./data/baselines/baseline-output-subtask3-train-it.txt"
    gold = "./data/data/it/train-labels-subtask-3.txt"
    score(predicted, gold)


if __name__ == "__main__":
    main()
