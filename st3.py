import os
import pandas as pd
from tqdm import tqdm


def make_dataframe(input_folder, labels_fn=None):
    # MAKE TXT DATAFRAME
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD = fil[7:].split('.')[0]
        lines = list(enumerate(open(input_folder + fil, 'r', encoding='utf-8').read().splitlines(), 1))
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=['id', 'line', 'text'])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text[df_text.text.str.strip().str.len() > 0].copy()
    df_text = df_text.set_index(['id', 'line'])

    df = df_text

    if labels_fn:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn, sep='\t', encoding='utf-8', header=None)
        labels = labels.rename(columns={0: 'id', 1: 'line', 2: 'labels'})
        labels = labels.set_index(['id', 'line'])
        labels = labels[labels.labels.notna()].copy()

        # JOIN
        df = labels.join(df_text)[['text', 'labels']]

    return df


def load_train_and_dev(language):
    test_dataset = make_dataframe(f"./data/data/{language}/train-articles-subtask-3/",
                                  f"./data/data/{language}/train-labels-subtask-3.txt")
    dev_dataset = make_dataframe(f"./data/data/{language}/dev-articles-subtask-3/")
    return test_dataset, dev_dataset


def main():
    df1, df2 = load_train_and_dev("en")
    print(df1)
    print(df2)


if __name__ == "__main__":
    main()
