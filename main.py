import numpy as np
import torch
from torch import from_numpy, cuda, optim
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from dataset_loader import get_preprocessed, get_labels
from sklearn.model_selection import train_test_split

from lstm import LSTMMultiLabelClassifier
from scorer import evaluate


def create_tensor_dataset(df):
    text = []
    labels = []
    for i, row in df.iterrows():
        text.append(row["text"])
        labels.append(row["labels"])
    text = np.stack(text)
    labels = np.stack(labels)
    return TensorDataset(from_numpy(text), from_numpy(labels))


def get_loaders(language):
    df, word2idx = get_preprocessed(language)
    train, test = train_test_split(df, test_size=0.1)
    train_data = create_tensor_dataset(train)
    test_data = create_tensor_dataset(test)
    batch_size = 400  # TODO: parametrize
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader, word2idx


def get_device():
    if cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model(device, word2idx):
    vocab_size = len(word2idx) + 1
    embedding_dim = 400
    num_layers = 2
    hidden_dim = 256
    output_size = len(get_labels())
    model = LSTMMultiLabelClassifier(vocab_size, embedding_dim, num_layers, hidden_dim, output_size)
    model.to(device)
    return model


def score(predicted, gold, loss):
    gold_list = []
    predicted_list = []
    tp = 0
    for i in range(gold.size(0)):
        gold_list.append((gold[i] == 1).nonzero().squeeze(-1).tolist())
        predicted_list.append((predicted[i] == 1).nonzero().squeeze(-1).tolist())
        if (gold[i] == predicted[i]).all():
            tp += 1

    micro_F1, macro_F1 = evaluate(predicted_list, gold_list, [i for i in range(23)])
    print(f"\nAcc. : {tp / gold.size(0)}")
    print(f"Loss: {loss}")
    print(f"Micro_F1. : {micro_F1}")
    print(f"Macro_F1. : {macro_F1}\n")


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    predicted = []
    gold = []
    losses = []
    for inputs, labels in tqdm(train_loader):
        gold.append(labels.clone())

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        predicted.append(outputs.clone().cpu())

        loss = criterion(outputs, labels)
        loss.backward()
        losses.append(loss.clone().cpu())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    gold = torch.cat(gold)
    predicted = torch.cat(predicted).round().int()
    score(gold, predicted, torch.stack(losses).sum())
    print("Model norm", sum([torch.norm(x) for x in model.parameters()]))


def main():
    train_loader, test_loader, word2idx = get_loaders("en")
    device = get_device()
    model = get_model(device, word2idx)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=0.00005)
    for _ in range(1000):
        train(model, criterion, optimizer, train_loader, device)


if __name__ == "__main__":
    main()
