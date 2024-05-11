import numpy as np
from tqdm import tqdm
import torch


def train(model, criterion, optimizer, data_loader, device):
    model.train()
    loss_val = 0
    for seq, genres, labels in data_loader:
        seq, genres, labels = seq.to(device), genres.to(device), labels.to(device)
        logits = model(seq, genres)  # (bs, t, vocab)
        logits = logits.view(-1, logits.size(-1))  # (bs * t, vocab)
        labels = labels.view(-1)  # (bs * t)

        optimizer.zero_grad()
        loss = criterion(logits, labels)

        loss_val += loss.item()

        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def evaluate(model, user_train, user_valid, max_len,
             bert4rec_dataset, make_sequence_dataset, device):
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    num_item_sample = 100

    users = [user for user in range(make_sequence_dataset.num_user)]

    for user in tqdm(users):
        seq = (user_train[user] +
               [make_sequence_dataset.num_item + 1])[-max_len:]
        genre_seq = ([make_sequence_dataset.movie_genres(i) 
                      for i in user_train[user]] + [[1]*18])[-max_len:]
        padding_len = max_len - len(seq)

        seq = [0] * padding_len + seq
        genre_seq = [[0]*18] * padding_len + genre_seq

        rated = user_train[user] + [user_valid[user]]
        items = [user_valid[user]] + bert4rec_dataset.random_neg_sampling(rated_item=rated,
                                                                          num_item_sample=num_item_sample)

        with torch.no_grad():
            genre_seq = torch.Tensor([genre_seq]).to(device)
            seq = torch.LongTensor([seq]).to(device)

            predictions = -model(seq, genre_seq)
            predictions = predictions[0][-1][items]
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10:  # Top10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    NDCG /= len(users)
    HIT /= len(users)

    return NDCG, HIT
