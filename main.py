import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MakeSequenceDataSet, BERTRecDataSet
from model import BERT
from tqdm import tqdm
from train_and_eval import train, evaluate


def main():
    config = {
        'data_path': '/kaggle/input/movielens-20m-dataset',
        'max_len': 80,
        'hidden_units': 256,  # Embedding
        'num_heads': 2,  # Multi-head layer
        'num_layers': 2,  # block encoder layer
        'dropout_rate': 0.1,  # dropout
        'lr': 0.001,
        'batch_size': 256,
        'num_epochs': 10,
        'num_workers': 2,
        'mask_prob': 0.15,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    make_sequence_dataset = MakeSequenceDataSet(config['data_path'])

    user_train, movie_genres, user_valid = make_sequence_dataset.get_train_valid_data()

    bert4rec_dataset = BERTRecDataSet(
        user_train=user_train,
        movie_genres=movie_genres,
        max_len=config['max_len'],
        num_user=make_sequence_dataset.num_user,
        num_item=make_sequence_dataset.num_item,
        mask_prob=config['mask_prob'],
    )

    data_loader = DataLoader(
        bert4rec_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['num_workers'],
    )

    model = BERT(
        num_items=make_sequence_dataset.num_item,
        genres_size=18,
        bert_hidden_units=config['hidden_units'],
        bert_num_heads=config['num_heads'],
        bert_num_blocks=config['num_layers'],
        bert_max_len=config['max_len'],
        bert_dropout=config['dropout_rate'],
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    loss_list = []
    for epoch in tqdm(range(1, config['num_epochs'] + 1)):
        train_loss = train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader,
            device=device
        )
        loss_list.append(train_loss)

        print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}')

    torch.save(model.state_dict(), 'weights.pth')

    model.load_state_dict(torch.load('weights.pth'))
    model.eval()

    ndcg, hit = evaluate(
        model=model,
        user_train=user_train,
        user_valid=user_valid,
        max_len=config['max_len'],
        make_sequence_dataset=make_sequence_dataset,
        bert4rec_dataset=bert4rec_dataset,
        device=device
    )

    print(f'NDCG@10: {ndcg}| HIT@10: {hit}')


if __name__ == "__main__":
    main()
