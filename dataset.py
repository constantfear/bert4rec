import random
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset


class MakeSequenceDataSet():
    """
    SequenceData

    Переделывет исходную таблицу (user: item) в таблицу (user: item_sequence)
    """
    def __init__(self, data_path):
        print('Reading data...')
        self.df = pd.read_csv(os.path.join(data_path, 'rating.csv'))
        self.movies = pd.read_csv(os.path.join(data_path, 'movie.csv'))
        print('Applying genres...')

        self.genres = [
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        for genre in self.genres:
            self.movies[genre] = self.movies["genres"].apply(
                lambda values: int(genre in values.split("|"))
            )

        self._movie_genres = self.movies[self.genres].to_numpy()

        print('Generate encoder and decoder...')
        self.item_encoder, self.item_decoder = self.generate_encoder_decoder(self.movies['movieId'])
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder(self.df['userId'])
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['movieId'].apply(lambda x : self.item_encoder[x] + 1)
        self.df['user_idx'] = self.df['userId'].apply(lambda x : self.user_encoder[x])

        self.df = self.df.sort_values(['user_idx', 'timestamp']) 

        print('Generate sequence data...')
        self.user_train, self.genres_seq, self.user_valid = self.generate_sequence_data()

        print('Finish!!!')

    def generate_encoder_decoder(self, col) -> dict:
        """
        encoder, decoder

        Args:
            col (str): columns
        Returns:
            dict: encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = col.unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def movie_genres(self, idx):
        return self._movie_genres[idx-1].tolist()

    def generate_sequence_data(self) -> dict:
        """
        sequence_data

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        genres_seq = {}
        user_valid = {}
        group_df = self.df.groupby('user_idx')
        for user, item in group_df:
            users[user].extend(item['item_idx'].tolist()) 

        for user in users:
            user_train[user] = users[user][:-1]
            genres_seq[user] = [self.movie_genres(i) for i in user_train[user]]
            user_valid[user] = users[user][-1]

        return user_train, genres_seq, user_valid

    def get_train_valid_data(self):
        return self.user_train, self.genres_seq, self.user_valid


class BERTRecDataSet(Dataset):
    def __init__(self, user_train, movie_genres, max_len, 
                 num_user, num_item, mask_prob):
        self.user_train = user_train
        self.movie_genres = movie_genres
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(1, self.num_item + 1)])

    def __len__(self):
        return self.num_user

    def __getitem__(self, user):
        user_seq = self.user_train[user]
        genre_seq = self.movie_genres[user]
        tokens = []
        genres_seq = []
        labels = []

        for s, g in zip(user_seq[-self.max_len:], genre_seq[-self.max_len:]):
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # mask_index: num_item + 1, 0: pad
                    tokens.append(self.num_item + 1)
                    genres_seq.append([1]*18)
                elif prob < 0.9:
                    # item random sampling
                    rnd_token = self.random_neg_sampling(rated_item=user_seq,
                                                         num_item_sample=1)[0]
                    tokens.append(rnd_token) 
                    genres_seq.append([
                        random.randint(0, 1) for _ in range(18)
                    ])
                else:
                    tokens.append(s)
                    genres_seq.append(g)
            else:
                tokens.append(s)
                genres_seq.append(g)
            labels.append(s)

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        genres_seq = [[0]*18] * mask_len + genres_seq
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.Tensor(genres_seq), torch.LongTensor(labels)

    def random_neg_sampling(self, rated_item: list, num_item_sample: int):
        nge_samples = random.sample(range(1, self.num_item + 1),
                                    num_item_sample)
        return nge_samples
