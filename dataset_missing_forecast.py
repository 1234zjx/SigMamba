import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MissingForecastDataset(Dataset):
    def __init__(
        self,
        csv_path="ETT-small/ETTh1.csv",
        split="train",
        seq_len=96,
        pred_len=96,
        missing_rate=0.2,
        seed=42,
        train_mean=None,
        train_std=None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # =====================================
        # 1. Load Raw Data
        # =====================================
        df = pd.read_csv(csv_path)
        data_raw = df.iloc[:, 1:].values.astype(np.float32)

        T, C = data_raw.shape
        train_end = int(T * 0.6)
        val_end = int(T * 0.8)

        # =====================================
        # 2. Build Train Missing Mask
        # =====================================
        rng_train = np.random.RandomState(seed)

        full_train_mask = (
            rng_train.rand(train_end, C) > missing_rate
        ).astype(np.float32)

        # =====================================
        # 3. Train Statistics (Train Only)
        # =====================================
        if train_mean is None or train_std is None:
            train_obs = data_raw[:train_end][full_train_mask == 1]

            self.mean = train_obs.mean()
            self.std = train_obs.std() + 1e-6
        else:
            self.mean = train_mean
            self.std = train_std

        # =====================================
        # 4. Split Dataset
        # =====================================
        if split == "train":

            self.raw_part = data_raw[:train_end]
            self.mask_part = full_train_mask

        elif split == "val":

            self.raw_part = data_raw[
                train_end - seq_len: val_end
            ]

            rng_val = np.random.RandomState(seed + 1)

            self.mask_part = (
                rng_val.rand(*self.raw_part.shape) > missing_rate
            ).astype(np.float32)

        elif split == "test":

            self.raw_part = data_raw[
                val_end - seq_len:
            ]

            rng_test = np.random.RandomState(seed + 2)

            self.mask_part = (
                rng_test.rand(*self.raw_part.shape) > missing_rate
            ).astype(np.float32)

        else:
            raise ValueError(
                "split must be train / val / test"
            )

        # =====================================
        # 5. Normalize
        # =====================================
        self.data_norm = (
            self.raw_part - self.mean
        ) / self.std

        # =====================================
        # 6. Build Delta
        # =====================================
        self.delta_part = self.build_delta(
            self.mask_part
        )

        self.delta_part = (
            self.delta_part / self.seq_len
        )

        # =====================================
        # 7. Apply Missing Mask to Input
        # =====================================
        self.data_missing = self.data_norm.copy()

        self.data_missing[
            self.mask_part == 0
        ] = 0.0

        # =====================================
        # 8. Sliding Window Build
        # =====================================
        (
            self.X,
            self.M,
            self.D,
            self.Y,
            self.TM
        ) = self.build_windows()

    def build_delta(self, mask):
        """
        Delta[t,c]:
        仅在缺失时累计（mask=0），观测时为0
        """
        T, C = mask.shape
    
        delta = np.zeros((T, C), dtype=np.float32)
    
        for c in range(C):
            for t in range(1, T):
    
                if mask[t, c] == 0:  # 当前是缺失
                    delta[t, c] = delta[t - 1, c] + 1
                else:  # 当前是观测
                    delta[t, c] = 0
    
        return delta

    def build_windows(self):
        X, M, D, Y, TM = [], [], [], [], []

        T = len(self.data_missing)

        end_idx = (
            T
            - self.seq_len
            - self.pred_len
            + 1
        )

        for i in range(end_idx):

            # ==============================
            # Historical Input
            # ==============================
            X.append(
                self.data_missing[
                    i: i + self.seq_len
                ]
            )

            M.append(
                self.mask_part[
                    i: i + self.seq_len
                ]
            )

            D.append(
                self.delta_part[
                    i: i + self.seq_len
                ]
            )

            # ==============================
            # Future Forecast Target
            # ==============================
            Y.append(
                self.data_norm[
                    i + self.seq_len:
                    i + self.seq_len + self.pred_len
                ]
            )

            # ==============================
            # Future Target Mask
            # ==============================
            TM.append(
                self.mask_part[
                    i + self.seq_len:
                    i + self.seq_len + self.pred_len
                ]
            )

        return (
            np.array(X, dtype=np.float32),
            np.array(M, dtype=np.float32),
            np.array(D, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(TM, dtype=np.float32),
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.M[idx]),
            torch.from_numpy(self.D[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.TM[idx]),
        )
