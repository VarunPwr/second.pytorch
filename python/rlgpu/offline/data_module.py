import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split


class TinyDataModule(pl.LightningDataModule):

    def __init__(self, file_name, batch_size=128, input_dict=None, output_dict=None):
        super().__init__()
        self.batch_size = batch_size
        self.input_dict = input_dict
        self.output_dict = output_dict
        data_dict = np.load("{}.npy".format(file_name), allow_pickle=True).item()
        self.load_dict(data_dict)
        self.input_size = self.tensor_data_dict["input"].shape[-1]
        self.output_size = self.tensor_data_dict["output"].shape[-1]
        self.tensor_data = torch.cat([self.tensor_data_dict["input"], self.tensor_data_dict["output"]], dim=-1)
        self.total_num = self.tensor_data.shape[0]
        self.train_val_num = int(0.9 * self.total_num)
        self.train_num = int(0.7 * self.total_num)
        self.train_data = self.tensor_data[:self.train_val_num]
        self.test_data = self.tensor_data[self.train_val_num:]

    def load_dict(self, dict):
        self.tensor_data_dict = {"input": [], "output": []}
        for k, v in dict.items():
            if k in self.input_dict:
                if len(np.shape(v)) == 1:
                    v = np.unsqueeze(v, -1)
                elif len(np.shape(v)) == 3:
                    v = v.reshape(v.shape[0], v.shape[1] * v.shape[2])
                self.tensor_data_dict["input"].append(torch.as_tensor(v))
            elif k in self.output_dict:
                if len(np.shape(v)) == 1:
                    v = np.unsqueeze(v, -1)
                elif len(np.shape(v)) == 3:
                    v = v.reshape(v.shape[0], v.shape[1] * v.shape[2])
                self.tensor_data_dict["output"].append(torch.as_tensor(v))
        self.tensor_data_dict["input"] = torch.cat(
            self.tensor_data_dict["input"], dim=-1)
        self.tensor_data_dict["output"] = torch.cat(
            self.tensor_data_dict["output"], dim=-1)

    def setup(self, stage):
        self.test_set = TensorDataset(self.test_data)
        train_data_full = TensorDataset(self.train_data)
        self.train_set, self.val_set = random_split(
            train_data_full, [self.train_num, self.train_val_num - self.train_num])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
