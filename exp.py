import numpy as np
import pandas as pd
import os
import sklearn
import random
import glob
from dataloader import CustomDataset
from models import *
from matplotlib import pyplot as plt
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from collections import OrderedDict

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        random.seed(self.args.seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
            print(f"GPU 개수: {torch.cuda.device_count()}")
            print(f"현재 사용 중인 GPU: {torch.cuda.current_device()}")
            print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("현재 GPU를 사용할 수 없습니다.")

        self.model_path = os.path.join('models', self.args.ex_name, self.args.model_name, self.args.mode)
        os.makedirs(self.model_path, exist_ok=True)

        ### model
        if self.args.model_name == 'TSMIXER':
            self.model = TSMixer(input_length=self.args.inp_seq,
                                input_channels= self.args.inp_features,
                                hidden_dim=self.args.hidden_size,
                                output_length=self.args.tar_seq,
                                n_blocks=self.args.n_blocks).to(self.device)
        elif self.args.model_name == "MLP":
            self.model = MLP(input_length=self.args.inp_seq,
                             input_channels= self.args.inp_features,
                             hidden_dim=self.args.hidden_size,
                             output_length=self.args.tar_seq,
                             num_layers=self.args.n_layers,
                             dropout_rate=self.args.dropout).to(self.device)
        elif self.args.model_name == "ResidualMLP":
            self.model = ResidaulMLP(input_length=self.args.inp_seq,
                             input_channels= self.args.inp_features,
                             hidden_dim=self.args.hidden_size,
                             output_length=self.args.tar_seq,
                             num_layers=self.args.n_layers,
                             dropout_rate=self.args.dropout).to(self.device)

    def prepare_data(self):
        if self.args.flag == 'train':
            train_dataset = CustomDataset(mode=self.args.mode, flag='train', data_idx=self.args.data_idx)
            val_dataset = CustomDataset(mode=self.args.mode, flag='valid', data_idx=self.args.data_idx)
            return train_dataset, val_dataset

        else:
            test_dataset = CustomDataset(mode=self.args.mode, flag='test', data_idx=self.args.data_idx)
            return test_dataset

    def custom_metric(self, pred, tar, alpha=0.5, eps=1e-6):
        weight = 1.0 + alpha * tar
        loss = weight * (pred - tar) ** 2
        return loss.mean()
        
    def model_train(self):
        train_dataset, val_dataset = self.prepare_data()
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False)
        print('train: ', len(train_loader), 'val: ', len(val_loader))
        model = self.model
        summary(model)
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        train_loss = []
        val_loss = []
        val_min = float('inf')

        for epoch in range(1, self.args.epochs+1):
            running_loss = 0.0
            model.train()
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            train_loss.append(epoch_loss)
            val_loss.append(self.validate(model, val_loader, criterion=nn.MSELoss()))
            print(f"Epoch [{epoch}/{self.args.epochs}] | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss[-1]:.5f}")

            # Save the model every 5 epoch
            if epoch % 5 == 0:
                self.model_save(model, epoch)

            # Save the model if the validation loss decreases
            if val_loss[-1] < val_min:
                val_min = val_loss[-1]
                self.best_model_save(model)
                print(f"Model saved at epoch {epoch} with val loss: {val_min:.5f}")

        # plt.plot(train_loss, label='train')
        # plt.plot(val_loss, label='val')
        # plt.legend()
        # plt.show()
        # plt.close()

    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_loss = val_loss / len(val_loader)
        return avg_loss

    def model_save(self, model, epoch):
        save_path = os.path.join(self.model_path, f"model_state_dict_{epoch:03d}.pth")
        torch.save(model, save_path)

    def best_model_save(self, model):
        save_path = os.path.join(self.model_path, "best_model_state_dict.pth")
        torch.save(model.state_dict(), save_path)

    def model_test(self):
        test_dataset = self.prepare_data()
        test_loader = DataLoader(test_dataset, batch_size=self.args.val_batch_size, shuffle=False)

        load_path = os.path.join(self.model_path, "best_model_state_dict.pth")

        model = self.model
        model.load_state_dict(torch.load(load_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # 평가 모드 전환
        print(f"Model loaded from {load_path}")

        pred = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                outputs = model(x)
                pred.append(outputs.detach().cpu())
        pred = torch.cat(pred, dim=0).numpy()
        print(pred.shape)
        os.makedirs(f'{self.args.res_dir}/{self.args.ex_name}/{self.args.model_name}/', exist_ok=True)
        np.save(f'{self.args.res_dir}/{self.args.ex_name}/{self.args.model_name}/{self.args.mode}_pred.npy', pred)

        print(f"The npy result file is saved | filename: {self.args.res_dir}/{self.args.ex_name}/{self.args.model_name}/{self.args.mode}_pred.npy")

        if self.args.save_model:
            scripted_model = torch.jit.script(model)
            scripted_model.save(f"{self.model_path}/{self.args.ex_name}_model.pt")
            print(f"Model saved to {self.args.ex_name}_model.pt")
