import csv
import numpy as np
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt
from translation_model import *


def load_data(path):
    file = open(path, mode="r", encoding="utf-8")
    data = list()
    pbar = tqdm.tqdm(csv.reader(file))
    pbar.set_description("Loading data from {}".format(path))
    for row in pbar:
        row = [int(v) for v in row]
        data.append(row)
    data = np.array(data)
    return data


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.src_list = load_data(train_en_path)
        self.tgt_list = load_data(train_zh_path)
        self.count_batch = 0

    def __getitem__(self, item):
        # 训练用的tgt要去尾，计算loss的tgt要掐头
        return self.src_list[item], self.tgt_list[item][:-1], self.tgt_list[item][1:]

    def __len__(self):
        return len(self.src_list)


def train():
    model = TranslationModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = torch.utils.data.DataLoader(TrainDataset(), batch_size=batch_size, shuffle=True, num_workers=2)
    model.train(True)
    file_loss_log = open(loss_log_path, mode="w", encoding="utf-8", newline="")
    for e in range(num_epoch):
        pbar = tqdm.tqdm(data_loader, ncols=80)
        pbar.set_description("Epoch {}".format(e + 1))
        loss_log = []
        count_batch = 1
        for src, tgt, tgt_label in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_label = tgt_label.long().to(device)
            optimizer.zero_grad()
            out = model(src, tgt)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), tgt_label.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": "{:f}".format(loss.item())})
            loss_log.append(loss.item())
            if count_batch % 10000 == 0:
                torch.save(model, model_path.format(count_batch))
            count_batch += 1
        csv.writer(file_loss_log).writerow(loss_log)


def show_loss():
    reader = csv.reader(open(loss_log_path, mode="r", encoding="utf-8"))
    loss_list = list()
    for row in reader:
        loss_list += [float(v) for v in row]
    plt.plot(list(range(1, len(loss_list) + 1)), loss_list)
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    train()
    show_loss()
