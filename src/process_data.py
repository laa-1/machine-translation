import csv
import tqdm
from transformers import BertTokenizer
from parameter import *


def text2tokens(tokenizer, text):
    tokens = tokenizer.encode(text, truncation=True, max_length=tokens_max_length - 2, add_special_tokens=False)
    tokens = [bos_token_id] + tokens + [eos_token_id] + [pad_token_id for _ in range(tokens_max_length - 2 - len(tokens))]
    return tokens


def process_train_data():
    file_en = open("../dataset/train.en", mode="r", encoding="utf-8")
    file_zh = open("../dataset/train.zh", mode="r", encoding="utf-8")
    writer_en = csv.writer(open(train_en_path, mode="w", encoding="utf-8", newline=""))
    writer_zh = csv.writer(open(train_zh_path, mode="w", encoding="utf-8", newline=""))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    for line in tqdm.tqdm(file_en):
        writer_en.writerow(text2tokens(tokenizer, line.strip()))
    for line in tqdm.tqdm(file_zh):
        writer_zh.writerow(text2tokens(tokenizer, line.strip()))


if __name__ == '__main__':
    process_train_data()
