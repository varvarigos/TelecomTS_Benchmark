import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.data_utils import preprocess
from utils.train_utils import evaluate, prepare

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

task_type = config["task_type"]

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

## Train

data = []
with open("src/data/train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

seq_len = config[f"{config['encoder_type']}_model"]["seq_len"]
X_train, y_train = preprocess(data, task_type, seq_len)

model, head, train_dataset, train_dataloader, optimizer, criterion = prepare(
    config, X_train, y_train
)

tqdm.write("Training...")

model.train()
head.train()
for epoch in range(config["train"]["epochs"]):
    train_losses = []
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"
    ):
        optimizer.zero_grad()
        outputs = model(batch[0].permute(0, 2, 1))
        logits = head(outputs)
        loss = criterion(logits, batch[1])
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    tqdm.write(
        f"Epoch {epoch+1}/{config['train']['epochs']}, Loss: {np.mean(train_losses):.4f}"
    )

evaluate(model, head, train_dataset, y_train, task_type)

## Evaluate

tqdm.write("Evaluating...")

test_data = []
with open("src/data/test.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

X_test, y_test = preprocess(test_data, task_type, seq_len)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64)
)
evaluate(model, head, test_dataset, y_test, task_type)
