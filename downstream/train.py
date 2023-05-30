import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.dataset import TimeSeriesDataset
from models.tempCNN import TemporalCNN
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch
import pandas as pd
import os
import sklearn.metrics
import random


def train(args):
        
    # Set a seed for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = TimeSeriesDataset(path=args.path)

    # Define the sizes for train, test, and validation sets
    train_ratio = 0.3  # 30% of the dataset for training
    test_ratio = 0.4   # 10% of the dataset for testing
    val_ratio = 0.3    # 10% of the dataset for validation

    # Calculate the sizes of each set
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)
    val_size = len(dataset) - train_size - test_size

    # Split the dataset into train, test, and validation sets
    train_ds, test_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    # Create data loaders for each set
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=args.num_workers, prefetch_factor=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=args.num_workers, prefetch_factor=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=args.num_workers, prefetch_factor=2)

    num_classes = 2
    # define model
    device = torch.device(args.device)
    model = get_model(args.mode, num_classes, device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.modelname += f"_learning-rate={args.learning_rate}_weight-decay={args.weight_decay}"
    print(f"Initialized {model.modelname}")

    logdir = os.path.join(args.logdir, model.modelname)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    log = []
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_dl, device)
        val_loss, y_true_val, y_pred_val, *_ = test_epoch(model, criterion, val_dl, device)
        val_scores = metrics(y_true_val.cpu(), y_pred_val.cpu())
        val_scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in val_scores.items()])
        val_loss = val_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"epoch {epoch}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f} " + val_scores_msg)

        scores = {}
        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["val_loss"] = val_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Load the best model for testing
    model.load_state_dict(best_model)

    # Perform the final evaluation on the test set
    test_loss, y_true_test, y_pred_test, *_ = test_epoch(model, criterion, test_dl, device)
    test_scores = metrics(y_true_test.cpu(), y_pred_test.cpu())
    test_scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in test_scores.items()])
    test_loss = test_loss.cpu().detach().numpy()[0]
    print(f"Final test_loss={test_loss:.2f} " + test_scores_msg)

    test_scores["test_loss"] = test_loss
    test_log = pd.DataFrame([test_scores])
    test_log.to_csv(os.path.join(logdir, "testlog.csv"))

    val_log_df = pd.DataFrame(log).set_index("epoch")
    val_log_df.to_csv(os.path.join(logdir, "vallog.csv"))



def get_model(mode, num_classes, device, pretrained_path=None, **hyperparameter):
    mode = mode.lower()  # make case invariant
    
    if mode == "supervised":
        model = TemporalCNN(num_classes=num_classes, **hyperparameter).to(device)
        
    elif mode == "freeze" or mode == "fine-tuning":
        model = TemporalCNN(num_classes=num_classes, **hyperparameter).to(device)
        
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path))  # Load pretrained model weights
        
        if mode == "freeze":
            for param in model.parameters():
                param.requires_grad = False  # Freeze the model parameters
    
    else:
        raise ValueError("Invalid model argument. Choose from 'supervised', 'freeze', or 'fine-tuning'.")

    return model


def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    losses = []
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x1, x2, y_true = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            y_true = y_true.to(device)

            # Forward pass
            logits = model(x1, x2)
            logprobabilities = torch.log_softmax(logits, dim=-1)

            # Compute loss
            loss = criterion(logprobabilities, y_true)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Record loss
            losses.append(loss.item())

            # Update progress bar
            iterator.set_description(f"train loss={loss.item():.2f}")

    return torch.stack(losses)



def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = []
        y_true_list = []
        y_pred_list = []
        y_score_list = []
        field_ids_list = []
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x1, x2, y_true, field_id = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_true = y_true.to(device)

                # Forward pass
                logits = model(x1, x2)
                logprobabilities = torch.log_softmax(logits, dim=-1)

                # Compute loss
                loss = criterion(logprobabilities, y_true)

                iterator.set_description(f"test loss={loss.item():.2f}")
                losses.append(loss.item())
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)

        return (torch.tensor(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list))


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate tempCNN pretrained model on multimodal time series dataset'
                                                 'This script trains a model on training dataset'
                                                 'set, evaluates performance on a validation and  test set'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument("--mode", type=str, choices=["supervised", "freeze", "fine-tuning"], default="supervised", help="Mode for downstream task")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the pretrained model checkpoint file")
    parser.add_argument(
        '-b', '--batchsize', type=int, default=1024, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='number of training epochs (training on entire dataset)')
    parser.add_argument(
        '-D', '--datapath', type=str, help='directory to time series dataset and labels file')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-6, help='optimizer weight_decay (default 1e-6)')
    parser.add_argument(
        '--learning-rate', type=float, default=1e-2, help='optimizer learning rate (default 1e-2)')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="logs", help='logdir to store progress and logs')
    args = parser.parse_args()


    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()

    train(args)
