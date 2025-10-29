"""
train_compare.py
- Trains SNN (readout-only) and MLP in PyTorch and plots learning curves.
- Uses SNN_torch.SNNReadoutOnly and SimpleMLP
- Quick mode (--quick) runs a single-batch smoke test for speed estimation.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from SNN_torch import SNNReadoutOnly, SimpleMLP, count_trainable_params, suggest_mlp_sizes_to_match_params

torch.backends.cudnn.benchmark = True


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size=256, num_workers=0):
    transform = transforms.Compose([transforms.ToTensor()])  # range [0,1]
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


def eval_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = xb.view(xb.size(0), -1)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, total_correct / total


def train_snn_readout(model: SNNReadoutOnly, train_loader, test_loader, device,
                      epochs=10, lr=1e-3, weight_decay=1e-4, scheduler_milestones=(7, 9), quick=False):
    optimizer = torch.optim.Adam(model.readout.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(scheduler_milestones), gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    model.to(device)

    if quick:
        model.train()
        xb, yb = next(iter(train_loader))
        xb = xb.to(device); yb = yb.to(device)
        xb = xb.view(xb.size(0), -1)
        t0 = time.perf_counter()
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0
        print(f"[SNN quick] single-batch forward/backward done time={elapsed:.3f}s loss={loss.item():.4f}")
        return history

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for i, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            xb = xb.view(xb.size(0), -1)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
            if i % 50 == 0:
                print(f"[SNN] ep {ep+1} batch {i} loss_batch={loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = eval_model(model, test_loader, device, criterion)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - t0
        print(f"[SNN] Epoch {ep+1}/{epochs} lr={history['lr'][-1]:.4g} train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% time={elapsed:.1f}s")
    return history


def train_mlp(model: SimpleMLP, train_loader, test_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4, scheduler_milestones=(7, 9), quick=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(scheduler_milestones), gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    model.to(device)

    if quick:
        xb, yb = next(iter(train_loader))
        xb = xb.to(device); yb = yb.to(device)
        xb = xb.view(xb.size(0), -1)
        t0 = time.perf_counter()
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0
        print(f"[MLP quick] single-batch forward/backward done time={elapsed:.3f}s loss={loss.item():.4f}")
        return history

    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for i, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            xb = xb.view(xb.size(0), -1)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
            if i % 100 == 0:
                print(f"[MLP] ep {ep+1} batch {i} loss_batch={loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = eval_model(model, test_loader, device, criterion)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - t0
        print(f"[MLP] Epoch {ep+1}/{epochs} lr={history['lr'][-1]:.4g} train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% time={elapsed:.1f}s")
    return history


def plot_history(history_snn, history_mlp, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    epochs = len(history_snn["train_loss"])
    x = np.arange(1, epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, history_snn["train_loss"], label="SNN train")
    plt.plot(x, history_snn["val_loss"], label="SNN val")
    plt.plot(x, history_mlp["train_loss"], label="MLP train")
    plt.plot(x, history_mlp["val_loss"], label="MLP val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(x, history_snn["train_acc"], label="SNN train")
    plt.plot(x, history_snn["val_acc"], label="SNN val")
    plt.plot(x, history_mlp["train_acc"], label="MLP train")
    plt.plot(x, history_mlp["val_acc"], label="MLP val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    fname = os.path.join("results", "learning_curves.png")
    plt.savefig(fname, dpi=150)
    print("Saved learning curves to", fname)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=0)

    # SNN setup
    snn = SNNReadoutOnly(input_dim=28 * 28, hidden_dim=args.hidden_dim, output_dim=10,
                         T=args.T, input_scale=args.input_scale, input_bias=args.input_bias,
                         normalize_counts=True, center_counts=True, seed=args.seed)
    snn_params = count_trainable_params(snn.readout)
    total_snn_params = count_trainable_params(snn)
    print(f"SNN trainable params (readout): {snn_params}, total params: {total_snn_params}")

    # MLP sizes suggested to match trainable params
    h1, h2 = suggest_mlp_sizes_to_match_params(snn_params, input_dim=28 * 28, output_dim=10)
    print(f"Suggested MLP hidden sizes to match trainable params: h1={h1}, h2={h2}")
    mlp = SimpleMLP(input_dim=28 * 28, hidden1=h1, hidden2=h2, output_dim=10, dropout=0.2)
    mlp_params = count_trainable_params(mlp)
    print(f"MLP trainable params: {mlp_params}")

    if args.quick:
        print("Running quick smoke test (single-batch forward/backward) — SNN then MLP")

    start = time.perf_counter()
    print("Training SNN (readout-only)...")
    history_snn = train_snn_readout(snn, train_loader, test_loader, device,
                                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                    scheduler_milestones=args.milestones, quick=args.quick)
    t_snn = time.perf_counter() - start

    print("Training MLP...")
    history_mlp = train_mlp(mlp, train_loader, test_loader, device,
                            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                            scheduler_milestones=args.milestones, quick=args.quick)
    t_mlp = time.perf_counter() - start - t_snn

    if not args.quick:
        criterion = nn.CrossEntropyLoss()
        val_loss_snn, val_acc_snn = eval_model(snn, test_loader, device, criterion)
        val_loss_mlp, val_acc_mlp = eval_model(mlp, test_loader, device, criterion)
        print("Final results:")
        print(f"SNN val acc: {val_acc_snn*100:.2f}%  time_total={t_snn:.1f}s")
        print(f"MLP val acc: {val_acc_mlp*100:.2f}%  time_total={t_mlp:.1f}s")
        plot_history(history_snn, history_mlp, outdir=args.outdir)
    else:
        print("Quick mode finished. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024, help="hidden units for SNN hidden layer H")
    parser.add_argument("--T", type=int, default=200, help="time steps for SNN")
    parser.add_argument("--input-scale", type=float, default=60.0)
    parser.add_argument("--input-bias", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--milestones", type=int, nargs='*', default=[7, 9])
    parser.add_argument("--quick", action="store_true", help="Run a single-batch smoke test then exit")
    args = parser.parse_args()
    args.milestones = tuple(args.milestones)
    main(args)