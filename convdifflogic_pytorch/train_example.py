from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .models import (
    LogicTreeNetCIFAR10,
    LogicTreeNetCIFARConfig,
    LogicTreeNetMNIST,
    LogicTreeNetMNISTConfig,
)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=16)
    p.add_argument(
        "--tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars during training/eval (default: True).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="How often (in steps) to refresh the tqdm postfix (default: 50).",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    if args.dataset == "mnist":
        # torchvision.transforms : 이미지 전처리를 위한 변환 함수/클래스들이 있다. (Resize, Crop, Normalize, ToTensor 등) 
        # transforms.ToTensor()
        ## ToTensor()는 입력 이미지를 PyTorch 텐서로 변환하는 객체
        ## 픽셀 값을 보통 0~255 정수 값 -> 0.0 ~ 1.0 실수 값으로 스케일링
        ## 차원 순서가 (H, W, C) -> (C, H, W) 형태로 변환
        # CIFAR-10 한 장 크기 32x32, 채널 3 --> ToTensor() 적용 후에 (3, 32, 32), torch.float32, [0, 1] 범위로 변환된다.
        # train_ds는 데이터 자체를 표현하는 객체, len(train_ds)는 데이터 샘플 개수, train_ds[i] : i번째 샘플을 하나 꺼내줌 -> (image_tensor, label) 
        tfm = transforms.ToTensor()
        train_ds = datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(args.data_root, train=False, download=True, transform=tfm)

        print(f"mnist train_ds length : {len(train_ds)}")
        print(f"mnist test_ds length : {len(test_ds)}")

        cfg = LogicTreeNetMNISTConfig(k=args.k, seed=args.seed)
        model = LogicTreeNetMNIST(cfg).to(device)

    else:
        tfm = transforms.ToTensor()
        train_ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)

        # For CIFAR-10, the paper uses much larger k (32..2048) and AdamW with weight decay.
        cfg = LogicTreeNetCIFARConfig(k=args.k, seed=args.seed)
        model = LogicTreeNetCIFAR10(cfg).to(device)

    # DataLoader(train_ds, batch_size=..., ...) : train_ds 기반으로 해서 batch 단위로 뽑아주는 반복자
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"train_loader length : {len(train_loader)}")
    print(f"test_loader length : {len(test_loader)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Helpful to show when the model silently auto-adjusted outputs_per_class
    # (see models.py for the robustness fix).
    if hasattr(model, "outputs_per_class_effective"):
        eff = int(getattr(model, "outputs_per_class_effective"))
        if eff != int(getattr(cfg, "outputs_per_class", eff)):
            print(f"[info] outputs_per_class was increased to {eff} to satisfy connectivity constraints")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        if args.tqdm:
            pbar = tqdm(
                enumerate(train_loader, start=1),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{args.epochs} [train]",
                dynamic_ncols=True,
            )
        else:
            pbar = enumerate(train_loader, start=1)

        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_correct += (logits.detach().argmax(dim=-1) == y).sum().item()
            total_seen += bs

            if args.tqdm and (step % args.log_interval == 0 or step == len(train_loader)):
                pbar.set_postfix(
                    loss=f"{(total_loss / max(total_seen, 1)):.4f}",
                    acc=f"{(total_correct / max(total_seen, 1)):.4f}",
                )

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        model.eval()
        test_correct = 0
        test_seen = 0
        if args.tqdm:
            pbar2 = tqdm(
                test_loader,
                total=len(test_loader),
                desc=f"Epoch {epoch}/{args.epochs} [eval]",
                dynamic_ncols=True,
                leave=False,
            )
        else:
            pbar2 = test_loader

        with torch.no_grad():
            for x, y in pbar2:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                test_correct += (logits.argmax(dim=-1) == y).sum().item()
                test_seen += x.shape[0]

        test_acc = test_correct / max(test_seen, 1)

        # Use tqdm-friendly printing.
        msg = (
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"test acc {test_acc:.4f}"
        )
        if args.tqdm:
            tqdm.write(msg)
        else:
            print(msg)


if __name__ == "__main__":
    main()