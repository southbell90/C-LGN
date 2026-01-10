## 실행 방법
root directory(C-LGN)에서 
```bash
python -m convdifflogic_pytorch.train_example --dataset mnist --epochs 1 --k 8
```
입력해서 실행한다.

## CUDA extension 빌드 (최적화)
```bash
pip install -e .
```
을 사용해 설치하고 실행하면 CUDA extension이 기본 적용된다.
CUDA extension이 없으면 자동으로 vanilla pytorch 구현이 사용된다.

CUDA Extension을 끄고 싶으면
```bash
CONVDIFFLOGIC_DISABLE_CUDA_EXT=1 python -m convdifflogic_pytorch.train_example --dataset mnist --epochs 1 --k 8
```

## arguments
- **--dataset**, choices=["mnist", "cifar10"], default="mnist"
- **--data-root**, type=str, default="./data"
- **--epochs**, type=int, default=1
- **--batch-size**, type=int, default=32
- **--lr, type=float**, default=1e-2
- **--weight-decay**, type=float, default=0.0
- **--device**, type=str, default="cuda" if torch.cuda.is_available() else "cpu"
- **--seed**, type=int, default=0
- **--k**, type=int, default=16, {S : 32, M : 256, B : 512, L : 1024, G : 2048 }
- **--tqdm**, default=True
- **--log-interval**, type=int, default=50,


