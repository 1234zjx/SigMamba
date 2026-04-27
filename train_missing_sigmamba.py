import argparse
import torch
from torch.utils.data import DataLoader

from dataset_missing_forecast import MissingForecastDataset
from model_missing_sigmamba import MissingAwareSigMamba


# ==========================================
# Args
# ==========================================
parser = argparse.ArgumentParser()

parser.add_argument("--csv_path", type=str, default="ETT-small/ETTh1.csv")
parser.add_argument("--seq_len", type=int, default=96)
parser.add_argument("--pred_len", type=int, default=96)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--patience", type=int, default=10)

parser.add_argument("--missing_rate", type=float, default=0.2)

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"




# ==========================================
# Masked Metrics
# ==========================================
def masked_mae(pred, target, mask):
    diff = torch.abs(pred - target)
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)


# ==========================================
# Dataset
# ==========================================
train_set = MissingForecastDataset(
    csv_path=args.csv_path,
    split="train",
    seq_len=args.seq_len,
    pred_len=args.pred_len,
    missing_rate=args.missing_rate,
)

val_set = MissingForecastDataset(
    csv_path=args.csv_path,
    split="val",
    seq_len=args.seq_len,
    pred_len=args.pred_len,
    missing_rate=args.missing_rate,
    train_mean=train_set.mean,
    train_std=train_set.std,
)

test_set = MissingForecastDataset(
    csv_path=args.csv_path,
    split="test",
    seq_len=args.seq_len,
    pred_len=args.pred_len,
    missing_rate=args.missing_rate,
    train_mean=train_set.mean,
    train_std=train_set.std,
)


train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=args.batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
)


# ==========================================
# Model
# ==========================================
model = MissingAwareSigMamba(
    num_vars=train_set.X.shape[-1],
    pred_len=args.pred_len,
    **ablation_cfg[args.ablation]
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)


# ==========================================
# Eval Function
# ==========================================
@torch.no_grad()
def evaluate(loader):
    model.eval()

    total_mae = 0
    total_mse = 0
    total_num = 0

    for x, mask, delta, y, target_mask in loader:

        x = x.to(device)
        mask = mask.to(device)
        delta = delta.to(device)
        y = y.to(device)
        target_mask = target_mask.to(device)

        pred = model(x, mask, delta)

        mae = masked_mae(pred, y, target_mask)
        mse = masked_mse(pred, y, target_mask)

        bs = x.size(0)

        total_mae += mae.item() * bs
        total_mse += mse.item() * bs
        total_num += bs

    return (
        total_mae / total_num,
        total_mse / total_num
    )


# ==========================================
# Training Loop
# ==========================================
best_val_mae = float("inf")
patience_counter = 0

save_path = f"best_{args.ablation}.pth"

print("=" * 60)
print(f"Dataset Path : {args.csv_path}")
print(f"Missing Rate : {args.missing_rate}")
print(f"Seq Len      : {args.seq_len}")
print(f"Pred Len     : {args.pred_len}")
print(f"Running Ablation: {args.ablation}")
print(f"Train Windows: {len(train_set)}")
print(f"Val Windows  : {len(val_set)}")
print(f"Test Windows : {len(test_set)}")
print(f"Num Vars     : {train_set.X.shape[-1]}")
print("=" * 60)

for epoch in range(1, args.epochs + 1):

    model.train()

    train_mae_sum = 0
    train_mse_sum = 0
    total_num = 0

    for x, mask, delta, y, target_mask in train_loader:

        x = x.to(device)
        mask = mask.to(device)
        delta = delta.to(device)
        y = y.to(device)
        target_mask = target_mask.to(device)

        pred = model(x, mask, delta)

        mae = masked_mae(pred, y, target_mask)
        mse = masked_mse(pred, y, target_mask)

        loss = mae

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        bs = x.size(0)

        train_mae_sum += mae.item() * bs
        train_mse_sum += mse.item() * bs
        total_num += bs

    train_mae = train_mae_sum / total_num
    train_mse = train_mse_sum / total_num

    val_mae, val_mse = evaluate(val_loader)

    print(
        f"[Epoch {epoch:03d}] "
        f"Train MAE: {train_mae:.6f} | "
        f"Train MSE: {train_mse:.6f} || "
        f"Val MAE: {val_mae:.6f} | "
        f"Val MSE: {val_mse:.6f}"
    )

    # ======================================
    # Early Stopping
    # ======================================
    if val_mae < best_val_mae:

        best_val_mae = val_mae
        patience_counter = 0

        torch.save(
            model.state_dict(),
            save_path
        )

        print(
            f"  >>> Best model saved "
            f"to {save_path} "
            f"(Val MAE={best_val_mae:.6f})"
        )

    else:

        patience_counter += 1

        print(
            f"  >>> Patience "
            f"{patience_counter}/{args.patience}"
        )

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break


# ==========================================
# Final Test
# ==========================================
print("\nLoading Best Model...")

model.load_state_dict(
    torch.load(save_path)
)

test_mae, test_mse = evaluate(test_loader)

print("\n================ FINAL TEST ================")
print(f"Best Val MAE : {best_val_mae:.6f}")
print(f"Test MAE     : {test_mae:.6f}")
print(f"Test MSE     : {test_mse:.6f}")
print("===========================================")
result_path = f"test_result_{args.ablation}_mr{args.missing_rate}.txt"

with open(result_path, "w") as f:
    f.write("=========== EXPERIMENT RESULT ===========\n")
    f.write(f"Ablation       : {args.ablation}\n")
    f.write(f"Missing Rate   : {args.missing_rate}\n")
    f.write(f"Seq Len        : {args.seq_len}\n")
    f.write(f"Pred Len       : {args.pred_len}\n")
    f.write("----------------------------------------\n")
    f.write(f"Best Val MAE   : {best_val_mae:.6f}\n")
    f.write(f"Test MAE       : {test_mae:.6f}\n")
    f.write(f"Test MSE       : {test_mse:.6f}\n")
    f.write("========================================\n")

print(f"\nResults saved to {result_path}")
