import torch
from torch.utils.data import DataLoader
from torchvision import transforms
transform = transforms.ToTensor()

dataset = CustomVOCDataset(root="merged_dataset", image_set="train", transforms=transform)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00023082965571758206)


# 学習ループ
model.train()
for epoch in range(4):
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        # データをデバイスに移動
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 勾配をゼロに
        optimizer.zero_grad()

        try:
            # 順伝播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # NaNチェック
            if torch.isnan(losses):
                print(f"NaN detected at epoch {epoch+1}, batch {batch_idx}")
                print(f"Loss dict: {loss_dict}")
                break

            # 逆伝播
            losses.backward()

            # 勾配クリッピング（重要）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #スケールの更新
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        except Exception as e:
            print(f"Error at epoch {epoch+1}, batch {batch_idx}: {e}")
            continue

    avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")