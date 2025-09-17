import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import sys

model.eval()
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

image_path = "" # ここに予測したい画像のパスを指定
orig_image = Image.open(image_path).convert("RGB")
input_tensor = transform(orig_image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)

output=outputs[0]

label_map = {
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: '+', 12: '-', 13: '*', 14: '/', 15: '='
}

# ====== 7. 描画準備 ======
draw_image = orig_image.copy()
draw = ImageDraw.Draw(draw_image)

# スコアしきい値（信頼度）を設定
score_threshold = 0.5

# フォント（macOS用、必要に応じて変更）
try:
    font = ImageFont.truetype("Arial.ttf", 32)
except:
    font = ImageFont.load_default(32)

# ====== 8. 結果の描画 ======
boxes = output["boxes"]
labels = output["labels"]
scores = output["scores"]
orig_w, orig_h = orig_image.size
x_scale = orig_w / 300
y_scale = orig_h / 300
equation=[]
for box, label, score in zip(boxes, labels, scores):
    if score >= score_threshold:
        x1, y1, x2, y2 = box.tolist()
        x1 *= x_scale
        x2 *= x_scale
        y1 *= y_scale
        y2 *= y_scale
        label_name = label_map[int(label)]
        text = f"{label_name} {score:.2f}"
        equation.append([x1,label_name])
        # テキスト位置
        text_x = x1
        text_y = y1+10  # 少し上に出す
        text_size = draw.textlength(text, font=font)
        # draw.rectangle(
        #     [text_x, text_y, text_x + text_size, text_y + 35],
        #     fill="black"  # 背景色を指定
        # )
        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        draw.text((x1, y1-30), f"{label_name} {score:.2f}", fill="red", font=font)
plt.figure(figsize=(8, 8))
plt.imshow(draw_image)
plt.axis("off")
plt.show()