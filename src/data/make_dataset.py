import os
import random
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

# 数式をランダム生成（例: "12+34=", "7-2="）
def generate_random_formula(min_digits=1, max_digits=2):
    def random_number():
        return ''.join(random.choices('0123456789', k=random.randint(min_digits, max_digits)))
    
    op = random.choice(['+', '-',"*","/"])
    left = random_number()
    right = random_number()
    return f"{left}{op}{right}="

# VOC形式アノテーションを保存
def save_voc_annotation(image_id, size, bboxes, labels, save_dir):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "JPEGImages"
    ET.SubElement(annotation, "filename").text = f"{image_id}.jpg"
    
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(size[0])
    ET.SubElement(size_el, "height").text = str(size[1])
    ET.SubElement(size_el, "depth").text = "3"

    for bbox, label in zip(bboxes, labels):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[3]))

    tree = ET.ElementTree(annotation)
    save_path = os.path.join(save_dir, f"{image_id}.xml")
    tree.write(save_path, encoding="utf-8", xml_declaration=True)

# フォントを読み込む関数
def load_fonts(font_paths, font_size):
    """複数のフォントパスから利用可能なフォントを読み込む"""
    fonts = []
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            fonts.append(font)
            print(f"✅ フォント読み込み成功: {font_path}")
        except Exception as e:
            print(f"❌ フォント読み込み失敗: {font_path} - {e}")
    
    # フォントが1つも読み込めなかった場合はデフォルトフォントを使用
    if not fonts:
        fonts.append(ImageFont.load_default())
        print("⚠️ デフォルトフォントを使用します")
    
    return fonts

def create_font_assignment(num_fonts, total_samples, font_counts=None):
    """
    フォント使用パターンを決定する関数
    
    Args:
        num_fonts: フォント数
        total_samples: 総サンプル数
        font_counts: フォント使用枚数のリスト [100, 50, 30] など
    
    Returns:
        font_assignment: 各サンプルに対するフォントインデックスのリスト
    """
    
    if font_counts is not None:
        # 枚数指定の場合
        print(f"📋 フォント枚数指定: {font_counts}")
        
        if len(font_counts) != num_fonts:
            raise ValueError(f"font_countsの長さ({len(font_counts)})がフォント数({num_fonts})と一致しません")
        
        if sum(font_counts) != total_samples:
            print(f"⚠️ 指定枚数の合計({sum(font_counts)})が総サンプル数({total_samples})と異なります")
            # 比率を調整
            scale = total_samples / sum(font_counts)
            font_counts = [int(count * scale) for count in font_counts]
            # 端数を調整
            diff = total_samples - sum(font_counts)
            for i in range(abs(diff)):
                if diff > 0:
                    font_counts[i % num_fonts] += 1
                else:
                    font_counts[i % num_fonts] -= 1
        
        assignment = []
        for font_idx, count in enumerate(font_counts):
            assignment.extend([font_idx] * count)
    
    else:
        # デフォルト：均等分割
        print("📋 均等分割モード")
        assignment = []
        for i in range(total_samples):
            assignment.append(i % num_fonts)
    
    # シャッフルしてランダム性を加える
    random.shuffle(assignment)
    
    return assignment

def calculate_text_dimensions(text, font):
    """テキストの描画サイズを計算（より正確な計算）"""
    # ダミー画像で文字サイズを測定
    dummy_img = Image.new("RGB", (1, 1), "white")
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # アセンダーとディセンダーを考慮した実際の高さ
    ascent, descent = font.getmetrics()
    actual_height = ascent + descent
    
    return width, actual_height

def generate_random_layout(formula, font, image_size, min_spacing=10):
    """
    文字のランダム配置を生成（順番は維持、枠内収まり保証）
    
    Args:
        formula: 数式文字列
        font: フォントオブジェクト
        image_size: 画像サイズ (width, height)
        min_spacing: 文字間の最小間隔
    
    Returns:
        positions: [(x, y, char), ...] のリスト
        bboxes: [xmin, ymin, xmax, ymax] のリスト
    """
    positions = []
    bboxes = []
    
    # 各文字のサイズを計算
    char_sizes = []
    for char in formula:
        width, height = calculate_text_dimensions(char, font)
        char_sizes.append((width, height))
    
    # 最大の文字高さを取得
    max_height = max([size[1] for size in char_sizes])
    
    # フォントのアセンダー、ディセンダーを取得
    ascent, descent = font.getmetrics()
    
    # 安全なマージンを計算（より大きく取る）
    margin = 30
    available_width = image_size[0] - 2 * margin
    available_height = image_size[1] - 2 * margin - max_height
    
    # Y座標の安全範囲を計算（アセンダーを考慮）
    y_min = margin
    y_max = image_size[1] - margin - max_height
    
    # ランダム配置パターンを選択
    layout_type = random.choice(['horizontal_random', 'slight_vertical', 'wave'])
    
    if layout_type == 'horizontal_random':
        # 水平方向にランダムな間隔で配置
        x_positions = generate_horizontal_positions(char_sizes, available_width, margin, min_spacing)
        y_base = margin + random.randint(0, max(1, available_height))
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            y = y_base + random.randint(-15, 15)  # 少し上下にブレ
            
            # Y座標を安全範囲内に制限
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            # バウンディングボックスを正確に計算
            bbox = [x, y, x + width, y + height]
            
            # 最終チェック：バウンディングボックスが画像内に収まっているか
            bbox[0] = max(0, bbox[0])  # xmin
            bbox[1] = max(0, bbox[1])  # ymin
            bbox[2] = min(image_size[0], bbox[2])  # xmax
            bbox[3] = min(image_size[1], bbox[3])  # ymax
            
            bboxes.append(bbox)
    
    elif layout_type == 'slight_vertical':
        # 軽い縦方向の変化を加えた配置
        x_positions = generate_horizontal_positions(char_sizes, available_width, margin, min_spacing)
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            # 緩やかな波形で縦位置を変化（範囲を小さく）
            y_offset = int(15 * random.uniform(-1, 1))
            y = margin + available_height // 2 + y_offset
            
            # Y座標を安全範囲内に制限
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            # バウンディングボックスを正確に計算
            bbox = [x, y, x + width, y + height]
            
            # 最終チェック
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_size[0], bbox[2])
            bbox[3] = min(image_size[1], bbox[3])
            
            bboxes.append(bbox)
    
    elif layout_type == 'wave':
        # 波形パターンで配置（振幅を小さく）
        x_positions = generate_horizontal_positions(char_sizes, available_width, margin, min_spacing)
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            # サイン波で縦位置を決定（振幅を制限）
            wave_amplitude = min(20, available_height // 6)
            y_offset = int(wave_amplitude * random.uniform(-1, 1))
            y = margin + available_height // 2 + y_offset
            
            # Y座標を安全範囲内に制限
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            # バウンディングボックスを正確に計算
            bbox = [x, y, x + width, y + height]
            
            # 最終チェック
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_size[0], bbox[2])
            bbox[3] = min(image_size[1], bbox[3])
            
            bboxes.append(bbox)
    
    return positions, bboxes

def generate_horizontal_positions(char_sizes, available_width, margin, min_spacing):
    """水平方向の位置をランダムに生成"""
    total_char_width = sum([size[0] for size in char_sizes])
    min_total_spacing = min_spacing * (len(char_sizes) - 1)
    
    if total_char_width + min_total_spacing > available_width:
        # 文字が多すぎる場合は等間隔で配置
        spacing = max(min_spacing, (available_width - total_char_width) // max(1, len(char_sizes) - 1))
        x_positions = []
        x = margin
        for width, _ in char_sizes:
            x_positions.append(x)
            x += width + spacing
    else:
        # ランダムな間隔で配置
        extra_space = available_width - total_char_width - min_total_spacing
        spacings = []
        
        for i in range(len(char_sizes) - 1):
            random_extra = random.randint(0, extra_space // max(1, len(char_sizes) - 1 - i))
            spacings.append(min_spacing + random_extra)
            extra_space -= random_extra
        
        x_positions = []
        x = margin
        for i, (width, _) in enumerate(char_sizes):
            x_positions.append(x)
            x += width
            if i < len(spacings):
                x += spacings[i]
    
    return x_positions

def generate_random_layout_dynamic(formula, font, image_size, font_size, min_spacing=5):
    """
    フォントサイズに応じて動的にマージンを調整するランダム配置
    """
    positions = []
    bboxes = []
    
    # 各文字のサイズを計算
    char_sizes = []
    for char in formula:
        width, height = calculate_text_dimensions(char, font)
        char_sizes.append((width, height))
    
    # 最大の文字高さを取得
    max_height = max([size[1] for size in char_sizes])
    
    # フォントサイズに応じてマージンを動的に調整
    margin_ratio = min(0.15, max(0.05, font_size / 1000))  # フォントサイズに応じた比率
    margin = int(image_size[1] * margin_ratio)
    margin = max(10, min(margin, 30))  # 最小10px、最大30px
    
    available_width = image_size[0] - 2 * margin
    available_height = image_size[1] - 2 * margin - max_height
    
    # Y座標の安全範囲を計算
    y_min = margin
    y_max = image_size[1] - margin - max_height
    
    # ランダム配置パターンを選択
    layout_type = random.choice(['horizontal_random', 'slight_vertical', 'wave'])
    
    if layout_type == 'horizontal_random':
        # 水平方向にランダムな間隔で配置
        x_positions = generate_horizontal_positions_dynamic(char_sizes, available_width, margin, min_spacing, font_size)
        y_base = margin + random.randint(0, max(1, available_height))
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            y_variation = min(20, font_size // 8)  # フォントサイズに応じた変動
            y = y_base + random.randint(-y_variation, y_variation)
            
            # Y座標を安全範囲内に制限
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            # バウンディングボックスを計算
            bbox = [x, y, x + width, y + height]
            
            # 最終チェック
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_size[0], bbox[2])
            bbox[3] = min(image_size[1], bbox[3])
            
            bboxes.append(bbox)
    
    elif layout_type == 'slight_vertical':
        # 軽い縦方向の変化を加えた配置
        x_positions = generate_horizontal_positions_dynamic(char_sizes, available_width, margin, min_spacing, font_size)
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            y_variation = min(25, font_size // 6)
            y_offset = int(y_variation * random.uniform(-1, 1))
            y = margin + available_height // 2 + y_offset
            
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            bbox = [x, y, x + width, y + height]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_size[0], bbox[2])
            bbox[3] = min(image_size[1], bbox[3])
            
            bboxes.append(bbox)
    
    elif layout_type == 'wave':
        # 波形パターンで配置
        x_positions = generate_horizontal_positions_dynamic(char_sizes, available_width, margin, min_spacing, font_size)
        
        for i, (char, (width, height)) in enumerate(zip(formula, char_sizes)):
            x = x_positions[i]
            wave_amplitude = min(30, font_size // 5, available_height // 4)
            y_offset = int(wave_amplitude * random.uniform(-1, 1))
            y = margin + available_height // 2 + y_offset
            
            y = max(y_min, min(y, y_max))
            
            positions.append((x, y, char))
            
            bbox = [x, y, x + width, y + height]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(image_size[0], bbox[2])
            bbox[3] = min(image_size[1], bbox[3])
            
            bboxes.append(bbox)
    
    return positions, bboxes

def generate_horizontal_positions_dynamic(char_sizes, available_width, margin, min_spacing, font_size):
    """フォントサイズに応じて間隔を調整する水平配置"""
    total_char_width = sum([size[0] for size in char_sizes])
    
    # フォントサイズに応じて最小間隔を調整
    adjusted_spacing = max(min_spacing, font_size // 20)
    min_total_spacing = adjusted_spacing * (len(char_sizes) - 1)
    
    if total_char_width + min_total_spacing > available_width:
        # 文字が多すぎる場合は最小間隔で配置
        spacing = max(2, (available_width - total_char_width) // max(1, len(char_sizes) - 1))
        x_positions = []
        x = margin
        for width, _ in char_sizes:
            x_positions.append(x)
            x += width + spacing
    else:
        # ランダムな間隔で配置
        extra_space = available_width - total_char_width - min_total_spacing
        spacings = []
        
        for i in range(len(char_sizes) - 1):
            max_extra = extra_space // max(1, len(char_sizes) - 1 - i)
            random_extra = random.randint(0, max_extra)
            spacings.append(adjusted_spacing + random_extra)
            extra_space -= random_extra
        
        x_positions = []
        x = margin
        for i, (width, _) in enumerate(char_sizes):
            x_positions.append(x)
            x += width
            if i < len(spacings):
                x += spacings[i]
    
    return x_positions

# データセット一括生成（ランダムレイアウト対応版）
def create_voc_dataset(
    output_dir="dataset",
    num_samples=100,
    image_size=(800, 200),
    font_size=128,
    formula_list=None,
    font_paths=None,
    font_counts=None,
    random_font_size=False,
    font_size_range=(80, 150),
    random_layout=False
):
    # ディレクトリ準備
    img_dir = os.path.join(output_dir, "JPEGImages")
    ann_dir = os.path.join(output_dir, "Annotations")
    sets_dir = os.path.join(output_dir, "ImageSets", "Main")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(sets_dir, exist_ok=True)

    # フォントの準備
    if font_paths is None:
        font_paths = [
            "Arial.ttf"
        ]
    
    base_fonts = load_fonts(font_paths, font_size)
    print(f"📝 使用可能なフォント数: {len(base_fonts)}")

    image_ids = []

    if formula_list is not None:
        formulas = formula_list
        actual_samples = len(formulas)
    else:
        formulas = [generate_random_formula() for _ in range(num_samples)]
        actual_samples = num_samples

    font_assignment = create_font_assignment(
        num_fonts=len(base_fonts),
        total_samples=actual_samples,
        font_counts=font_counts
    )
    
    font_usage_count = {i: 0 for i in range(len(base_fonts))}

    for i in range(actual_samples):
        formula = formulas[i]
        image_id = f"image_{i:03}"
        image_ids.append(image_id)

        font_index = font_assignment[i]
        base_font_path = font_paths[font_index] if font_index < len(font_paths) else "default"
        font_usage_count[font_index] += 1

        # フォントサイズを決定（制限を緩和）
        if random_font_size:
            # より柔軟なサイズ制限に変更
            max_safe_size = min(font_size_range[1], int(image_size[1] * 0.8))  # 画像高さの80%まで
            min_safe_size = max(font_size_range[0], 30)
            current_font_size = random.randint(min_safe_size, max_safe_size)
        else:
            current_font_size = min(font_size, int(image_size[1] * 0.6))  # 制限を緩和
        
        # フォントを作成
        try:
            if base_font_path != "default":
                font = ImageFont.truetype(base_font_path, current_font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # 画像作成
        img = Image.new("RGB", image_size, "white")
        draw = ImageDraw.Draw(img)

        if random_layout:
            # ランダムレイアウトで配置（マージンを動的に調整）
            positions, bboxes = generate_random_layout_dynamic(formula, font, image_size, current_font_size)
            labels = []
            
            for pos, bbox in zip(positions, bboxes):
                x, y, char = pos
                draw.text((x, y), char, font=font, fill="black")
                
                if char.isdigit() or char in "+-*/=()":
                    labels.append(char)
        else:
            # 従来の水平レイアウト（改善版）
            x_offset = 20  # マージンを小さく
            # Y座標を安全に計算
            font_height = font.getmetrics()[0] + font.getmetrics()[1]  # ascent + descent
            y_offset = max(10, (image_size[1] - font_height) // 2)
            
            bboxes = []
            labels = []

            for char in formula:
                char_width, char_height = calculate_text_dimensions(char, font)
                
                xmin = int(x_offset)
                ymin = int(y_offset)
                xmax = int(x_offset + char_width)
                ymax = int(y_offset + char_height)
                
                # 境界チェック（より柔軟に）
                if xmax > image_size[0] - 10:
                    break  # 右端に達したら終了
                if ymax > image_size[1] - 10:
                    ymin = image_size[1] - char_height - 10
                    ymax = image_size[1] - 10

                draw.text((xmin, ymin), char, font=font, fill="black")

                if char.isdigit() or char in "+-*/=()":
                    labels.append(char)
                    bboxes.append([xmin, ymin, xmax, ymax])

                x_offset += char_width + 15  # 間隔を少し縮める

        # 保存
        img.save(os.path.join(img_dir, f"{image_id}.jpg"))
        save_voc_annotation(image_id, image_size, bboxes, labels, ann_dir)

    # ImageSets/Main/train.txt を保存
    with open(os.path.join(sets_dir, "train.txt"), "w") as f:
        for image_id in image_ids:
            f.write(image_id + "\n")

    # 統計表示
    print("\n📊 フォント使用統計:")
    for i, count in font_usage_count.items():
        font_name = font_paths[i] if i < len(font_paths) else "デフォルト"
        percentage = (count / actual_samples) * 100
        print(f"  フォント{i+1} ({font_name}): {count}回使用 ({percentage:.1f}%)")

    if formula_list is not None:
        print(f"✅ 指定された{actual_samples}個の数式でデータ生成が完了しました。保存先: {output_dir}")
    else:
        print(f"✅ ランダム生成で{actual_samples}枚のデータ生成が完了しました。保存先: {output_dir}")
    print(f"🎨 使用したフォント数: {len(base_fonts)}")
    
    if random_font_size:
        print(f"📏 フォントサイズ範囲: {min_safe_size}-{max_safe_size}")
    if random_layout:
        print(f"🎲 ランダムレイアウト: 有効")
    
    print(f"🛡️ 枠はみ出し防止機能: 有効")

# 変数名説明:
# output_dir: データセットの出力ディレクトリ
# num_samples: 生成するサンプル数
# image_size: 画像サイズ (幅, 高さ)
# font_size: 基本フォントサイズ
# formula_list: 数式リスト（Noneの場合はランダム生成）例: ["12+34=", "7-2="]
# font_paths: 使用するフォントファイルのパスリスト
# font_counts: フォントごとの使用枚数リスト（Noneの場合は均等分割）
# random_font_size: フォントサイズをランダムにするか
# font_size_range: ランダムフォントサイズの範囲 (最小, 最大)
# random_layout: ランダムレイアウトを使用するか
# output_dirとnum_samplesは必須引数

# 実行

create_voc_dataset(
    output_dir="dataset",
    num_samples=3900,
)