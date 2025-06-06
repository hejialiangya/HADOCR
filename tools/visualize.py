import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def parse_system_results(file_path):
    """
    解析 system_results.txt 文件，将其转换为一个字典。

    返回值:
        ocr_results: 字典，键为图片名称，值为 OCR 标注列表
    """
    ocr_results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                image_name, ocr_json = line.split('\t', 1)
                annotations = json.loads(ocr_json)
                ocr_results[image_name] = annotations
            except ValueError:
                print(f"第 {line_number} 行格式错误，无法分割为图片名和 OCR 数据。跳过该行。")
                continue
            except json.JSONDecodeError:
                print(f"第 {line_number} 行的 OCR 数据不是有效的 JSON 格式。跳过该行。")
                continue
    return ocr_results

def annotate_images(input_folder, output_folder, ocr_results, font_path):
    """
    为每张图片绘制 OCR 标注并保存到输出文件夹。

    参数:
        input_folder: 原始图片所在文件夹路径
        output_folder: 保存标注后图片的文件夹路径
        ocr_results: OCR 结果字典
        font_path: 支持中文的 TrueType 字体文件路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, size=20)
    except IOError:
        print(f"无法加载字体文件：{font_path}")
        return

    total_images = len(ocr_results)
    print(f"总共有 {total_images} 张图片需要处理。")

    for idx, (image_name, annotations) in enumerate(ocr_results.items(), start=1):
        image_path = os.path.join(input_folder, image_name)

        # 检查图片是否存在
        if not os.path.isfile(image_path):
            print(f"[{idx}/{total_images}] 图片 {image_name} 不存在于 {input_folder} 中。跳过。")
            continue

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{idx}/{total_images}] 无法读取图片 {image_name}。跳过。")
            continue

        # 绘制多边形（边界框）使用 OpenCV
        for anno in annotations:
            transcription = anno.get("transcription", "")
            points = anno.get("points", [])
            score = anno.get("score", 0.0)

            if not points or len(points) < 4:
                print(f"图片 {image_name} 中的一个标注点数据不足。跳过该标注。")
                continue

            # 将点转换为整数元组
            try:
                pts = [(int(x), int(y)) for x, y in points]
            except ValueError:
                print(f"图片 {image_name} 中的标注点包含非整数值。跳过该标注。")
                continue

            # 绘制多边形（边界框）使用 OpenCV
            cv2.polylines(image, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # 转换颜色空间从 BGR 到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 绘制文本
        for anno in annotations:
            transcription = anno.get("transcription", "")
            points = anno.get("points", [])
            score = anno.get("score", 0.0)

            if not points or len(points) < 4:
                continue

            # 将点转换为整数元组
            try:
                pts = [(int(x), int(y)) for x, y in points]
            except ValueError:
                continue

            # 计算文本的起始位置（根据左上角坐标稍作调整）
            text_x, text_y = pts[0]
            # 避免文本绘制到图片外
            if text_y - 10 < 0:
                text_y += 20
            else:
                text_y -= 10

            # 使用 Pillow 绘制中文文本
            draw.text((text_x, text_y), transcription, font=font, fill=(255, 0, 0))

            # 可选：在边界框旁显示置信度分数
            score_text = f"{score:.2f}"
            score_x, score_y = pts[0]
            if score_y - 30 < 0:
                score_y += 40
            else:
                score_y -= 30
            draw.text((score_x, score_y), score_text, font=font, fill=(0, 0, 255))

        # 将 PIL 图像转换回 OpenCV 图像
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 保存标注后的图片到输出文件夹
        output_path = os.path.join(output_folder, image_name)
        success = cv2.imwrite(output_path, annotated_image)
        if success:
            print(f"[{idx}/{total_images}] 已保存标注后的图片到 {output_path}")
        else:
            print(f"[{idx}/{total_images}] 保存图片 {output_path} 失败。")

    print("所有图片处理完毕。")

def main():
    # 文件路径配置
    system_results_file = "system_results.txt"  # system_results.txt 在当前文件夹
    input_folder = "./"  # 当前文件夹
    output_folder = "./output"  # 输出文件夹
    font_path = r"C:\Windows\Fonts\simhei.ttf"  # 字体文件路径

    # 解析 system_results.txt
    print("正在解析 system_results.txt 文件...")
    ocr_results = parse_system_results(system_results_file)
    print(f"已解析 {len(ocr_results)} 张图片的 OCR 结果。")

    # 注释图片
    print("正在为图片添加标注...")
    annotate_images(input_folder, output_folder, ocr_results, font_path)


if __name__ == "__main__":
    main()
