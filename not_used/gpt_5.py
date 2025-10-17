import os
from openai import OpenAI
import base64
import cv2
import re
import argparse
import pickle
from utils import evaluate, get_labelme_gt


def parse_args():
    parser = argparse.ArgumentParser(description="测试gpt5")
    parser.add_argument("--input-dir", type=str, default='data/00-test', help="输入目录")
    parser.add_argument("--output-dir", type=str, default='data/00-result-gpt', help="输出目录")
    parser.add_argument("--key", type=str, default='', help="key")
    args = parser.parse_args()
    return args


def predict(client, prompt, image_path, output_dir):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    bbox_content = response.output_text
    print('message.content=', bbox_content)

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    pred_bboxes = []

    # 检查结果格式是否正确
    for m in re.findall('<bbox>.*</bbox>', bbox_content):
        coords_str = m[len('<bbox>'):-len('</bbox>')]
        coords = list(map(int, coords_str.split()))
        if len(coords) != 4:  # 验证坐标数量(xmin, ymin, xmax, ymax)
            raise ValueError("we need 4 numbers!")
        x_min, y_min, x_max, y_max = coords

        # 获取图像尺寸并缩放坐标(模型输出范围为0-1000)
        x_min_real = int(x_min * w / 1000)
        y_min_real = int(y_min * h / 1000)
        x_max_real = int(x_max * w / 1000)
        y_max_real = int(y_max * h / 1000)

        pred_bboxes.append([x_min_real, y_min_real, x_max_real, y_max_real])
        cv2.rectangle(image, (x_min_real, y_min_real), (x_max_real, y_max_real), (0, 0, 255), 3)

    output_path = os.path.join(output_dir, os.path.split(image_path)[1])
    cv2.imwrite(output_path, image)
    print(f"save result image to: {output_path}")
    return pred_bboxes


def main(args):
    prompt = '框出图中有人乞讨的位置，输出 bounding box 的坐标, 若无人乞讨则不要输出bounding box'
    client = OpenAI(api_key=args.key)

    result = []

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if not file.endswith('.jpg'):
                continue
            
            image_path = os.path.join(root, file)
            json_path = os.path.splitext(image_path)[0] + '.json'

            gt_bboxes = []
            if os.path.exists(json_path):
                gt_bboxes = get_labelme_gt(json_path)

            pred_bboxes = predict(client, prompt, image_path, args.output_dir)
            result.append((image_path, gt_bboxes, pred_bboxes))
            break
        break

    # with open('cache.pkl', 'wb') as f:
    #     pickle.dump(result, f)
    evaluate(result)


def tmp(args):
    client = OpenAI(api_key=args.key)
    print(client)
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input="what is today?",
        max_tokens=1000
    )

    print(response.output_text)


if __name__ == '__main__':
    args = parse_args()
    tmp(args)