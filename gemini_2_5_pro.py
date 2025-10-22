import os
from google import genai
import base64
import cv2
import re
import argparse
import pickle
import json
import time
from utils import evaluate, get_labelme_gt_txt


def parse_args():
    parser = argparse.ArgumentParser(description="测试gemini 2.5 pro")
    parser.add_argument("--input-dir", type=str, default='data/01-tongfengguan', help="输入目录")
    parser.add_argument("--output-dir", type=str, default='data/01-result-gemini', help="输出目录")
    parser.add_argument("--keys", type=str, nargs='+', default=[], help="keys")
    parser.add_argument("--cache", type=str, default='cache.pkl', help="keys")
    args = parser.parse_args()
    return args


def predict(client, prompt, image_path, output_dir):
    my_file = client.files.upload(file=image_path)

    try:
        bbox_content = ''
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[my_file, prompt],
        )
        
        bbox_content = response.text
        print('message.content=', bbox_content)
    
    except Exception as e:
        print(f'get error {e}')
        return bbox_content + f'get error {e}'



    if '```json' in bbox_content:
        bbox_content = bbox_content.replace('```json', '').replace('```', '')
        bbox_content = json.loads(bbox_content)
    else:
        bbox_content = []
    print(bbox_content)

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    pred_bboxes = []

    # 检查结果格式是否正确
    for m in bbox_content:
        coords = m['box_2d']
        if len(coords) != 4:  # 验证坐标数量(xmin, ymin, xmax, ymax)
            raise ValueError("we need 4 numbers!")
        y_min, x_min, y_max, x_max = coords

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
    prompt = '框出图中通风管结霜的位置，输出 bounding box 的坐标, 若无通风管结霜则不要输出bounding box'
    ikey = 0
    client = genai.Client(api_key=args.keys[ikey])
    
    result = []
    finished = []
    if os.path.exists(args.cache):
        with open(args.cache, 'rb') as f:
            result = pickle.load(f)
            finished = [x[0] for x in result]

    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if not file.endswith('.jpg'):
                continue
            
            image_path = os.path.join(root, file)
            if image_path in finished:
                print('finished: ', image_path)
                continue

            json_path = os.path.splitext(image_path)[0] + '.txt'

            gt_bboxes = []
            if os.path.exists(json_path):
                gt_bboxes = get_labelme_gt_txt(json_path)

            pred_bboxes = predict(client, prompt, image_path, args.output_dir)

            while isinstance(pred_bboxes, str) and 'quato' in pred_bboxes and ikey < len(args.keys):
                ikey += 1
                client = genai.Client(api_key=args.keys[ikey])
                pred_bboxes = predict(client, prompt, image_path, args.output_dir)

            if isinstance(pred_bboxes, list):
                result.append((image_path, gt_bboxes, pred_bboxes))
                time.sleep(30)  # 免费用户每分钟限制2个请求，每天限制50个请求
            elif pred_bboxes is None:
                break

    with open(args.cache, 'wb') as f:
        pickle.dump(result, f)
    if result:
        evaluate(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)