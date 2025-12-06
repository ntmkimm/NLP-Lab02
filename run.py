
# csv = ID, gt, Han Char, Image Box  | VÍ DỤ: LSE_001.004.001.01, ... cái này là từng patch nè (mã sách, hồi, trang, cột)
# json | VÍ DỤ: LSE_01_04 (mã sách, hồi trang) [lưu theo trang]
# file_id, meta, sect, page, stc
# file_id: LSE_001, metadata: {}, sect_id: tên hồi, page_id: 001, stc: { LSE_001.004.001.01: text }

from paddleocr import PaddleOCR
import cv2  
import numpy as np
from pathlib import Path
import pandas as pd
import json
from jiwer import cer

gt_file = Path("./data/gt.json")
patches = Path("./data/Patches")
scale_factor = 1.1

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det", 
    text_recognition_model_name="PP-OCRv5_server_rec",
    lang="ch",
    use_textline_orientation=True,
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    text_rec_score_thresh=0.01,
    text_det_box_thresh=0.01,
    text_recognition_batch_size=16,
    # text_det_unclip_ratio=2.0,
    # det_limit_side_len=15000, 
    # det_limit_type='max',     
)

def process_image_resolution(image_path, scale_factor=2.0):
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img_upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img_upscaled

def process_polygon(polys, scale_factor, gt_poly):
    '''
    Logic này cần được update thêm đề mapping về lại Pages trên ảnh gốc (file gt.json) - do có augment bằng cách rotate ảnh nữa
    '''
    # polys: list of polygons (each polygon = list of [x, y])
    # step 1 - merge_polygons logic
    if not polys:
        return []

    all_points = [pt for poly in polys for pt in poly]

    top_left = min(all_points, key=lambda p: (p[1], p[0]))
    top_right = min(all_points, key=lambda p: (p[1], -p[0]))
    bottom_left = max(all_points, key=lambda p: (p[1], p[0]))
    bottom_right = max(all_points, key=lambda p: (p[1], -p[0]))

    merged = [top_left, top_right, bottom_right, bottom_left]

    # step 2 - scale_polygon logic
    scaled = [[p[0] / scale_factor, p[1] / scale_factor] for p in merged]

    # step 3 - add_polygons logic
    if len(scaled) != len(gt_poly):
        raise ValueError("Polygons must have the same number of points.")

    final_poly = [
        [p1[0] + p2[0], p1[1] + p2[1]]
        for p1, p2 in zip(scaled, gt_poly)
    ]

    return final_poly

result = []
with open(gt_file, "r") as fi:
    gt_data = json.load(fi)

# input_dir = patches / "DVSKTT-1 Quyen thu"
img_paths = []
for input_dir in sorted(patches.iterdir()):
    if not input_dir.is_dir() or input_dir.name == "Transcriptions":
            continue
    for img_path in sorted(input_dir.glob("*.jpg")):
        splits = img_path.stem.split("_")
        img_paths.append(img_path)

for img_path in img_paths:
    try:
        file_id, sect, page_id, col = "_".join(splits[:-3]), splits[-3], splits[-2], splits[-1]
        _id = f"{file_id}.{sect}.{page_id}.{col}"
    except:
        continue
    
    # augment rotate
    high_res_img = process_image_resolution(img_path, scale_factor=scale_factor)
    high_res_img_rotated = cv2.rotate(high_res_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if high_res_img is not None:
        page_res = ocr.predict(high_res_img) 
        page_res_rotated = ocr.predict(high_res_img_rotated)
        gt = gt_data[_id]
        # print(f"{len(page_res_rotated)} > {len(page_res)}")
        results = page_res_rotated if len(" ".join(page_res_rotated[0]["rec_texts"])) > len(" ".join(page_res[0]["rec_texts"])) else page_res
        
        for res in results:
            res.save_to_img(f"output/{_id}.jpg")
            # res.save_to_json(f"output/{img_path.stem}.json")
            res = res._save_funcs[0].__self__
            polys = [p.tolist() for p in res["rec_polys"]]
            dic = {
                "ID": _id,
                "HanChar": " ".join(res["rec_texts"]), # merge các kết quả lại với nhau
                "GroundTruth": gt["label"],
                "ImageBox": process_polygon(polys=polys, scale_factor=scale_factor, gt_poly=gt["poly"]),
            }
            result.append(dic)

df = pd.DataFrame(result)
df["GroundTruth"] = df["GroundTruth"].fillna("").astype(str)
df["HanChar"] = df["HanChar"].fillna("").astype(str)

df.to_csv("output.csv", index=False, encoding="utf-8-sig")

df["CER"] = df.apply(lambda row: cer(row["GroundTruth"], row["HanChar"]), axis=1)
mean_cer = df["CER"].mean()
print("Mean CER:", mean_cer)