from pathlib import Path
import json 

ROOTDIR = Path("/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/US/NLP/semi/data/Pages")

dic = {}

for input_folder in sorted(ROOTDIR.iterdir()):
    if not input_folder.is_dir() or input_folder.name == "Transcriptions":
        continue

    gt_folder = input_folder / "gts"

    for txt_file in sorted(gt_folder.glob("*.txt")):
        splits = txt_file.stem.split("_")
        try:
            file_id, sect, page_id = "_".join(splits[:-2]), splits[-2], splits[-1]
        except:
            print(txt_file)
            continue
        with open(txt_file, "r", encoding="utf-8") as fi:
            data = fi.readlines()

        for col, line in enumerate(data):
            line = line.strip()

            parts = line.split(",")
            
            # First 8 numbers -> polygon
            poly_bbox = list(map(float, parts[:8]))
            bbox = []
            poly_bbox = [poly_bbox[0:2], poly_bbox[2:4], poly_bbox[4:6], poly_bbox[6:8]]
                
            # Remaining text after the 8 numbers -> label
            label = "".join(parts[8])    # join back as string

            dic[f"{file_id}.{sect}.{page_id}.{col}"] = {
                "poly": poly_bbox,
                "label": label,
            }

with open("./data/gt.json", "w", encoding="utf-8") as f:
    json.dump(dic, f, ensure_ascii=False, indent=2)
