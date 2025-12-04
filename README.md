Xử lý data, tạo thư mục data và tải data trên kaggle về
```
mkdir data
cd data
curl -L -o nomnaocr.zip  https://www.kaggle.com/api/v1/datasets/download/quandang/nomnaocr
unzip nomnaocr.zip
cd ..
```
Setup môi trường
```
conda create -n paddle python==3.10 -y
conda activate paddle
pip install paddlepaddle-gpu
pip install paddleocr[all]
pip install jiwer
```
QUAN TRỌNG: Ta cần tạo file chứa groundtruth được xử lí để dễ truy vấn: `./data/gt.json` từ dataset để đánh giá CER
```
python get_gt.py
```
Sau khi setup data và file groundtruth, ta chạy paddleocr trên `./data/Patches`:
```
python run.py
```
Sau khi chạy xong sẽ in ra CER của results và groundtruth, tùy chỉnh thư mục chạy trong code