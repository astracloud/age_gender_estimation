# Age Gender 分類器

## 訓練
```
python train_tf2.py
```
訓練時，會在checkpoints 產生多個h5權重檔

## 輸出模型
```
python h5tomodel.py
```
修改H5_PATH的指定h5檔，執行後會在saved_model產生模型，再依照需求包成zip檔並丟到S3雲端

## Docker Image
如果在其他地方clone下來沒有模型檔，想從雲端下載，則請先執行
```
python download_script.py
```
此時在saved_model裡就會有該模型，再執行
```
docker-compose build
```
即可建立image

若要執行印象檔
```
docker-compose up
```
則會產生 age_gender_estimation_web_1 和 age_gender_estimation_serving_1，兩個container，即可對 age_gender_estimation_web_1 發API來使用。

測試container是否正常執行可使用
```
python web_client.py
```
