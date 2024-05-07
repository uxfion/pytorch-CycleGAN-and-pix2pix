# 移动超声图像的可控超分辨率重建（掌超超分辨率）

## 代码

https://github.com/Gemlab-Super-Resolution/chenhaoming

## 数据

位置：数据服务器`/mnt/gemlab_data/Medical_image_database/掌超超分辨数据集`

## Conda环境

```bash
conda activate lecter
```

## 训练

```bash
python train.py --dataroot ./datasets/nature_ultrasound --name 7.changsha --model my_cycle_gan --batch_size 5 --gpu_ids 1 --preprocess scale_width_and_crop --load_size 286 --crop_size 256 --dataset_mode my_aligned
```

## 推理

```bash
python cyclegan_infer.py
```

## Demo网页

```bash

python cyclegan_infer_api.py
streamlit run web_api.py
```

然后访问`8501`端口