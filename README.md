# 移动超声图像的可控超分辨率重建（掌超超分辨率）

## 论文

[链接](https://scholar.google.com/)

## 代码

<https://github.com/Gemlab-Super-Resolution/chenhaoming>

本地运行：

```bash
git clone https://github.com/Gemlab-Super-Resolution/chenhaoming.git pytorch-CycleGAN-and-pix2pix
```

## 数据

数据服务器 `/mnt/gemlab_data/Medical_image_database/US/掌超超分辨数据集`

解压后将需要的数据（纯图片）放到`./pytorch-CycleGAN-and-pix2pix_latest/datasets/nature_ultrasound/train`下。


## Conda环境

conda打包文件：`/mnt/gemlab_data/User_database/chenhaoming/env/env-lecter-2023-12-05.tar.gz`

```bash
tar -xzf ./env-lecter-2023-12-05.tar.gz -C ~/anaconda3/envs/lecter
conda activate lecter
conda install certifi  # cert可能会报错
```

## 训练

```bash
cd ./pytorch-CycleGAN-and-pix2pix_latest
python train.py --dataroot ./datasets/nature_ultrasound --name 7.changsha --model my_cycle_gan --batch_size 5 --gpu_ids 1 --preprocess scale_width_and_crop --load_size 286 --crop_size 256 --dataset_mode my_aligned
```

## 推理

```bash
python cyclegan_infer.py
```

如果需要参数请详见代码。

## Demo网页

```bash
python cyclegan_infer_api.py
# 新开一个终端再运行
streamlit run web_api.py
```

然后访问本机`8501`端口

Demo后台运行在`z790`服务器上，内网地址：<http://100.100.100.4:8501>，外网地址：<http://gemlab.site:58501>