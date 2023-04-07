import cv2
import os
import numpy as np
import tqdm
import random


class Blur:
    def __init__(self, input_path, output_path, type, times):
        self.input_path = input_path
        self.output_path = output_path
        self.type = type
        self.times = times

    def down_up(self, img_path):
        img_hr = cv2.imread(img_path)
        img = img_hr
        for i in range(self.times):
            img = cv2.pyrDown(img)
            img = cv2.pyrUp(img)
        img_lr = img
        return img_lr

    # gaussian noise
    def gaussian(self, img_path):
        mean = 0
        var = 0.002
        image = cv2.imread(img_path)
        image = np.array(image / 255, dtype=float)  # 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        noise = np.random.normal(
            mean, var**0.5, image.shape
        )  # 创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
        if out.min() < 0:
            low_clip = -1.0
        else:
            low_clip = 0.0
        out = np.clip(
            out, low_clip, 1.0
        )  # clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        out = np.uint8(out * 255)  # 解除归一化，乘以255将加噪后的图像的像素值恢复
        # cv.imshow("gasuss", out)
        noise = noise * 255
        return out

    # salt and pepper noise
    def salt_pepper(self, img_path):
        prob = 0.01
        image = cv2.imread(img_path)
        output = np.zeros(image.shape, np.uint8)
        noise_out = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()  # 随机生成0-1之间的数字
                if rdn < prob:  # 如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                    output[i][j] = 0
                    noise_out[i][j] = 0
                elif rdn > thres:  # 如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                    output[i][j] = 255
                    noise_out[i][j] = 255
                else:
                    output[i][j] = image[i][j]  # 其他情况像素点不变
                    noise_out[i][j] = 100
        # result = [noise_out,output]#返回椒盐噪声和加噪图像
        result = output  # 返回椒盐噪声和加噪图像
        return result

    def blur_all(self):
        for root, dirs, files in os.walk(input_path):
            if not os.path.exists(root.replace(self.input_path, self.output_path)):
                os.makedirs(root.replace(self.input_path, self.output_path))
            for file in tqdm.tqdm(files):
                img_path = os.path.join(root, file)
                if self.type == "down_up":
                    img = self.down_up(img_path)
                    file = file.split(".")[0] + "_down_up.jpg"
                elif self.type == "gaussian":
                    img = self.gaussian(img_path)
                    file = file.split(".")[0] + "_gaussian.jpg"
                elif self.type == "salt_pepper":
                    img = self.salt_pepper(img_path)
                    file = file.split(".")[0] + "_salt_pepper.jpg"
                output_img_path = os.path.join(
                    root.replace(self.input_path, self.output_path), file
                )
                cv2.imwrite(output_img_path, img)


if __name__ == "__main__":
    # input_path = "./lr2hr/testB"
    input_path = "./ultrasound_test"
    output_path = "./ultrasound_test_blur"

    blur = Blur(input_path, output_path, "down_up", 20)
    blur.blur_all()
