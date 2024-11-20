import os
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import glob

blur = 8
original_dir = './datasets/all/test'
results_base_dir = './results/xijing/vq'

# Find all test directories matching the pattern
test_dirs = glob.glob(f'{results_base_dir}/test_degradation{blur}_sr*')
test_dirs.sort()  # Sort to process in order

print(f"Found {len(test_dirs)} test directories")

for test_dir in test_dirs:
    # Extract sr value from directory name
    sr = test_dir.split('sr')[-1]
    print(f"\nProcessing degradation {blur}, sr {sr}")
    
    ssim_values = []
    psnr_values = []

    # Get original images
    original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f)) and f.endswith(".jpg")]
    original_files.sort()

    print(f"{len(original_files)} images found in {original_dir}")

    # Create output text file
    output_path = os.path.join(test_dir, f'metrics_deg{blur}_sr{sr}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Results for degradation {blur}, sr {sr}\n")
        f.write(f"Original directory: {original_dir}\n")
        f.write(f"Test directory: {test_dir}\n\n")

        # Process each image
        for file in original_files:
            original_path = os.path.join(original_dir, file)
            contrast_path = os.path.join(test_dir, file)
            
            if not os.path.exists(contrast_path):
                f.write(f"Contrast image not found for {file}\n")
                continue

            # Read images
            original = Image.open(original_path)
            contrast = Image.open(contrast_path)

            if original.size != contrast.size:
                contrast = contrast.resize(original.size)

            original = np.array(original)
            contrast = np.array(contrast)

            f.write(f"Processing: {file}\n")

            # Calculate SSIM
            ssim_value = compare_ssim(original, contrast, multichannel=True, win_size=3)
            ssim_values.append(ssim_value)
            f.write(f"SSIM: {ssim_value:.3f}\n")

            # Calculate PSNR
            psnr_value = compare_psnr(original, contrast)
            psnr_values.append(psnr_value)
            f.write(f"PSNR: {psnr_value:.3f} dB\n\n")

        # Calculate and write mean values
        mean_ssim = round(np.mean(ssim_values), 3)
        mean_psnr = round(np.mean(psnr_values), 3)

        f.write("=" * 50 + "\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Mean SSIM: {mean_ssim}\n")
        f.write(f"Mean PSNR: {mean_psnr} dB\n")

    print(f"Results saved to {output_path}")
    print(f"Mean SSIM: {mean_ssim}")
    print(f"Mean PSNR: {mean_psnr} dB")
