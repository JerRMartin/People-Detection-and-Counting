# utils/processing.py
import cv2
import config as C
import numpy as np

def preprocess_frame(name, img, *args):
    processed_img = img.copy()
    processed_filename = f"processed_{name}"  # e.g., processed_seq_000123.jpg
    print(f"[o] ==== Processing {name} ====")

    for adjustment in args:
        match adjustment:
            case C.IMAGE_ADJUSTMENT.SHARPEN:
                print(f"Sharpening {name}...")
                processed_img = sharpen_image(processed_img)
            case C.IMAGE_ADJUSTMENT.CONTRAST_ENHANCEMENT:
                print(f"Enhancing Contrast of {name}...")
                processed_img = contrast_enhance(processed_img)
            case C.IMAGE_ADJUSTMENT.GAUSSIAN_BLUR:
                print(f"Blurring {name}...")
                processed_img = blur_image(processed_img)
            case C.IMAGE_ADJUSTMENT.RANDOM_NOISE:
                print(f"Adding Gaussian Noise to {name}...")
                processed_img = add_random_noise(processed_img)

    processed_path = C.FRAME_DIRECTORY.PROCESSED / processed_filename

    cv2.imwrite(str(processed_path), processed_img)
    print(f"[o] Processed frame saved to: {processed_path}")

    return([name, processed_img])


def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)
    return sharpened

def contrast_enhance(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=-25)

def blur_image(img): 
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    return blurred

def add_random_noise(image, intensity=25):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image