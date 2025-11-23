# utils/processing.py
import cv2
from pathlib import Path
import config as C

# TODO: Task 1: Preprocessing (10%)
'''
Apply basic preprocessing operations such as resizing, denoising, and contrast enhancement to improve the visual quality of the frames; 
you may also optionally perform video stabilization if the footage is shaky. In your report, clearly describe and justify each preprocessing
step, explaining why it is appropriate for low-quality people-detection and counting.
'''
def preprocess_frame(name, img):

    # Sharpening
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(img, 2.0, blurred, -1.0, 0)

    # Contrast Enhancement
    contrast_enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=-25)

    processed_filename = f"processed_{name}"  # e.g., processed_seq_000123.jpg
    processed_path = C.PROCESSED_FRAMES_DIR / processed_filename

    cv2.imwrite(str(processed_path), contrast_enhanced)
    print(f"[o] Processed frame saved to: {processed_path}")

    # Showing original VS Result
    # cv2.imshow(f"Original: {name}", img)
    # cv2.imshow(f"Pre-Processed: {processed_filename}", sharpened)

    #key = cv2.waitKey(0)
    #if key == ord('q'):
    #    cv2.destroyAllWindows()

    return([name, contrast_enhanced])






'''
    # NONE OF THIS HELPED WITH THE IMAGES
    #**********************************************
    #resizing
    
    #=============
    # ----- 1. Contrast Enhancement with CLAHE -----
    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)

    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #cl = clahe.apply(l)

    #enhanced_lab = cv2.merge((cl, a, b))
    #enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # ----- 2. Gentle Detail Enhancement -----
    #detail = cv2.detailEnhance(img, sigma_s=20, sigma_r=0.10)
'''