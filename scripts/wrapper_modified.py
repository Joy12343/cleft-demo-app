#!/usr/bin/env python3
"""
Modified wrapper functions to work with Flask backend.
Extracted from the original wrapper.py for better modularity.
"""

import os
import shutil
import tempfile
import subprocess
import yaml
from PIL import Image
import numpy as np
import cv2
import face_alignment
from skimage import io

def convert_pngs_to_jpgs(folder):
    """Convert any .png in folder to .jpg (delete .png afterward)."""
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue

        png_path = os.path.join(folder, fname)
        jpg_name = os.path.splitext(fname)[0] + ".jpg"
        jpg_path = os.path.join(folder, jpg_name)

        with Image.open(png_path) as im:
            rgb = im.convert("RGB")
            rgb.save(jpg_path, quality=95)

        os.remove(png_path)

def generate_landmarks(folder, photo):
    """Run face_alignment, draw landmarks and save .txt coords."""
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu'
    )
    img_path = os.path.join(folder, photo)
    img = io.imread(img_path)
    preds = fa.get_landmarks(img)
    if not preds:
        raise RuntimeError(f"No faces detected in {photo}")

    lm = preds[0]  # (68,2)
    # draw and save image with landmarks
    img_lm = img.copy()
    for x, y in lm:
        cv2.circle(img_lm, (int(x), int(y)), 2, (0,255,0), -1)
    lm_jpg = photo.replace('.jpg','_landmark.jpg')
    cv2.imwrite(os.path.join(folder, lm_jpg), cv2.cvtColor(img_lm, cv2.COLOR_RGB2BGR))

    # save flattened coords to txt
    flat = lm.flatten()
    lm_txt = photo.replace('.jpg','_landmark.txt')
    with open(os.path.join(folder, lm_txt), 'w') as f:
        f.write(" ".join(map(str,flat)) + "\n")

    return lm_jpg, lm_txt

def generate_binary_mask(folder, mask):
    """Threshold the user-painted mask into a strict 0/255 binary image."""
    mask_path = os.path.join(folder, mask)
    img = cv2.imread(mask_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
    outname = 'binary_mask.jpg'
    cv2.imwrite(os.path.join(folder, outname), bin_mask)
    return outname

def run_inpainting_process(folder, photo, binary_mask, landmark_txt):
    """Run the inpainting process and return the result path."""
    
    # Path to the config file (adjust this path as needed)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'config.yml')
    
    # Check if config exists
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    
    with tempfile.TemporaryDirectory() as tmp:
        # temp subdirs
        photos_dir    = os.path.join(tmp,'photos');    os.makedirs(photos_dir)
        masks_dir     = os.path.join(tmp,'masks');     os.makedirs(masks_dir)
        landmarks_dir = os.path.join(tmp,'landmarks'); os.makedirs(landmarks_dir)

        # copy the three files into temp
        shutil.copy(os.path.join(folder, photo), photos_dir)
        shutil.copy(os.path.join(folder, binary_mask), masks_dir)
        shutil.copy(os.path.join(folder, landmark_txt), landmarks_dir)

        # load + patch config
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        cfg['TEST_INPAINT_IMAGE_FLIST']    = [os.path.join(photos_dir, photo)]
        cfg['TEST_MASK_FLIST']             = [os.path.join(masks_dir, binary_mask)]
        cfg['TEST_INPAINT_LANDMARK_FLIST'] = [os.path.join(landmarks_dir, landmark_txt)]

        # write temp config
        tmp_cfg = os.path.join(tmp,'config.yml')
        with open(tmp_cfg,'w') as f:
            yaml.safe_dump(cfg, f)

        # backup & overwrite
        backup = CONFIG_PATH + '.bak'
        shutil.copy(CONFIG_PATH, backup)
        
        try:
            shutil.copy(tmp_cfg, CONFIG_PATH)
            
            # Change to the directory containing test.py before running
            original_cwd = os.getcwd()
            test_py_dir = os.path.dirname(CONFIG_PATH)
            os.chdir(test_py_dir)
            
            result = subprocess.run(['python3', 'test.py'], 
                                  capture_output=True, text=True, check=True)
            
            os.chdir(original_cwd)
            
            # Find the generated result file
            # This depends on how your test.py saves results
            # Adjust the path pattern based on your actual output structure
            results_pattern = os.path.join(test_py_dir, 'results', '**', '*.jpg')
            import glob
            result_files = glob.glob(results_pattern, recursive=True)
            
            if not result_files:
                raise RuntimeError("No result files generated")
            
            # Return the most recent result file
            latest_result = max(result_files, key=os.path.getctime)
            
            # Copy result to our folder
            result_filename = f"inpainted_{photo}"
            result_path = os.path.join(folder, result_filename)
            shutil.copy(latest_result, result_path)
            
            return result_path
            
        finally:
            shutil.copy(backup, CONFIG_PATH)
            os.remove(backup)
