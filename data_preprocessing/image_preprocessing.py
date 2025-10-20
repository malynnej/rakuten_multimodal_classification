# General
import numpy as np
import pandas as pd

# Visualization:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# Image preprocessing
from matplotlib.image import imread
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import re
import cv2


class BackgroundDetector:
    """
    Check white background in images
    """
    
    def __init__(self, white_threshold: int = 240, border_ratio: float = 0.1, 
                 white_percentage_threshold: float = 0.9):
        """
        Args:
            white_threshold: Threshold for white pixels (0-255)
            border_ratio: Border ratio for analysis
            white_percentage_threshold: Percentage for white background classification
        """
        self.white_threshold = white_threshold
        self.border_ratio = border_ratio
        self.white_percentage_threshold = white_percentage_threshold
    
    def is_white_background(self, image: np.ndarray) -> bool:
        """
        Checks if image has white background
        
        Args:
            image: BGR image as numpy array

        Returns:
            True if white background is detected, else False
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        border_size = int(min(h, w) * self.border_ratio)

        # Extract border pixels
        borders = [
            gray[:border_size, :],
            gray[-border_size:, :],
            gray[:, :border_size],
            gray[:, -border_size:]
        ]
        
        border_pixels = np.concatenate([b.flatten() for b in borders])
        white_percentage = np.sum(border_pixels >= self.white_threshold) / len(border_pixels)
        
        return white_percentage > self.white_percentage_threshold
    
    def is_white_background_from_path(self, image_path: str) -> bool:
        """
        Checks if image has white background from file path
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.is_white_background(img)

class ImageCropper:
    """
    Class for adaptive image cropping operations
    """

    def __init__(self, padding: int = 10, min_crop_ratio: float = 0.7, min_aspect_ratio: float = 0.5):
        """
        Args:
            padding: Padding around detected object in pixels
            min_crop_ratio: Minimum ratio of cropped area to original area (0-1)
            min_aspect_ratio: Minimum allowed aspect ratio (width/height or height/width)
        """
        self.padding = padding
        self.min_crop_ratio = min_crop_ratio
        self.min_aspect_ratio = min_aspect_ratio

    def crop_white_background(self, image: np.ndarray, threshold: int = 240) -> np.ndarray:
        """
        Removes white background using contour detection, but keeps a minimum crop size and aspect ratio.

        Args:
            image: BGR image as numpy array
            threshold: Threshold for white detection
            min_crop_ratio: Minimum ratio of cropped area to original area (0-1)
            min_aspect_ratio: Minimum allowed aspect ratio (width/height or height/width)

        Returns:
            Cropped image
        """
        #grayscaling image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #inverse binary thresholding for contour detection: foreground objects become white (255) and background becomes black (0)
        _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        #elliptical morphological closing to close small holes in the foreground
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        #find contours and get bounding box of largest contour
        #the retr_external flag retrieves only the outermost contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image, gray

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # padding is added to prevent tight cropping that might cut off edges of the object 
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(image.shape[1] - x, w + 2 * self.padding)
        h = min(image.shape[0] - y, h + 2 * self.padding)

        # Minimum crop size and aspect ratio check before cropping
        orig_area = image.shape[0] * image.shape[1]
        crop_area = w * h
        crop_ratio = crop_area / orig_area
        aspect_ratio = w / h if w > h else h / w

        #crop_ratio check ensures we don't crop too much of the image
        #aspect_ratio check prevents extreme aspect ratios that could distort the image
        if crop_ratio < self.min_crop_ratio or aspect_ratio < self.min_aspect_ratio:
            # If crop is too small or aspect ratio is off, return original image
            return image, gray

        return image[y:y+h, x:x+w], gray
    
    def crop_colored_background(self, image: np.ndarray, target_ratio: float = 1.0) -> np.ndarray:
        """
        Center-Cropping for images with colored background
        
        Args:
            image: BGR image as numpy array
            target_ratio: Target aspect ratio (width/height)

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        
        if w > h:
            # Landscape format
            new_w = int(h * target_ratio)
            if new_w > w:
                return image
            start_x = (w - new_w) // 2
            return image[:, start_x:start_x+new_w]
        else:
            # Portrait format
            new_h = int(w / target_ratio)
            if new_h > h:
                return image
            start_y = (h - new_h) // 2
            return image[start_y:start_y+new_h, :]
        
class ImagePreprocessor:
    """
    Main class for complete image preprocessing
    Combines BackgroundDetector and ImageCropper
    """
    
    def __init__(self, background_detector: Optional[BackgroundDetector] = None,
                 image_cropper: Optional[ImageCropper] = None):
        """
        Args:
            background_detector: Instance of BackgroundDetector
            image_cropper: Instance of ImageCropper
        """
        self.detector = background_detector or BackgroundDetector()
        self.cropper = image_cropper or ImageCropper()
        
        # Statistics    
        self.stats = {
            'white_bg_count': 0,
            'colored_bg_count': 0,
            'total_processed': 0,
            'failed': 0
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Processes a single image adaptively based on background type
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Preprocessed image
        """
        if self.detector.is_white_background(image):
            self.stats['white_bg_count'] += 1
            cropped, _ = self.cropper.crop_white_background(image)
            return cropped
        else:
            self.stats['colored_bg_count'] += 1
            return self.cropper.crop_colored_background(image)
    
    def preprocess_image_from_path(self, input_path: str, output_path: str, gray_output_path: Optional[str]=None) -> bool:
        """
        Loads, processes and saves image
        
        Args:
            input_path: Path to input image
            output_path: Path to output image
            gray_output_path: Path to save greyscale version of the image (optional)
        
        Returns:
            True if successful, False if failed
        """
        try:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Image could not be loaded: {input_path}")
            
            # If white background, get both cropped and gray image
            if self.detector.is_white_background(image):
                cropped, gray = self.cropper.crop_white_background(image)
                if gray_output_path:
                    cv2.imwrite(gray_output_path, gray)
            else:
                cropped = self.cropper.crop_colored_background(image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if gray_output_path:
                    cv2.imwrite(gray_output_path, gray)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, cropped)
            self.stats['total_processed'] += 1
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            self.stats['failed'] += 1
            return False
    
    def preprocess_dataset(self, input_dir: str, output_dir: str, 
                          file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
        """
        (Deprecated) Processes entire dataset with category structure.
        Kept for backward compatibility. If your images are in a flat folder and
        categories are stored in a dataframe, use `preprocess_from_dataframe` instead.
        
        Args:
            input_dir: Directory with category folders
            output_dir: Target directory
            file_extensions: Allowed file extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}
        print(f"Starting preprocessing from {input_path} to {output_path}")
        # Iterate through categories
        for category_dir in input_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            output_category_dir = output_path / category_name

            # all images in the category
            image_files = [f for f in category_dir.iterdir()
                           if f.suffix.lower() in file_extensions]
            print(f"Found {len(image_files)} images in category '{category_name}'")
            # Process with progress bar
            for image_file in tqdm(image_files, desc=f'Category: {category_name}'):
                input_file = str(image_file)
                output_file = str(output_category_dir / image_file.name)
                self.preprocess_image_from_path(input_file, output_file)
        
        self._print_statistics()

    def preprocess_from_dataframe(self, df: pd.DataFrame, output_dir: str, output_dir_greyscale: Optional[str]=None,
                                  image_path_col: str = 'image_path',
                                  category_col: str = 'prdtypecode',
                                  file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
        """
        Processes images listed in a dataframe. Derives category from `category_col` (e.g. 'prdtypecode')
        and writes preprocessed images into `output_dir/<category>/` while keeping original filenames.

        Args:
            df: DataFrame containing at least the image path column and category column.
            output_dir: Root output directory where per-category subfolders will be created.
            image_path_col: Column name in df that points to the image file path.
            category_col: Column name in df that contains the category code to use as folder name.
            file_extensions: Allowed file extensions (used to filter rows if needed).
        """
        output_root = Path(output_dir)
        if output_dir_greyscale:
            output_root_greyscale = Path(output_dir_greyscale)
              
        # Reset statistics
        self.stats = {k: 0 for k in self.stats.keys()}

        # Iterate rows with progress bar
        rows = df.iterrows()
        total = len(df)
        print(f"Starting preprocessing of {total} images into {output_root}")
        for _, row in tqdm(rows, total=total, desc='Images'):
            img_path = row.get(image_path_col)
            if pd.isna(img_path):
                self.stats['failed'] += 1
                print(f"Missing image path for row: {row.name}")
                continue

            img_path = str(img_path)
            p = Path(img_path)
            if not p.exists():
                # try joining with a common input folder if paths are relative
                print(f"Image not found: {img_path} (skipping)")
                self.stats['failed'] += 1
                continue

            # derive category name and sanitize
            category_value = row.get(category_col, 'unknown')
            if pd.isna(category_value):
                category_name = 'unknown'
            else:
                category_name = str(category_value)
            # sanitize folder name
            category_name = re.sub(r"[^0-9A-Za-z_\-]", "_", category_name)

            output_category_dir = output_root / category_name
            output_category_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_category_dir / p.name)

            if output_dir_greyscale:
                output_category_dir_greyscale = output_root_greyscale / category_name
                output_category_dir_greyscale.mkdir(parents=True, exist_ok=True)
                gray_output_file = str(output_category_dir_greyscale / p.name)
                self.preprocess_image_from_path(str(p), output_file, gray_output_file)
            else:   
                self.preprocess_image_from_path(str(p), output_file)

        self._print_statistics()
    
    def _print_statistics(self):
        """Gives processing statistics"""
        print("\n" + "="*50)
        print("Preprocessing completed")
        print("="*50)
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"White background: {self.stats['white_bg_count']}")
        print(f"Colored background: {self.stats['colored_bg_count']}")
        print(f"Failed: {self.stats['failed']}")
        print("="*50)