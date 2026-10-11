import os
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class MVTecDefectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        object_class,
        split="train",
        transform=None,
        defect_type=None,
        dilate_mask=False,
        mask_kernel_size=3,
        pairs=None,
        max_transform_retries=8,
    ):
        """
        Args:
            root_dir (str): Directory with MVTec AD dataset
            object_class (str): Object class (e.g., 'bottle', 'cable', etc.)
            split (str): 'train' or 'test'
            transform: Optional transform to be applied
            defect_type (str): Specific defect type to load (e.g., 'broken_large'). 
                              If None, loads all defect types.
        """
        # Ensure path uses correct operating system format
        self.root_dir = os.path.normpath(root_dir)
        self.object_class = object_class
        self.split = split
        self.transform = transform
        self.target_defect_type = defect_type
        self.dilate_mask = dilate_mask
        self.mask_kernel_size = mask_kernel_size
        self.max_transform_retries = max_transform_retries
        
        print(f"Initializing Dataset: root_dir={self.root_dir}, object_class={object_class}, split={split}")
        if defect_type:
            print(f"Target defect type: {defect_type}")

        # Identify defect types
        self.defect_types = []
        self.layout = "custom"
        if split == "train":
            defect_path = os.path.join(self.root_dir, object_class, "train", "defective")
            print(f"Searching for defect types in: {defect_path}")
            if os.path.exists(defect_path):
                all_defect_types = [d for d in os.listdir(defect_path) if os.path.isdir(os.path.join(defect_path, d))]
                
                # If a specific type is requested, only use that
                if defect_type and defect_type in all_defect_types:
                    self.defect_types = [defect_type]
                    print(f"Using specified defect type: {self.defect_types}")
                elif defect_type:
                    print(f"Warning: Requested defect type '{defect_type}' not found. Available: {all_defect_types}")
                    self.defect_types = all_defect_types
                else:
                    self.defect_types = all_defect_types
                print(f"Found defect types: {self.defect_types}")
            else:
                standard_test_path = os.path.join(self.root_dir, object_class, "test")
                standard_gt_path = os.path.join(self.root_dir, object_class, "ground_truth")
                print(f"Custom training directory not found; checking standard MVTec layout: {standard_test_path}")
                if os.path.exists(standard_test_path) and os.path.exists(standard_gt_path):
                    self.layout = "standard_mvtec"
                    all_defect_types = [
                        d for d in os.listdir(standard_test_path)
                        if d != "good" and os.path.isdir(os.path.join(standard_test_path, d))
                    ]
                    if defect_type and defect_type in all_defect_types:
                        self.defect_types = [defect_type]
                    elif defect_type:
                        print(f"Warning: Requested defect type '{defect_type}' not found. Available: {all_defect_types}")
                        self.defect_types = all_defect_types
                    else:
                        self.defect_types = all_defect_types
                    print(f"Found standard MVTec defect types: {self.defect_types}")
                else:
                    print(f"Warning: Directory does not exist {defect_path}")

                    # Debugging helper: Check parent directory
                    parent_dir = os.path.dirname(defect_path)
                    if os.path.exists(parent_dir):
                        print(f"Parent directory {parent_dir} exists, containing: {os.listdir(parent_dir)}")

                    if os.path.exists(self.root_dir):
                        root_contents = os.listdir(self.root_dir)
                        print(f"Root {self.root_dir} exists, containing: {root_contents[:5]}... ({len(root_contents)} items total)")

        # Load image and mask paths
        self.images = []
        self.masks = []
        
        if pairs is not None:
            for pair in pairs:
                image_path = pair["image"]
                mask_path = pair["mask"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.root_dir, *image_path.replace("\\", "/").split("/"))
                if not os.path.isabs(mask_path):
                    mask_path = os.path.join(self.root_dir, *mask_path.replace("\\", "/").split("/"))
                self.images.append(os.path.normpath(image_path))
                self.masks.append(os.path.normpath(mask_path))

        # Handle Test Split (only loads the 'good' directory for anomaly detection baselines)
        elif split == "test":
            good_dir = os.path.join(self.root_dir, object_class, "test", "good")
            print(f"Loading test set from: {good_dir}")
            if os.path.exists(good_dir):
                good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"Found {len(good_files)} 'good' samples in test set")
                
                for good_file in good_files:
                    self.images.append(os.path.join(good_dir, good_file))
                    # Masks are generated randomly for good images during training/inference
                    self.masks.append(None)
            else:
                print(f"Warning: Test directory not found {good_dir}")
                
        # Load Defect Images (Train Split)
        else:
            for defect_type in self.defect_types:
                if self.layout == "standard_mvtec":
                    img_dir = os.path.join(self.root_dir, object_class, "test", defect_type)
                    mask_dir = os.path.join(self.root_dir, object_class, "ground_truth", defect_type)
                else:
                    img_dir = os.path.join(self.root_dir, object_class, "train", "defective", defect_type)
                    mask_dir = os.path.join(self.root_dir, object_class, "train", "defective_masks", defect_type)
                
                print(f"Processing defect: {defect_type}")
                print(f"  Images: {img_dir}")
                print(f"  Masks:  {mask_dir}")
                
                if os.path.exists(img_dir) and os.path.exists(mask_dir):
                    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"  Found {len(img_files)} image files")
                    
                    matched = 0
                    for img_file in img_files:
                        img_path = os.path.join(img_dir, img_file)
                        
                        # Convention: mask is base_name + _mask.png
                        base_name = os.path.splitext(img_file)[0]
                        mask_file = f"{base_name}_mask.png"
                        mask_path = os.path.join(mask_dir, mask_file)
                        
                        # Fallback for alternative mask naming patterns
                        if not os.path.exists(mask_path):
                            possible_masks = [f for f in os.listdir(mask_dir) if base_name in f]
                            if possible_masks:
                                mask_path = os.path.join(mask_dir, possible_masks[0])
                            else:
                                print(f"  Warning: No mask found for {img_file}, skipping")
                                continue
                        
                        self.images.append(img_path)
                        self.masks.append(mask_path)
                        matched += 1
                    
                    print(f"  Successfully paired {matched} image-mask sets")
        
        print(f"Total loaded: {len(self.images)} {split} images")
        if len(self.images) == 0:
            print("Warning: Dataset is empty!")

    def __len__(self):
        return len(self.images)
    
    def generate_random_mask(self, image_size):
        """Generate random rectangular masks for object loss (integrity learning)"""
        mask = np.zeros(image_size, dtype=np.float32)
        num_rectangles = 30
        
        h, w = image_size
        for _ in range(num_rectangles):
            # Rectangle size between 3% and 25% of image dimensions
            min_size = int(min(h, w) * 0.03)
            max_size = int(min(h, w) * 0.25)
            
            rect_h = np.random.randint(min_size, max_size)
            rect_w = np.random.randint(min_size, max_size)
            
            y = np.random.randint(0, h - rect_h)
            x = np.random.randint(0, w - rect_w)
            
            mask[y:y+rect_h, x:x+rect_w] = 1.0
            
        return mask
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image (OpenCV loads BGR, convert to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask or generate a random one
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Dilate mask (only if enabled)
            if self.dilate_mask:
                # Ensure kernel size is odd, otherwise OpenCV crashes
                k_size = self.mask_kernel_size if self.mask_kernel_size % 2 == 1 else self.mask_kernel_size + 1
                
                kernel = np.ones((k_size, k_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            mask = mask.astype(np.float32) / 255.0  # Normalize to [0, 1]
        else:
            mask = self.generate_random_mask((image.shape[0], image.shape[1]))
        
        # Apply Albumentations transformations
        if self.transform:
            augmented = None
            for _ in range(self.max_transform_retries):
                candidate = self.transform(image=image, mask=mask)
                if mask_path is None or float(candidate['mask'].sum()) > 0:
                    augmented = candidate
                    break
            if augmented is None:
                augmented = build_eval_transform()(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Create background (masked image) for the inpainting input: I * (1 - M)
        background = image * (1 - mask)
        
        # Adjusted mask used for Object Loss calculation
        adjusted_mask = mask + 0.3 * (1 - mask) if mask_path is None else mask
        
        return {
            'image': image,
            'mask': mask,
            'background': background,
            'adjusted_mask': adjusted_mask,
            'is_defect': mask_path is not None,
            'object_class': self.object_class,
            'image_path': img_path,
            'mask_path': mask_path or "",
            'view_mode': 'full',
        }

def build_train_transform():
    return A.Compose([
        A.Resize(
            512,
            512,
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.RandomScale(
            scale_limit=(0.0, 0.125),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        ),
        A.RandomCrop(height=512, width=512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def build_eval_transform():
    return A.Compose([
        A.Resize(
            512,
            512,
            interpolation=cv2.INTER_AREA,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def get_data_loaders(
    root_dir,
    object_class,
    batch_size=4,
    defect_type=None,
    dilate_mask=False,
    mask_kernel_size=3,
    train_pairs=None,
    num_workers=4,
):
    """Creates training and testing DataLoaders with preprocessing pipelines"""
    
    # Training pipeline: Includes random scaling for better generalization
    train_transform = build_train_transform()
    
    # Test pipeline: Simple resize and normalize
    test_transform = build_eval_transform()
    
    train_dataset = MVTecDefectDataset(
        root_dir=root_dir,
        object_class=object_class,
        split="train",
        transform=train_transform,
        defect_type=defect_type,
        dilate_mask=dilate_mask,
        mask_kernel_size=mask_kernel_size,
        pairs=train_pairs,
    )
    
    test_dataset = MVTecDefectDataset(
        root_dir=root_dir,
        object_class=object_class,
        split="test",
        transform=test_transform,
        dilate_mask=dilate_mask,
        mask_kernel_size=mask_kernel_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
