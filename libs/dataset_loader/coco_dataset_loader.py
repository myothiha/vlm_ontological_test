import os
import cv2
import gc
import sys
import random
from dotenv import load_dotenv
from pycocotools.coco import COCO
from PIL import Image
from libs.dataset_loader.dataset import Dataset

class COCOLoader:
    def __init__(self):
        load_dotenv()
        self.annotation_path = os.getenv("COCO_ANNOTATION_PATH")
        self.image_dir = os.getenv("COCO_IMAGE_PATH")

        if not self.annotation_path or not os.path.exists(self.annotation_path):
            raise FileNotFoundError("❌ COCO_ANNOTATION_PATH is not set or file not found.")
        if not self.image_dir or not os.path.isdir(self.image_dir):
            raise FileNotFoundError("❌ COCO_IMAGE_PATH is not set or directory not found.")

        print(f"📂 Loading annotations from: {self.annotation_path}")
        print(f"🖼️  Loading images from: {self.image_dir}")
        self.coco = COCO(self.annotation_path)

    def get_image_ids(self):
        return self.coco.getImgIds()

    def load_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return image, image_info

    def get_annotations(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)

    def get_image_with_annotations(self, image_id):
        image, info = self.load_image(image_id)
        anns = self.get_annotations(image_id)
        return image, info, anns
    
    def get_detected_objects(self, image_id):
        annotations = self.get_annotations(image_id)
        category_ids = list(set([ann["category_id"] for ann in annotations]))
        categories = self.coco.loadCats(category_ids)
        return [cat["name"] for cat in categories]
    
    def get_all_categories(self):
        """
        Returns all category names in the dataset.
        """
        cat_ids = self.coco.getCatIds()
        categories = self.coco.loadCats(cat_ids)
        return [cat["name"] for cat in categories]

    def __len__(self):
        return len(self.get_image_ids())

    def __getitem__(self, idx):
        image_ids = self.get_image_ids()
        image_id = image_ids[idx]
        image, info = self.load_image(image_id)
        anns = self.get_annotations(image_id)
        # Get category names for this image
        category_ids = list(set([ann["category_id"] for ann in anns]))
        categories = self.coco.loadCats(category_ids)
        category_names = [cat["name"] for cat in categories]
        return {
            "image_id": image_id,
            "image": image,
            "text": category_names,
        }

    def get_all(self):
        """Return all items as a list of dicts."""
        data = [self[i] for i in range(len(self))]
        dataset = Dataset(data)
        return dataset

    def sample(self, n=1, seed=None):
        """Return n random rows from the dataset as a list of dicts."""
        if seed is not None:
            random.seed(seed)
        indices = random.sample(range(len(self)), min(n, len(self)))
        data = [self[i] for i in indices]
        dataset = Dataset(data)
        return dataset

    def sample_original_dataset(self, n=1, seed=None):
        """Return n random rows from the dataset as a list of dicts."""
        if seed is not None:
            random.seed(seed)
        indices = random.sample(range(len(self)), min(n, len(self)))
        data = [self[i] for i in indices]
        dataset = Dataset(data)
        return dataset


# ✅ Example usage when running directly
if __name__ == "__main__":
    print("🔧 Running COCOLoader demo...")

    try:
        loader = COCOLoader()
        image_ids = loader.get_image_ids()
        print(f"📸 Total images: {len(image_ids)}")

        # Load and print one sample
        sample_id = image_ids[0]
        _, info, anns = loader.get_image_with_annotations(sample_id)

        print("📝 Image Info:", info)
        print(f"🧩 Annotations for image {sample_id}:")
        for ann in anns:
            print(f"  - id: {ann['id']}, category: {ann['category_id']}, bbox: {ann['bbox']}")

        print("Fetching all categories.\n", loader.get_all_categories())

    except Exception as e:
        print(f"🚨 Error: {e}")

    # 🔻 Clean up + force exit
    print("🧹 Cleaning up...")
    del loader
    gc.collect()
    sys.exit(0)
