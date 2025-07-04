import os
import requests
from PIL import Image
import json
from pathlib import Path

class DataManager:
    def __init__(self, project_root="./"):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.images_dir = self.data_dir / "images"
        self.text_dir = self.data_dir / "text"
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
    
    def download_sample_images(self):
        """Download sample images for testing"""
        sample_urls = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Nature
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",   # Cat
            "https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=400",    # City
        ]
        
        filenames = ["nature_scene.jpg", "cute_cat.jpg", "city_view.jpg"]
        
        print("📥 Downloading sample images...")
        for url, filename in zip(sample_urls, filenames):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    filepath = self.images_dir / filename
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Downloaded {filename}")
                else:
                    print(f"❌ Failed to download {filename}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
    
    def create_sample_text_prompts(self):
        """Create sample text prompts for testing"""
        sample_prompts = {
            "nature_prompts": [
                "a tranquil forest scene with morning mist",
                "sunlight filtering through tall trees",
                "a peaceful natural landscape",
                "wildlife in their natural habitat"
            ],
            "urban_prompts": [
                "a bustling city street at night",
                "modern architecture and skyscrapers",
                "urban life and city culture",
                "metropolitan landscape"
            ],
            "creative_prompts": [
                "an artistic interpretation of emotions",
                "abstract concepts made visual",
                "creative expression through imagery",
                "imaginative and surreal scenes"
            ]
        }
        
        # Save to JSON file
        prompts_file = self.text_dir / "sample_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(sample_prompts, f, indent=2)
        
        print("✅ Sample text prompts created")
        return sample_prompts
    
    def get_available_images(self):
        """Get list of available images"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        images = []
        
        for file_path in self.images_dir.iterdir():
            if file_path.suffix.lower() in image_extensions:
                images.append(str(file_path))
        
        return images
    
    def prepare_demo_data(self):
        """Prepare all demo data"""
        print("🔄 Preparing demo data...")
        self.download_sample_images()
        self.create_sample_text_prompts()
        print("🎉 Demo data ready!")

if __name__ == "__main__":
    dm = DataManager()
    dm.prepare_demo_data()
