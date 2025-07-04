import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

class MultimodalContentGenerator:
    def __init__(self):
        """Initialize the multimodal AI system with CLIP and BLIP models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP for image-text understanding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("✅ Multimodal models loaded successfully!")
    
    def analyze_image_text_similarity(self, image_path, text_descriptions):
        """
        Analyze similarity between an image and multiple text descriptions
        """
        try:
            # Load and preprocess image
            if image_path.startswith('http'):
                image = Image.open(BytesIO(requests.get(image_path).content))
            else:
                image = Image.open(image_path)
            
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize(text_descriptions).to(self.device)
            
            # Get similarity scores
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Calculate similarities
                similarities = torch.cosine_similarity(image_features, text_features)
                
            return similarities.cpu().numpy()
            
        except Exception as e:
            print(f"Error in similarity analysis: {e}")
            return None
    
    def generate_image_caption(self, image_path):
        """
        Generate detailed caption for an image using BLIP
        """
        try:
            # Load image
            if image_path.startswith('http'):
                image = Image.open(BytesIO(requests.get(image_path).content))
            else:
                image = Image.open(image_path)
            
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"Error in caption generation: {e}")
            return "Unable to generate caption"
    
    def suggest_mood_enhancements(self, base_description, target_mood):
        """
        Enhance content description based on target emotional mood
        """
        mood_mappings = {
            "upbeat": ["vibrant", "energetic", "bright", "cheerful", "dynamic"],
            "melancholy": ["soft", "muted", "contemplative", "gentle", "nostalgic"],
            "mysterious": ["shadowy", "enigmatic", "atmospheric", "dramatic", "intriguing"],
            "peaceful": ["serene", "calm", "tranquil", "harmonious", "soothing"],
            "adventurous": ["bold", "exciting", "rugged", "spirited", "thrilling"]
        }
        
        mood_words = mood_mappings.get(target_mood.lower(), ["enhanced", "improved"])
        enhanced_description = f"{base_description}, with {', '.join(mood_words[:3])} qualities"
        
        return enhanced_description
    
    def generate_soundscape_suggestions(self, image_caption, mood):
        """
        Suggest audio elements that would complement the visual content
        """
        soundscape_database = {
            "nature": ["forest ambience", "bird songs", "flowing water", "rustling leaves"],
            "urban": ["city traffic", "footsteps", "ambient noise", "distant conversations"],
            "peaceful": ["soft piano", "meditation bells", "gentle rainfall", "ocean waves"],
            "energetic": ["upbeat music", "rhythmic drums", "electronic beats", "lively instruments"]
        }
        
        # Simple keyword matching for soundscape suggestion
        suggestions = []
        caption_lower = image_caption.lower()
        
        for category, sounds in soundscape_database.items():
            if any(keyword in caption_lower for keyword in [category, mood.lower()]):
                suggestions.extend(sounds[:2])
        
        return suggestions if suggestions else ["ambient background music", "nature sounds"]

# Test the model
if __name__ == "__main__":
    generator = MultimodalContentGenerator()
    print("🎉 Multimodal AI system ready!")
