# model_manager.py - برای مدیریت مدل‌ها
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

def download_models_offline(models_dir="./models"):
    """دانلود مدل‌ها برای استفاده آفلاین"""
    os.makedirs(models_dir, exist_ok=True)
    
    blip_path = os.path.join(models_dir, "blip-image-captioning-base")
    clip_path = os.path.join(models_dir, "clip-vit-base-patch32")
    
    print("شروع دانلود مدل‌ها...")
    
    # دانلود BLIP
    if not os.path.exists(blip_path):
        print("📥 دانلود مدل BLIP...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        processor.save_pretrained(blip_path)
        model.save_pretrained(blip_path)
        print("✅ BLIP دانلود و ذخیره شد")
    
    # دانلود CLIP
    if not os.path.exists(clip_path):
        print("📥 دانلود مدل CLIP...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor.save_pretrained(clip_path)
        model.save_pretrained(clip_path)
        print("✅ CLIP دانلود و ذخیره شد")
    
    print("🎉 تمام مدل‌ها آماده استفاده آفلاین هستند!")

if __name__ == "__main__":
    download_models_offline()