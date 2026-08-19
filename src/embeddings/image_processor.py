import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np

class ImageToTextAndVector:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # مسیرهای محلی برای مدل‌ها
        self.blip_path = os.path.join(models_dir, "blip-image-captioning-base")
        self.clip_path = os.path.join(models_dir, "clip-vit-base-patch32")
        
        # بارگذاری یا دانلود مدل‌ها
        self._load_models()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_model.to(self.device)
        self.clip_model.to(self.device)
        
        print(f"مدل‌ها از مسیر محلی بارگذاری شدند")
        print(f"دستگاه پردازش: {self.device}")

    def _download_models(self):
        """دانلود مدل‌ها در صورت عدم وجود"""
        print("در حال دانلود مدل‌ها...")
        
        # دانلود BLIP
        if not os.path.exists(self.blip_path):
            print("دانلود مدل BLIP...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            processor.save_pretrained(self.blip_path)
            model.save_pretrained(self.blip_path)
        
        # دانلود CLIP
        if not os.path.exists(self.clip_path):
            print("دانلود مدل CLIP...")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor.save_pretrained(self.clip_path)
            model.save_pretrained(self.clip_path)

    def _load_models(self):
        """بارگذاری مدل‌ها از مسیر محلی"""
        # ابتدا مدل‌ها را دانلود کن اگر وجود ندارند
        self._download_models()
        
        print("بارگذاری مدل‌ها از حافظه محلی...")
        
        # بارگذاری BLIP
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_path)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_path)
        
        # بارگذاری CLIP
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path)
        self.clip_model = CLIPModel.from_pretrained(self.clip_path)

    
    def image_to_text(self, image_path):
        """تبدیل تصویر به متن"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # خط ۷۱-۷۵ را به این صورت تغییر دهید:
            with torch.no_grad():
                output = self.blip_model.generate(
                    **inputs, 
                    max_length=150,    # افزایش قابل توجه
                    num_beams=6,       # افزایش جستجو
                    early_stopping=False,
                    length_penalty=2,
                    repetition_penalty=1.2,
                    temperature=0.8,   # اضافه کردن برای تنوع بیشتر
                    do_sample=True     # اضافه کردن برای خلاقیت بیشتر
                    no_repeat_ngram_size=3  # جلوگیری از تکرار
                )
            
            
            caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"خطا در پردازش تصویر {image_path}: {str(e)}")
            return None

    def image_to_vector(self, image_path):
        """تبدیل تصویر به بردار"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # نرمالایز کردن بردار
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"خطا در پردازش تصویر {image_path}: {str(e)}")
            return None

    def process_folder(self, folder_path, output_dir="./results"):
        """پردازش تمام تصاویر در یک فولدر"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # لیست فایل‌های تصویری
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not image_files:
            print("هیچ فایل تصویری در فولدر یافت نشد!")
            return results
        
        print(f"تعداد تصاویر برای پردازش: {len(image_files)}")
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(folder_path, filename)
            print(f"پردازش تصویر {i+1}/{len(image_files)}: {filename}")
            
            text_description = self.image_to_text(image_path)
            vector_embedding = self.image_to_vector(image_path)
            
            if text_description and vector_embedding is not None:
                result = {
                    'filename': filename,
                    'text': text_description,
                    'vector': vector_embedding,
                    'vector_dim': len(vector_embedding)
                }
                results.append(result)
                
                print(f"   متن: {text_description}")
                print(f"   بعد بردار: {len(vector_embedding)}")
                print("-" * 50)
        
        return results

    def save_results(self, results, output_dir="./results"):
        """ذخیره نتایج در فرمت‌های مختلف"""
        os.makedirs(output_dir, exist_ok=True)
        
        # ذخیره در NPZ
        filenames = [result['filename'] for result in results]
        texts = [result['text'] for result in results]
        vectors = np.array([result['vector'] for result in results])
        
        npz_path = os.path.join(output_dir, "image_embeddings.npz")
        np.savez(npz_path, filenames=filenames, texts=texts, vectors=vectors)
        
        # ذخیره در TXT برای خوانایی
        txt_path = os.path.join(output_dir, "image_descriptions.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"فایل: {result['filename']}\n")
                f.write(f"توضیح: {result['text']}\n")
                f.write(f"بعد بردار: {result['vector_dim']}\n")
                f.write("-" * 50 + "\n")
        
        # ذخیره بردارها در CSV
        csv_path = os.path.join(output_dir, "image_vectors.csv")
        np.savetxt(csv_path, vectors, delimiter=',')
        
        print(f"نتایج ذخیره شد در: {output_dir}")
        print(f"  - embeddings: {npz_path}")
        print(f"  - descriptions: {txt_path}")
        print(f"  - vectors: {csv_path}")

def check_models_exist(models_dir="./models"):
    """بررسی وجود مدل‌ها در سیستم"""
    blip_path = os.path.join(models_dir, "blip-image-captioning-base")
    clip_path = os.path.join(models_dir, "clip-vit-base-patch32")
    
    blip_exists = os.path.exists(blip_path)
    clip_exists = os.path.exists(clip_path)
    
    print(f"مدل BLIP موجود: {'✅' if blip_exists else '❌'}")
    print(f"مدل CLIP موجود: {'✅' if clip_exists else '❌'}")
    
    return blip_exists and clip_exists

# استفاده از کلاس
def main():
    # مسیر فولدر تصاویر
    #folder_path = "path/to/your/images"  # تغییر دهید
    folder_path = "D:/Part 1/500px/ALL_City/African/task_537_part01"
    if not os.path.exists(folder_path):
        print("فولدر تصاویر یافت نشد!")
        return
    
    # بررسی وجود مدل‌ها
    print("بررسی مدل‌های محلی...")
    models_exist = check_models_exist()
    
    if not models_exist:
        print("مدل‌ها برای اولین بار دانلود خواهند شد...")
        print("لطفا اتصال اینترنت داشته باشید")
    
    # ایجاد پردازشگر
    processor = ImageToTextAndVector()
    
    # پردازش تصاویر
    results = processor.process_folder(folder_path)
    
    # نمایش و ذخیره نتایج
    if results:
        print(f"\n✅ پردازش کامل شد! تعداد تصاویر پردازش شده: {len(results)}")
        processor.save_results(results)
    else:
        print("\n❌ هیچ تصویری پردازش نشد!")

if __name__ == "__main__":
    main()