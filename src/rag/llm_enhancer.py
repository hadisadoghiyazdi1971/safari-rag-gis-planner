import os
import requests
import json
import time
from typing import Dict, List, Optional
from PIL import Image
import base64
import io
from dotenv import load_dotenv

class LLMImageEnhancer:
    """افزایش توضیحات تصاویر با استفاده از LLM"""
    
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        
        # دریافت کلید API از محیط
        self.api_key = os.getenv("OPENROUTER_API_KEY") 
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        print(f"✅ LLM Enhancer loaded with model: {self.llm_model}")
    
    def _encode_image_to_base64(self, image_path: str, max_size: tuple = (512, 512)) -> Optional[str]:
        """تبدیل تصویر به base64 با اندازه بهینه"""
        try:
            if not os.path.exists(image_path):
                return None
            
            with Image.open(image_path) as img:
                # تغییر اندازه برای کاهش حجم
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # تبدیل به base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return img_str
        except Exception as e:
            print(f"خطا در پردازش تصویر {image_path}: {e}")
            return None
    
    def _create_llm_prompt(self, blip_description: str, metadata: Dict) -> str:
        """ایجاد پرمپت برای LLM"""
        prompt = f"""
You are a wildlife photography expert analyzing an African wildlife image.

CURRENT BASIC DESCRIPTION: {blip_description}

ADDITIONAL CONTEXT:
- Location: {metadata.get('location', 'Not specified')}
- Country: {metadata.get('country', 'Not specified')}
- Tags: {metadata.get('keywords', 'No tags')}
- Original Title: {metadata.get('title', 'No title')}

Please provide a rich, detailed description of this wildlife photograph with the following structure:

1. **Main Subject & Action**: Describe the main animal and its behavior in detail
2. **Habitat & Environment**: Describe the setting, vegetation, time of day, lighting
3. **Context & Significance**: Mention ecological or conservation context if relevant

Requirements:
- Write 3-4 lines total (50-80 words)
- Be descriptive and engaging
- Use professional wildlife photography terminology
- Focus on visual elements and natural behavior
- Include specific details from the context when available

Respond only with the description text, no additional formatting.
"""
        return prompt
    
    def enhance_image_description(self, image_path: str, blip_description: str, metadata: Dict) -> Optional[str]:
        """افزایش توضیحات تصویر با LLM"""
        try:
            # تبدیل تصویر به base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                print(f"❌ Cannot process image: {image_path}")
                return None
            
            # ایجاد پرمپت
            prompt = self._create_llm_prompt(blip_description, metadata)
            
            # آماده کردن درخواست
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            # ارسال درخواست
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                enhanced_description = result['choices'][0]['message']['content'].strip()
                print(f"✅ Enhanced description generated for {os.path.basename(image_path)}")
                return enhanced_description
            else:
                print(f"❌ LLM API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error enhancing description: {e}")
            return None
    
    def batch_enhance_descriptions(self, image_data_list: List[Dict], delay: float = 1.0) -> Dict[str, str]:
        """افزایش توضیحات برای لیستی از تصاویر"""
        enhanced_descriptions = {}
        
        for i, item in enumerate(image_data_list):
            print(f"🔄 Enhancing {i+1}/{len(image_data_list)}: {item['filename']}")
            
            enhanced_desc = self.enhance_image_description(
                item['image_path'],
                item['blip_description'],
                {
                    'title': item['title'],
                    'location': item['location'],
                    'country': item['country'],
                    'keywords': item['keywords']
                }
            )
            
            if enhanced_desc:
                enhanced_descriptions[item['filename']] = enhanced_desc
            else:
                # استفاده از توضیح پیش‌فرض در صورت شکست
                enhanced_descriptions[item['filename']] = self._create_fallback_description(
                    item['blip_description'], 
                    item
                )
            
            # تاخیر برای جلوگیری از rate limit
            time.sleep(delay)
        
        return enhanced_descriptions
    
    def _create_fallback_description(self, blip_desc: str, metadata: Dict) -> str:
        """ایجاد توضیح جایگزین در صورت شکست LLM"""
        parts = [blip_desc.capitalize()]
        
        if metadata.get('location'):
            parts.append(f"Photographed at {metadata['location']}")
        
        if metadata.get('country') and metadata['country'] not in ['Unknown', 'Africa (Region)']:
            parts.append(f"in {metadata['country']}.")
        
        # اضافه کردن جزئیات بیشتر
        parts.append("This wildlife image captures natural behavior in the African wilderness.")
        
        return " ".join(parts)