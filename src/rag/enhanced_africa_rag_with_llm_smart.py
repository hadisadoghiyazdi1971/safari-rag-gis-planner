import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import folium
from folium import IFrame
import base64
import os
import re
from PIL import Image
import io
from llm_enhancer import LLMImageEnhancer
from description_manager import DescriptionManager

class AfricaGeoFilter:
    """فیلتر جغرافیایی برای محدود کردن داده‌ها به قاره آفریقا"""
    
    def __init__(self):
        # مرزهای جغرافیایی آفریقا
        self.africa_bounds = {
            'min_lat': -35.0,  # جنوبی‌ترین نقطه
            'max_lat': 37.5,   # شمالی‌ترین نقطه  
            'min_lon': -18.0,  # غربی‌ترین نقطه
            'max_lon': 52.0    # شرقی‌ترین نقطه
        }
        
        # لیست کامل کشورهای آفریقایی
        self.african_countries = {
            'algeria', 'angola', 'benin', 'botswana', 'burkina faso', 'burundi',
            'cabo verde', 'cameroon', 'central african republic', 'chad', 'comoros',
            'congo', 'cote divoire', 'djibouti', 'egypt', 'equatorial guinea',
            'eritrea', 'eswatini', 'ethiopia', 'gabon', 'gambia', 'ghana', 'guinea',
            'guinea-bissau', 'kenya', 'lesotho', 'liberia', 'libya', 'madagascar',
            'malawi', 'mali', 'mauritania', 'mauritius', 'morocco', 'mozambique',
            'namibia', 'niger', 'nigeria', 'rwanda', 'sao tome and principe',
            'senegal', 'seychelles', 'sierra leone', 'somalia', 'south africa',
            'south sudan', 'sudan', 'tanzania', 'togo', 'tunisia', 'uganda',
            'zambia', 'zimbabwe'
        }
    
    def is_in_africa(self, lat: float, lon: float) -> bool:
        """بررسی آیا مختصات در محدوده آفریقا قرار دارد"""
        if pd.isna(lat) or pd.isna(lon):
            return False
        
        return (self.africa_bounds['min_lat'] <= lat <= self.africa_bounds['max_lat'] and
                self.africa_bounds['min_lon'] <= lon <= self.africa_bounds['max_lon'])
    
    def extract_country_advanced(self, text: str) -> str:
        """استخراج پیشرفته نام کشور از متن"""
        if pd.isna(text) or not isinstance(text, str):
            return "Unknown"
        
        text = text.lower().strip()
        
        # جستجوی مستقیم نام کشورها
        for country in self.african_countries:
            if country in text:
                return country.title()
        
        # جستجوی نام‌های متداول
        common_names = {
            'ivory coast': "Cote d'Ivoire",
            'cape verde': "Cabo Verde",
            'swaziland': "Eswatini",
            'dr congo': "Congo",
            'republic of congo': "Congo",
            'cote d\'ivoire': "Cote d'Ivoire"
        }
        
        for common_name, official_name in common_names.items():
            if common_name in text:
                return official_name
        
        return "Unknown"
    
    def validate_coordinates(self, lat: float, lon: float) -> Tuple[bool, str]:
        """اعتبارسنجی مختصات جغرافیایی"""
        if pd.isna(lat) or pd.isna(lon):
            return False, "مختصات نامعتبر"
        
        if not (-90 <= lat <= 90):
            return False, "عرض جغرافیایی نامعتبر"
        
        if not (-180 <= lon <= 180):
            return False, "طول جغرافیایی نامعتبر"
        
        if not self.is_in_africa(lat, lon):
            return False, "خارج از مرزهای آفریقا"
        
        return True, "معتبر"

class SmartRAGEngineWithLLM:
    """موتور RAG هوشمند با مدیریت ذخیره‌سازی و LLM"""
    
    def __init__(self, excel_path: str, npz_path: str, images_base_path: str = "", use_llm_enhancement: bool = True):
        # بارگذاری داده‌ها
        self.excel_data = pd.read_excel(excel_path)
        self.npz_data = np.load(npz_path)
        self.images_base_path = images_base_path
        
        # ایجاد فیلتر جغرافیایی
        self.geo_filter = AfricaGeoFilter()
        
        # مدیر ذخیره‌سازی
        self.desc_manager = DescriptionManager("africa_descriptions_database.json")
        
        # ایجاد enhancer در صورت نیاز
        self.use_llm_enhancement = use_llm_enhancement
        self.llm_enhancer = None
        
        if use_llm_enhancement:
            try:
                self.llm_enhancer = LLMImageEnhancer()
                print("✅ LLM Enhancement enabled")
            except Exception as e:
                print(f"⚠️ LLM Enhancement disabled: {e}")
                self.use_llm_enhancement = False
        
        # ادغام داده‌ها با فیلتر پیشرفته و ذخیره‌سازی BLIP
        self.images_data = self._enhanced_merge_data_with_storage()
        
        # نمایش آمار ذخیره‌سازی
        self._show_storage_stats()
        
        print(f"✅ Smart RAG Engine loaded: {len(self.images_data)} African images")
    
    def _enhanced_merge_data_with_storage(self) -> List[Dict]:
        """ادغام داده‌ها با ذخیره‌سازی خودکار BLIP"""
        merged_data = []
        skipped_non_africa = 0
        skipped_invalid_coords = 0
        new_blip_saved = 0
        
        for i, filename in enumerate(self.npz_data['filenames']):
            # یافتن ردیف متناظر در اکسل
            excel_row = self.excel_data[self.excel_data['File Name'] == filename]
            
            if not excel_row.empty:
                row = excel_row.iloc[0]
                
                # استخراج تمام فیلدهای متنی
                title = str(row.get('Name', '')) if not pd.isna(row.get('Name')) else ""
                description = str(row.get('Description', '')) if not pd.isna(row.get('Description')) else ""
                location = str(row.get('Location', '')) if not pd.isna(row.get('Location')) else ""
                tags = str(row.get('Tags', '')) if not pd.isna(row.get('Tags')) else ""
                country = str(row.get('Country', '')) if not pd.isna(row.get('Country')) else ""
                
                # مختصات جغرافیایی
                lat = row.get('Latitude', 0)
                lon = row.get('Longitude', 0)
                
                # اعتبارسنجی مختصات
                is_valid, validation_msg = self.geo_filter.validate_coordinates(lat, lon)
                
                if not is_valid:
                    skipped_invalid_coords += 1
                    continue
                
                # استخراج پیشرفته کشور
                if country == "Unknown" or not country:
                    search_text = f"{title} {description} {location} {tags}"
                    country = self.geo_filter.extract_country_advanced(search_text)
                
                # اگر کشور نامشخص است ولی مختصات در آفریقاست
                if country == "Unknown" and self.geo_filter.is_in_africa(lat, lon):
                    country = "Africa (Region)"
                
                # فقط داده‌های آفریقایی را نگه دار
                if country == "Unknown" and not self.geo_filter.is_in_africa(lat, lon):
                    skipped_non_africa += 1
                    continue
                
                # دریافت یا ذخیره توضیحات BLIP
                blip_desc = self.npz_data['texts'][i]
                stored_data = self.desc_manager.get_description(filename)
                
                if not stored_data['blip_description']:
                    # ذخیره BLIP اگر موجود نیست
                    self.desc_manager.save_blip_description(filename, blip_desc)
                    new_blip_saved += 1
                
                merged_data.append({
                    'filename': filename,
                    'blip_description': blip_desc,
                    'vector': self.npz_data['vectors'][i],
                    'latitude': lat,
                    'longitude': lon,
                    'title': title,
                    'description': description,
                    'location': location,
                    'keywords': tags,
                    'country': country,
                    'full_text': f"{title} {description} {location} {tags} {blip_desc}",
                    'coordinates_valid': is_valid,
                    'image_path': os.path.join(self.images_base_path, filename) if self.images_base_path else "",
                    'has_llm_description': stored_data['has_llm']
                })
        
        print(f"📊 Filtering results:")
        print(f"   - African images: {len(merged_data)}")
        print(f"   - Non-African skipped: {skipped_non_africa}")
        print(f"   - Invalid coordinates skipped: {skipped_invalid_coords}")
        print(f"   - New BLIP descriptions saved: {new_blip_saved}")
        
        return merged_data
    
    def _show_storage_stats(self):
        """نمایش آمار ذخیره‌سازی"""
        stats = self.desc_manager.get_stats()
        print(f"💾 Storage Statistics:")
        print(f"   - Total files in database: {stats['total_files']}")
        print(f"   - With LLM descriptions: {stats['with_llm']}")
        print(f"   - Without LLM descriptions: {stats['without_llm']}")
        print(f"   - LLM coverage: {stats['llm_coverage']:.1f}%")
    
    def _enhance_map_descriptions(self, results: List[Dict]):
        """افزایش توضیحات برای تصاویر نقشه با LLM به صورت هوشمند"""
        if not self.use_llm_enhancement or not self.llm_enhancer:
            print("⚠️ LLM enhancement not available")
            return
        
        # فایل‌هایی که LLM ندارند و تصویر موجود است
        files_to_enhance = []
        for result in results:
            if (result['coordinates_valid'] and 
                not result['has_llm_description'] and
                os.path.exists(result.get('image_path', ''))):
                files_to_enhance.append(result)
        
        if not files_to_enhance:
            print("✅ All map images already have LLM descriptions")
            return
        
        print(f"🔄 Enhancing {len(files_to_enhance)} images with LLM...")
        
        enhanced_count = 0
        failed_count = 0
        
        for i, result in enumerate(files_to_enhance):
            print(f"   Processing {i+1}/{len(files_to_enhance)}: {result['filename']}")
            
            try:
                enhanced_desc = self.llm_enhancer.enhance_image_description(
                    result['image_path'],
                    result['blip_description'],
                    {
                        'title': result['title'],
                        'location': result['location'],
                        'country': result['country'],
                        'keywords': result['keywords']
                    }
                )
                
                if enhanced_desc:
                    self.desc_manager.save_llm_description(result['filename'], enhanced_desc)
                    enhanced_count += 1
                    print(f"     ✅ Enhanced successfully")
                else:
                    failed_count += 1
                    print(f"     ❌ Enhancement failed")
                    
            except Exception as e:
                print(f"     ❌ Error: {e}")
                failed_count += 1
            
            # تاخیر برای جلوگیری از rate limit
            import time
            time.sleep(1.5)
        
        print(f"✅ Enhanced {enhanced_count} images, Failed: {failed_count}")
        self._show_storage_stats()
    
    def get_final_description(self, filename: str) -> Tuple[str, bool]:
        """دریافت توضیحات نهایی (ترجیحاً LLM)"""
        stored_data = self.desc_manager.get_description(filename)
        
        if stored_data['has_llm'] and stored_data['llm_description']:
            return stored_data['llm_description'], True
        else:
            return stored_data['blip_description'], False
    
    def _get_image_base64(self, image_path: str, max_size: tuple = (200, 150)) -> str:
        """تبدیل تصویر به base64 با اندازه بهینه"""
        try:
            if not os.path.exists(image_path):
                return ""
            
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"خطا در پردازش تصویر {image_path}: {e}")
            return ""
    
    def _create_popup_html(self, result: Dict) -> str:
        """ایجاد HTML برای پاپ‌آپ نقشه با نمایش هر دو توضیح"""
        # آماده کردن تصویر
        image_html = ""
        if result.get('image_path') and os.path.exists(result['image_path']):
            img_base64 = self._get_image_base64(result['image_path'])
            if img_base64:
                image_html = f'<div style="text-align: center; margin-bottom: 10px;"><img src="{img_base64}" style="max-width: 200px; max-height: 150px; border-radius: 5px; border: 2px solid #ddd;"></div>'
        
        # دریافت توضیحات نهایی
        final_desc, is_enhanced = self.get_final_description(result['filename'])
        
        # ایجاد HTML پاپ‌آپ
        popup_content = f"""
        <div style="width: 450px; font-family: Arial, sans-serif; max-height: 600px; overflow-y: auto;">
            {image_html}
            <div style="padding: 12px;">
                <h3 style="color: #2c3e50; margin-bottom: 12px; border-bottom: 2px solid #3498db; padding-bottom: 8px; font-size: 16px;">
                    {result['title']} {'🤖' if is_enhanced else ''}
                </h3>
                
                <div style="margin-bottom: 10px; font-size: 13px;">
                    <strong>📍 Location:</strong> {result['location']}<br>
                    <strong>🇺🇳 Country:</strong> {result['country']}<br>
                    <strong>📏 Coordinates:</strong> {result['latitude']:.4f}, {result['longitude']:.4f}
                </div>
                
                <div style="margin-bottom: 10px; background: #e8f6f3; padding: 10px; border-radius: 5px; border-left: 4px solid #27ae60;">
                    <strong>🤖 AI Enhanced Description:</strong><br>
                    <em style="font-size: 12px; line-height: 1.4;">{final_desc}</em>
                </div>
                
                <div style="margin-bottom: 10px; background: #f8f9fa; padding: 8px; border-radius: 4px; font-size: 11px;">
                    <strong>📷 Original BLIP Description:</strong><br>
                    {result['blip_description']}
                </div>
                
                <div style="margin-bottom: 10px; font-size: 12px;">
                    <strong>📝 Full Description:</strong><br>
                    {result['description'][:120]}{'...' if len(result['description']) > 120 else ''}
                </div>
                
                <div style="margin-bottom: 10px; font-size: 12px;">
                    <strong>🏷️ Tags:</strong><br>
                    <span style="color: #e74c3c; font-size: 11px;">{result['keywords']}</span>
                </div>
                
                <div style="background: #e8f4fd; padding: 8px; border-radius: 4px; border-right: 4px solid #3498db; font-size: 12px;">
                    <strong>🔍 Search Match:</strong><br>
                    {result['search_match'] or 'Full image context'}
                </div>
                
                <div style="margin-top: 10px; text-align: right; font-size: 11px; color: #7f8c8d;">
                    <strong>Relevance Score:</strong> {result['score']} | 
                    <strong>File:</strong> {result['filename'][:20]}...
                </div>
            </div>
        </div>
        """
        return popup_content

    # بقیه متدها (search_animal, generate_map, generate_report, process_query) 
    # مانند فایل قبلی هستند، فقط با تغییرات زیر:
    
    def search_animal(self, animal_name: str) -> List[Dict]:
        """جستجوی حیوان خاص در داده‌های آفریقا"""
        animal_name = animal_name.lower()
        results = []
        
        for item in self.images_data:
            search_text = item['full_text'].lower()
            
            if animal_name in search_text:
                score = search_text.count(animal_name)
                
                results.append({
                    'filename': item['filename'],
                    'title': item['title'],
                    'blip_description': item['blip_description'],
                    'description': item['description'],
                    'location': item['location'],
                    'keywords': item['keywords'],
                    'latitude': item['latitude'],
                    'longitude': item['longitude'],
                    'country': item['country'],
                    'score': score,
                    'search_match': self._extract_context(search_text, animal_name),
                    'image_path': item['image_path'],
                    'coordinates_valid': item['coordinates_valid'],
                    'has_llm_description': item['has_llm_description']
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _extract_context(self, text: str, animal: str) -> str:
        """استخراج جمله‌ای که حیوان در آن ذکر شده"""
        sentences = text.split('.')
        for sentence in sentences:
            if animal in sentence:
                return sentence.strip() + '.'
        return ""
    
    def generate_map(self, results: List[Dict], output_path: str = "africa_animal_map_smart.html"):
        """تولید نقشه تعاملی"""
        valid_results = [r for r in results if r['latitude'] and r['longitude'] and 
                        not np.isnan(r['latitude']) and not np.isnan(r['longitude'])]
        
        if not valid_results:
            print("⚠️ No results with valid geographic coordinates found")
            return None
        
        # افزایش توضیحات با LLM برای نتایج جدید
        self._enhance_map_descriptions(valid_results)
            
        # ایجاد نقشه
        m = folium.Map(location=[0, 20], zoom_start=3)
        
        for result in valid_results:
            popup_html = self._create_popup_html(result)
            iframe = IFrame(popup_html, width=470, height=600)
            popup = folium.Popup(iframe, max_width=500)
            
            # انتخاب رنگ و آیکون
            final_desc, is_enhanced = self.get_final_description(result['filename'])
            
            if result['score'] >= 5:
                icon_color = 'red'
            elif result['score'] >= 3:
                icon_color = 'orange'
            else:
                icon_color = 'green'
            
            icon_type = 'robot' if is_enhanced else 'camera'
            
            folium.Marker(
                [result['latitude'], result['longitude']],
                popup=popup,
                tooltip=f"{result['title']} - Score: {result['score']} {'🤖' if is_enhanced else ''}",
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
            ).add_to(m)
        
        # اضافه کردن کنترل‌ها و عنوان
        folium.LayerControl().add_to(m)
        
        title_html = '''
        <h3 align="center" style="font-size:20px; margin-top:10px; background: white; padding: 10px; border-radius: 5px;">
        🦁 African Wildlife Map - Smart LLM Descriptions</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        m.save(output_path)
        print(f"🗺️ Smart map created: {output_path}")
        return output_path
    
    def generate_report(self, animal: str, results: List[Dict]) -> str:
        """تولید گزارش تحلیل"""
        if not results:
            return f"No information found about '{animal}' in African data."
        
        # تحلیل داده‌ها
        countries = [r['country'] for r in results 
                    if r['country'] != 'Unknown' and not pd.isna(r['country'])]
        
        country_counts = {}
        for country in countries:
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # آمار LLM
        llm_count = sum(1 for r in results if r['has_llm_description'])
        storage_stats = self.desc_manager.get_stats()
        
        # تولید گزارش
        report = f"""
# Analysis Report: {animal.title()} in Africa

## 📊 Statistical Summary
- Total Observations: {len(results)}
- Countries Observed: {len(set(countries))}
- Most Frequent Location: {max(country_counts, key=country_counts.get) if country_counts else 'Unknown'}
- AI Enhanced Descriptions: {llm_count}/{len(results)} ({llm_count/len(results)*100:.1f}%)

## 💾 Storage Status
- Total in Database: {storage_stats['total_files']}
- With LLM: {storage_stats['with_llm']}
- LLM Coverage: {storage_stats['llm_coverage']:.1f}%

## 🗺️ Geographical Distribution
"""
        
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {country}: {count} observations\n"
        
        report += f"""
## 📝 Key Samples
"""
        
        for i, result in enumerate(results[:5]):
            final_desc, is_enhanced = self.get_final_description(result['filename'])
            
            report += f"""
### Sample {i+1} {'🤖' if is_enhanced else '📷'}
- **Location**: {result['country']} (Lat: {result['latitude']:.4f}, Lon: {result['longitude']:.4f})
- **Title**: {result['title']}
- **{'AI Description' if is_enhanced else 'BLIP Description'}: {final_desc}
- **Tags**: {result['keywords']}
- **Relevance Score**: {result['score']}
"""
        
        valid_countries = [str(country) for country in country_counts.keys() 
                          if country and not pd.isna(country)]
        
        report += f"""
## 💡 Recommendations
Based on {len(results)} observations across {len(valid_countries)} countries.
Focus conservation efforts in {', '.join(valid_countries[:3])}.

## 🔄 Next Steps
- Run again after purchasing credits to enhance remaining {storage_stats['without_llm']} images
- Current database automatically stores all descriptions for future use
"""
        
        return report
    
    def process_query(self, animal_query: str):
        """پردازش کامل پرسش کاربر"""
        print(f"🔍 Searching Africa for: {animal_query}")
        
        results = self.search_animal(animal_query)
        
        if not results:
            return {
                'status': 'not_found',
                'message': f'No information found about "{animal_query}" in African data.'
            }
        
        map_path = self.generate_map(results, f"africa_{animal_query.replace(' ', '_')}_map_smart.html")
        report = self.generate_report(animal_query, results)
        
        return {
            'status': 'success',
            'animal': animal_query,
            'results_count': len(results),
            'llm_enhanced_count': sum(1 for r in results if r['has_llm_description']),
            'map_path': map_path,
            'report': report,
            'sample_results': results[:3],
            'storage_stats': self.desc_manager.get_stats()
        }
    
    def export_database(self):
        """خروجی دیتابیس به CSV"""
        self.desc_manager.export_to_csv("africa_descriptions_export.csv")

def main():
    # مسیر فایل‌ها
    EXCEL_PATH = "African.xls"
    NPZ_PATH = "results/image_embeddings.npz"
    IMAGES_BASE_PATH = "D:/Part 1/500px/ALL_City/African/task_537_part01"
    
    if not os.path.exists(EXCEL_PATH) or not os.path.exists(NPZ_PATH):
        print("❌ Data files not found!")
        return
    
    # ایجاد موتور هوشمند
    print("🚀 Initializing Smart RAG Engine with LLM...")
    rag_engine = SmartRAGEngineWithLLM(EXCEL_PATH, NPZ_PATH, IMAGES_BASE_PATH, use_llm_enhancement=True)
    
    # تست با پرسش نمونه
    query = "lion"
    result = rag_engine.process_query(query)
    
    # نمایش نتایج
    print("\n" + "="*60)
    print(f"Results for: {query}")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"✅ Results: {result['results_count']}")
        print(f"🤖 AI Enhanced: {result['llm_enhanced_count']}")
        print(f"🗺️ Map: {result['map_path']}")
        print("\n📄 Report:")
        print(result['report'])
        
        # خروجی دیتابیس
        rag_engine.export_database()
        print(f"\n💾 Database exported to: africa_descriptions_export.csv")
        
    else:
        print(result['message'])

if __name__ == "__main__":
    main()