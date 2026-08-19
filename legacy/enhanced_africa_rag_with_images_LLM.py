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

class EnhancedRAGEngineWithLLM:
    """موتور RAG بهبودیافته با فیلتر جغرافیایی آفریقا و نمایش تصاویر و LLM"""
    
    def __init__(self, excel_path: str, npz_path: str, images_base_path: str = "", use_llm_enhancement: bool = True):
        # بارگذاری داده‌ها
        self.excel_data = pd.read_excel(excel_path)
        self.npz_data = np.load(npz_path)
        self.images_base_path = images_base_path
        
        # ایجاد فیلتر جغرافیایی
        self.geo_filter = AfricaGeoFilter()
        
        # ایجاد enhancer در صورت نیاز
        self.use_llm_enhancement = use_llm_enhancement
        self.llm_enhancer = None
        self.enhanced_descriptions_cache = {}
        
        if use_llm_enhancement:
            try:
                self.llm_enhancer = LLMImageEnhancer()
                print("✅ LLM Enhancement enabled")
            except Exception as e:
                print(f"⚠️ LLM Enhancement disabled: {e}")
                self.use_llm_enhancement = False
        
        # ادغام داده‌ها با فیلتر پیشرفته
        self.images_data = self._enhanced_merge_data()
        
        print(f"✅ Enhanced RAG Engine with LLM loaded: {len(self.images_data)} African images")
    
    def _enhanced_merge_data(self) -> List[Dict]:
        """ادغام پیشرفته داده‌ها با استخراج اطلاعات کامل"""
        merged_data = []
        skipped_non_africa = 0
        skipped_invalid_coords = 0
        
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
                
                merged_data.append({
                    'filename': filename,
                    'blip_description': self.npz_data['texts'][i],
                    'vector': self.npz_data['vectors'][i],
                    'latitude': lat,
                    'longitude': lon,
                    'title': title,
                    'description': description,
                    'location': location,
                    'keywords': tags,
                    'country': country,
                    'full_text': f"{title} {description} {location} {tags} {self.npz_data['texts'][i]}",
                    'coordinates_valid': is_valid,
                    'image_path': os.path.join(self.images_base_path, filename) if self.images_base_path else ""
                })
        
        print(f"📊 Filtering results:")
        print(f"   - African images: {len(merged_data)}")
        print(f"   - Non-African skipped: {skipped_non_africa}")
        print(f"   - Invalid coordinates skipped: {skipped_invalid_coords}")
        
        return merged_data
    
    def _enhance_map_descriptions(self, results: List[Dict]):
        """افزایش توضیحات برای تصاویر نقشه با LLM"""
        if not self.use_llm_enhancement or not self.llm_enhancer:
            return
        
        print("🔄 Enhancing descriptions for map images with LLM...")
        
        # فقط برای تصاویری که مختصات معتبر دارند و هنوز افزایش نیافته‌اند
        images_to_enhance = []
        for result in results:
            if (result['coordinates_valid'] and 
                result['filename'] not in self.enhanced_descriptions_cache and
                os.path.exists(result.get('image_path', ''))):
                images_to_enhance.append(result)
        
        if not images_to_enhance:
            print("✅ All map images already enhanced")
            return
        
        print(f"🔄 Enhancing {len(images_to_enhance)} new images...")
        
        # افزایش توضیحات
        enhanced_descriptions = self.llm_enhancer.batch_enhance_descriptions(images_to_enhance)
        
        # ذخیره در کش
        self.enhanced_descriptions_cache.update(enhanced_descriptions)
        
        print(f"✅ Enhanced {len(enhanced_descriptions)} image descriptions")
    
    def get_africa_stats(self) -> Dict[str, Any]:
        """آمار مربوط به داده‌های آفریقا"""
        countries = [item['country'] for item in self.images_data if item['country'] != 'Unknown']
        country_counts = pd.Series(countries).value_counts()
        
        # مختصات معتبر
        valid_coords = [(item['latitude'], item['longitude']) 
                       for item in self.images_data if item['coordinates_valid']]
        
        return {
            'total_images': len(self.images_data),
            'unique_countries': len(country_counts),
            'country_distribution': country_counts.to_dict(),
            'valid_coordinates_count': len(valid_coords),
            'coverage_percentage': (len(valid_coords) / len(self.images_data)) * 100,
            'llm_enhancement': self.use_llm_enhancement,
            'enhanced_descriptions_count': len(self.enhanced_descriptions_cache)
        }
    
    def _get_image_base64(self, image_path: str, max_size: tuple = (200, 150)) -> str:
        """تبدیل تصویر به base64 با اندازه بهینه"""
        try:
            if not os.path.exists(image_path):
                return ""
            
            with Image.open(image_path) as img:
                # تغییر اندازه تصویر
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # تبدیل به base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"خطا در پردازش تصویر {image_path}: {e}")
            return ""
    
    def _create_popup_html(self, result: Dict) -> str:
        """ایجاد HTML برای پاپ‌آپ نقشه"""
        # آماده کردن تصویر
        image_html = ""
        if result.get('image_path') and os.path.exists(result['image_path']):
            img_base64 = self._get_image_base64(result['image_path'])
            if img_base64:
                image_html = f'<div style="text-align: center; margin-bottom: 10px;"><img src="{img_base64}" style="max-width: 200px; max-height: 150px; border-radius: 5px; border: 2px solid #ddd;"></div>'
        
        # استفاده از توضیحات پیشرفته اگر موجود است
        enhanced_desc = self.enhanced_descriptions_cache.get(result['filename'], result['blip_description'])
        
        # بررسی آیا توضیح از LLM آمده
        is_enhanced = result['filename'] in self.enhanced_descriptions_cache
        description_label = "🤖 LLM Enhanced Description" if is_enhanced else "📷 BLIP Description"
        
        # ایجاد HTML پاپ‌آپ
        popup_content = f"""
        <div style="width: 400px; font-family: Arial, sans-serif; max-height: 500px; overflow-y: auto;">
            {image_html}
            <div style="padding: 12px;">
                <h3 style="color: #2c3e50; margin-bottom: 12px; border-bottom: 2px solid #3498db; padding-bottom: 8px; font-size: 16px;">{result['title']}</h3>
                
                <div style="margin-bottom: 10px; font-size: 13px;">
                    <strong>📍 Location:</strong> {result['location']}<br>
                    <strong>🇺🇳 Country:</strong> {result['country']}<br>
                    <strong>📏 Coordinates:</strong> {result['latitude']:.4f}, {result['longitude']:.4f}
                </div>
                
                <div style="margin-bottom: 10px; background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db;">
                    <strong>{description_label}:</strong><br>
                    <em style="font-size: 12px; line-height: 1.4;">{enhanced_desc}</em>
                </div>
                
                <div style="margin-bottom: 10px; font-size: 12px;">
                    <strong>📝 Original Description:</strong><br>
                    {result['description'][:150]}{'...' if len(result['description']) > 150 else ''}
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
                    {is_enhanced and '<br><span style="color: #27ae60;">✓ AI Enhanced</span>' or ''}
                </div>
            </div>
        </div>
        """
        return popup_content
    
    def search_animal(self, animal_name: str) -> List[Dict]:
        """جستجوی حیوان خاص در داده‌های آفریقا"""
        animal_name = animal_name.lower()
        results = []
        
        for item in self.images_data:
            # جستجو در تمام فیلدهای متنی
            search_text = item['full_text'].lower()
            
            if animal_name in search_text:
                # محاسبه امتیاز شباهت
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
                    'coordinates_valid': item['coordinates_valid']
                })
        
        # مرتب‌سازی بر اساس امتیاز
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _extract_context(self, text: str, animal: str) -> str:
        """استخراج جمله‌ای که حیوان در آن ذکر شده"""
        sentences = text.split('.')
        for sentence in sentences:
            if animal in sentence:
                return sentence.strip() + '.'
        return ""
    
    def generate_map(self, results: List[Dict], output_path: str = "africa_animal_map_llm.html"):
        """تولید نقشه تعاملی آفریقا با تصاویر و توضیحات LLM"""
        # فیلتر نتایج با مختصات معتبر
        valid_results = [r for r in results if r['latitude'] and r['longitude'] and 
                        not np.isnan(r['latitude']) and not np.isnan(r['longitude'])]
        
        if not valid_results:
            print("⚠️ No results with valid geographic coordinates found")
            return None
        
        # افزایش توضیحات با LLM برای نتایج نقشه
        self._enhance_map_descriptions(valid_results)
            
        # ایجاد نقشه پایه - مرکز آفریقا
        m = folium.Map(location=[0, 20], zoom_start=3)
        
        # اضافه کردن نشانگر برای هر نتیجه
        for result in valid_results:
            # ایجاد پاپ‌آپ با HTML پیشرفته
            popup_html = self._create_popup_html(result)
            iframe = IFrame(popup_html, width=420, height=500)
            popup = folium.Popup(iframe, max_width=500)
            
            # انتخاب رنگ نشانگر بر اساس امتیاز
            if result['score'] >= 5:
                icon_color = 'red'
                icon_type = 'star'
            elif result['score'] >= 3:
                icon_color = 'orange'
                icon_type = 'info-sign'
            else:
                icon_color = 'green'
                icon_type = 'camera'
            
            # بررسی آیا توضیح افزایش یافته است
            is_enhanced = result['filename'] in self.enhanced_descriptions_cache
            
            folium.Marker(
                [result['latitude'], result['longitude']],
                popup=popup,
                tooltip=f"{result['title']} - Score: {result['score']} {'🤖' if is_enhanced else ''}",
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
            ).add_to(m)
        
        # اضافه کردن لایه کنترل
        folium.LayerControl().add_to(m)
        
        # اضافه کردن عنوان نقشه
        title_html = '''
        <h3 align="center" style="font-size:20px; margin-top:10px; background: white; padding: 10px; border-radius: 5px;">
        🦁 African Wildlife Distribution Map with AI Descriptions</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # اضافه کردن legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; background: white; padding: 10px; border: 2px solid grey; z-index: 9999; font-size: 14px;">
        <p><strong>Map Legend</strong></p>
        <p>🟥 High Relevance (Score ≥ 5)</p>
        <p>🟧 Medium Relevance (Score 3-4)</p>
        <p>🟩 Low Relevance (Score < 3)</p>
        <p>🤖 AI Enhanced Description</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # ذخیره نقشه
        m.save(output_path)
        print(f"🗺️ Interactive AI-enhanced map created: {output_path}")
        return output_path
    
    def generate_report(self, animal: str, results: List[Dict]) -> str:
        """تولید گزارش تحلیل"""
        if not results:
            return f"No information found about '{animal}' in African data."
        
        # تحلیل داده‌ها - فیلتر مقادیر نامعتبر
        countries = [r['country'] for r in results 
                    if r['country'] != 'Unknown' 
                    and not pd.isna(r['country']) 
                    and isinstance(r['country'], str)]
        
        country_counts = {}
        for country in countries:
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # تعداد توضیحات افزایش یافته
        enhanced_count = sum(1 for r in results if r['filename'] in self.enhanced_descriptions_cache)
        
        # تولید گزارش انگلیسی
        report = f"""
# Analysis Report: {animal.title()} in Africa

## 📊 Statistical Summary
- Total Observations: {len(results)}
- Countries Observed: {len(set(countries))}
- Most Frequent Location: {max(country_counts, key=country_counts.get) if country_counts else 'Unknown'}
- AI Enhanced Descriptions: {enhanced_count}/{len(results)} ({enhanced_count/len(results)*100:.1f}%)

## 🗺️ Geographical Distribution in Africa
"""
        
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {country}: {count} observations\n"
        
        report += f"""
## 📝 Key Samples with AI Descriptions
"""
        
        for i, result in enumerate(results[:5]):
            enhanced_desc = self.enhanced_descriptions_cache.get(result['filename'], result['blip_description'])
            is_enhanced = result['filename'] in self.enhanced_descriptions_cache
            
            report += f"""
### Sample {i+1} {'🤖' if is_enhanced else ''}
- **Location**: {result['country']} (Lat: {result['latitude']:.4f}, Lon: {result['longitude']:.4f})
- **Title**: {result['title']}
- **AI Description**: {enhanced_desc}
- **Original Description**: {result['description'][:100]}{'...' if len(result['description']) > 100 else ''}
- **Tags**: {result['keywords']}
- **Relevance Score**: {result['score']}
"""
        
        # فیلتر کشورهای معتبر برای بخش توصیه‌ها
        valid_countries = [str(country) for country in country_counts.keys() 
                          if country and not pd.isna(country)]
        
        report += f"""
## 💡 Conservation Recommendations for Africa
Based on {len(results)} observations, this species is most frequently observed in {', '.join(valid_countries[:3]) if valid_countries else 'various'} African regions.
Conservation programs should focus on these key areas.

## 🎯 Key Insights
- **Primary Habitats**: {', '.join(list(country_counts.keys())[:3])}
- **Observation Density**: {len(results)} total sightings across Africa
- **Regional Focus**: {max(country_counts, key=country_counts.get) if country_counts else 'N/A'} shows highest concentration
- **AI Enhancement**: {enhanced_count} images enriched with detailed descriptions
"""
        
        return report
    
    def process_query(self, animal_query: str):
        """پردازش کامل پرسش کاربر"""
        print(f"🔍 Searching Africa for: {animal_query}")
        
        # جستجو در داده‌ها
        results = self.search_animal(animal_query)
        
        if not results:
            return {
                'status': 'not_found',
                'message': f'No information found about "{animal_query}" in African data.'
            }
        
        # تولید نقشه
        map_path = self.generate_map(results, f"africa_{animal_query.replace(' ', '_')}_map_llm.html")
        
        # تولید گزارش
        report = self.generate_report(animal_query, results)
        
        return {
            'status': 'success',
            'animal': animal_query,
            'results_count': len(results),
            'enhanced_count': sum(1 for r in results if r['filename'] in self.enhanced_descriptions_cache),
            'map_path': map_path,
            'report': report,
            'sample_results': results[:3]
        }

def main():
    # مسیر فایل‌ها
    EXCEL_PATH = "African.xls"
    NPZ_PATH = "results/image_embeddings.npz"
    IMAGES_BASE_PATH = "D:/Part 1/500px/ALL_City/African/task_537_part01"  # مسیر فولدر تصاویر
    
    if not os.path.exists(EXCEL_PATH) or not os.path.exists(NPZ_PATH):
        print("❌ Data files not found!")
        return
    
    # ایجاد موتور RAG بهبودیافته با LLM
    print("🚀 Initializing Enhanced RAG Engine with LLM...")
    rag_engine = EnhancedRAGEngineWithLLM(EXCEL_PATH, NPZ_PATH, IMAGES_BASE_PATH, use_llm_enhancement=True)
    
    # دریافت آمار آفریقا
    stats = rag_engine.get_africa_stats()
    print("\n📊 African Data Statistics:")
    print(f"   - African images: {stats['total_images']}")
    print(f"   - Unique countries: {stats['unique_countries']}")
    print(f"   - Geographic coverage: {stats['coverage_percentage']:.1f}%")
    print(f"   - LLM Enhancement: {'Enabled' if stats['llm_enhancement'] else 'Disabled'}")
    print(f"   - Pre-enhanced descriptions: {stats['enhanced_descriptions_count']}")
    
    # توزیع کشورها
    print("\n🗺️ Top Country Distribution:")
    for country, count in list(stats['country_distribution'].items())[:10]:
        print(f"   - {country}: {count} images")
    
    # تست با پرسش نمونه
    query = "lion"  # ابتدا با "lion" تست کنید
    result = rag_engine.process_query(query)
    
    # نمایش نتایج
    print("\n" + "="*60)
    print(f"Results for: {query}")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"✅ Number of results: {result['results_count']}")
        print(f"🤖 AI Enhanced descriptions: {result['enhanced_count']}")
        if result['map_path']:
            print(f"🗺️ Interactive AI map created: {result['map_path']}")
        print("\n📄 Report:")
        print(result['report'])
        
        print("\n🔍 Sample results:")
        for i, sample in enumerate(result['sample_results']):
            is_enhanced = sample['filename'] in rag_engine.enhanced_descriptions_cache
            print(f"{i+1}. {sample['title']} - {sample['country']} {'🤖' if is_enhanced else ''}")
            enhanced_desc = rag_engine.enhanced_descriptions_cache.get(sample['filename'], sample['blip_description'])
            print(f"   AI Description: {enhanced_desc[:80]}...")
            print(f"   Score: {sample['score']}")
            print(f"   Location: {sample['location']}")
            print()
    else:
        print(result['message'])

if __name__ == "__main__":
    main()