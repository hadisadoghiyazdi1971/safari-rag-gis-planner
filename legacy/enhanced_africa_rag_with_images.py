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

class EnhancedRAGEngine:
    """موتور RAG بهبودیافته با فیلتر جغرافیایی آفریقا و نمایش تصاویر"""
    
    def __init__(self, excel_path: str, npz_path: str, images_base_path: str = ""):
        # بارگذاری داده‌ها
        self.excel_data = pd.read_excel(excel_path)
        self.npz_data = np.load(npz_path)
        self.images_base_path = images_base_path
        
        # ایجاد فیلتر جغرافیایی
        self.geo_filter = AfricaGeoFilter()
        
        # ادغام داده‌ها با فیلتر پیشرفته
        self.images_data = self._enhanced_merge_data()
        
        print(f"✅ Enhanced RAG Engine loaded: {len(self.images_data)} African images")
    
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
            'coverage_percentage': (len(valid_coords) / len(self.images_data)) * 100
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
        
        # ایجاد HTML پاپ‌آپ
        popup_content = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            {image_html}
            <div style="padding: 10px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">{result['title']}</h3>
                
                <div style="margin-bottom: 8px;">
                    <strong>📍 Location:</strong> {result['location']}<br>
                    <strong>🇺🇳 Country:</strong> {result['country']}<br>
                    <strong>📏 Coordinates:</strong> {result['latitude']:.4f}, {result['longitude']:.4f}
                </div>
                
                <div style="margin-bottom: 8px; background: #f8f9fa; padding: 8px; border-radius: 4px;">
                    <strong>📷 BLIP Description:</strong><br>
                    <em>{result['blip_description']}</em>
                </div>
                
                <div style="margin-bottom: 8px;">
                    <strong>📝 Original Description:</strong><br>
                    {result['description'][:150]}{'...' if len(result['description']) > 150 else ''}
                </div>
                
                <div style="margin-bottom: 8px;">
                    <strong>🏷️ Tags:</strong><br>
                    <span style="color: #e74c3c;">{result['keywords']}</span>
                </div>
                
                <div style="background: #e8f4fd; padding: 8px; border-radius: 4px; border-right: 4px solid #3498db;">
                    <strong>🔍 Search Match:</strong><br>
                    {result['search_match'] or 'Full image description'}
                </div>
                
                <div style="margin-top: 8px; text-align: right;">
                    <small style="color: #7f8c8d;">
                        <strong>Score:</strong> {result['score']} | 
                        <strong>File:</strong> {result['filename']}
                    </small>
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
                    'image_path': item['image_path']
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
    
    def generate_map(self, results: List[Dict], output_path: str = "africa_animal_map.html"):
        """تولید نقشه تعاملی آفریقا با تصاویر"""
        # فیلتر نتایج با مختصات معتبر
        valid_results = [r for r in results if r['latitude'] and r['longitude'] and 
                        not np.isnan(r['latitude']) and not np.isnan(r['longitude'])]
        
        if not valid_results:
            print("⚠️ هیچ نتیجه‌ای با مختصات جغرافیایی معتبر یافت نشد")
            return None
            
        # ایجاد نقشه پایه - مرکز آفریقا
        m = folium.Map(location=[0, 20], zoom_start=3)
        
        # اضافه کردن نشانگر برای هر نتیجه
        for result in valid_results:
            # ایجاد پاپ‌آپ با HTML پیشرفته
            popup_html = self._create_popup_html(result)
            iframe = IFrame(popup_html, width=350, height=450)
            popup = folium.Popup(iframe, max_width=500)
            
            # انتخاب رنگ نشانگر بر اساس امتیاز
            if result['score'] >= 5:
                icon_color = 'red'
            elif result['score'] >= 3:
                icon_color = 'orange'
            else:
                icon_color = 'green'
            
            folium.Marker(
                [result['latitude'], result['longitude']],
                popup=popup,
                tooltip=f"{result['title']} - Score: {result['score']}",
                icon=folium.Icon(color=icon_color, icon='camera', prefix='fa')
            ).add_to(m)
        
        # اضافه کردن لایه کنترل
        folium.LayerControl().add_to(m)
        
        # اضافه کردن عنوان نقشه
        title_html = '''
        <h3 align="center" style="font-size:20px; margin-top:10px;">
        🦁 African Wildlife Distribution Map</h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # ذخیره نقشه
        m.save(output_path)
        print(f"🗺️ نقشه تعاملی آفریقا ایجاد شد: {output_path}")
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
        
        # تولید گزارش انگلیسی
        report = f"""
# Analysis Report: {animal.title()} in Africa

## 📊 Statistical Summary
- Total Observations: {len(results)}
- Countries Observed: {len(set(countries))}
- Most Frequent Location: {max(country_counts, key=country_counts.get) if country_counts else 'Unknown'}

## 🗺️ Geographical Distribution in Africa
"""
        
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {country}: {count} observations\n"
        
        report += f"""
## 📝 Key Samples
"""
        
        for i, result in enumerate(results[:5]):
            report += f"""
### Sample {i+1}
- **Location**: {result['country']} (Lat: {result['latitude']:.4f}, Lon: {result['longitude']:.4f})
- **Title**: {result['title']}
- **BLIP Description**: {result['blip_description']}
- **Original Description**: {result['description'][:100]}{'...' if len(result['description']) > 100 else ''}
- **Tags**: {result['keywords']}
- **Search Match**: {result['search_match'] or 'Full image context'}
- **Relevance Score**: {result['score']}
"""
        
        # فیلتر کشورهای معتبر برای بخش توصیه‌ها
        valid_countries = [str(country) for country in country_counts.keys() 
                          if country and not pd.isna(country)]
        
        report += f"""
## 💡 Conservation Recommendations for Africa
Based on observations, this species is most frequently observed in {', '.join(valid_countries[:3]) if valid_countries else 'various'} African regions.
Conservation programs should focus on these key areas.

## 🎯 Key Insights
- **Primary Habitats**: {', '.join(list(country_counts.keys())[:3])}
- **Observation Density**: {len(results)} total sightings across Africa
- **Regional Focus**: {max(country_counts, key=country_counts.get) if country_counts else 'N/A'} shows highest concentration
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
        map_path = self.generate_map(results, f"africa_{animal_query.replace(' ', '_')}_map.html")
        
        # تولید گزارش
        report = self.generate_report(animal_query, results)
        
        return {
            'status': 'success',
            'animal': animal_query,
            'results_count': len(results),
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
    
    # ایجاد موتور RAG بهبودیافته
    rag_engine = EnhancedRAGEngine(EXCEL_PATH, NPZ_PATH, IMAGES_BASE_PATH)
    
    # دریافت آمار آفریقا
    stats = rag_engine.get_africa_stats()
    print("\n📊 African Data Statistics:")
    print(f"   - African images: {stats['total_images']}")
    print(f"   - Unique countries: {stats['unique_countries']}")
    print(f"   - Geographic coverage: {stats['coverage_percentage']:.1f}%")
    
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
        if result['map_path']:
            print(f"🗺️ Interactive map created: {result['map_path']}")
        print("\n📄 Report:")
        print(result['report'])
        
        print("\n🔍 Sample results:")
        for i, sample in enumerate(result['sample_results']):
            print(f"{i+1}. {sample['title']} - {sample['country']}")
            print(f"   Description: {sample['blip_description']}")
            print(f"   Score: {sample['score']}")
            print(f"   Location: {sample['location']}")
            print()
    else:
        print(result['message'])

if __name__ == "__main__":
    main()