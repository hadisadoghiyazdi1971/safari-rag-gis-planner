import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import folium
import os
import re

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
    """موتور RAG بهبودیافته با فیلتر جغرافیایی آفریقا"""
    
    def __init__(self, excel_path: str, npz_path: str):
        # بارگذاری داده‌ها
        self.excel_data = pd.read_excel(excel_path)
        self.npz_data = np.load(npz_path)
        
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
                    'coordinates_valid': is_valid
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
                    'keywords': item['keywords'],
                    'latitude': item['latitude'],
                    'longitude': item['longitude'],
                    'country': item['country'],
                    'score': score,
                    'search_match': self._extract_context(search_text, animal_name)
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
        """تولید نقشه تعاملی آفریقا"""
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
            popup_text = f"""
            <b>{result['title']}</b><br>
            <i>{result['blip_description']}</i><br>
            کشور: {result['country']}<br>
            امتیاز: {result['score']}<br>
            تطابق: {result['search_match']}
            """
            
            folium.Marker(
                [result['latitude'], result['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=result['title'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # ذخیره نقشه
        m.save(output_path)
        print(f"🗺️ نقشه آفریقا ایجاد شد: {output_path}")
        return output_path
    
    def generate_report(self, animal: str, results: List[Dict]) -> str:
        """تولید گزارش تحلیل"""
        if not results:
            return f"هیچ اطلاعاتی درباره '{animal}' در داده‌های آفریقا یافت نشد."
        
        # تحلیل داده‌ها - فیلتر مقادیر نامعتبر
        countries = [r['country'] for r in results 
                    if r['country'] != 'Unknown' 
                    and not pd.isna(r['country']) 
                    and isinstance(r['country'], str)]
        
        country_counts = {}
        for country in countries:
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # تولید گزارش
        report = f"""
# گزارش تحلیل: {animal} در آفریقا

## 📊 خلاصه آماری
- تعداد مشاهدات: {len(results)}
- کشورهای مشاهده شده: {len(set(countries))}
- بیشترین مشاهده در: {max(country_counts, key=country_counts.get) if country_counts else 'نامشخص'}

## 🗺️ پراکنش جغرافیایی در آفریقا
"""
        
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {country}: {count} مشاهده\n"
        
        report += f"""
## 📝 نمونه‌های کلیدی
"""
        
        for i, result in enumerate(results[:5]):
            report += f"""
### نمونه {i+1}
- **موقعیت**: {result['country']} (عرض: {result['latitude']:.4f}, طول: {result['longitude']:.4f})
- **عنوان**: {result['title']}
- **توضیح خودکار**: {result['blip_description']}
- **تطابق**: {result['search_match'] or 'توضیح کامل در تصویر'}
- **کلمات کلیدی**: {result['keywords']}
"""
        
        # فیلتر کشورهای معتبر برای بخش توصیه‌ها
        valid_countries = [str(country) for country in country_counts.keys() 
                          if country and not pd.isna(country)]
        
        report += f"""
## 💡 توصیه‌های حفاظتی برای آفریقا
بر اساس مشاهدات، این گونه بیشتر در مناطق {', '.join(valid_countries[:3]) if valid_countries else 'مختلف'} آفریقا مشاهده شده است.
پیشنهاد می‌شود برنامه‌های حفاظتی در این مناطق متمرکز شود.
"""
        
        return report
    
    def process_query(self, animal_query: str):
        """پردازش کامل پرسش کاربر"""
        print(f"🔍 جستجو در آفریقا: {animal_query}")
        
        # جستجو در داده‌ها
        results = self.search_animal(animal_query)
        
        if not results:
            return {
                'status': 'not_found',
                'message': f'هیچ اطلاعاتی درباره "{animal_query}" در داده‌های آفریقا یافت نشد.'
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
    
    if not os.path.exists(EXCEL_PATH) or not os.path.exists(NPZ_PATH):
        print("❌ فایل‌های داده یافت نشد!")
        return
    
    # ایجاد موتور RAG بهبودیافته
    rag_engine = EnhancedRAGEngine(EXCEL_PATH, NPZ_PATH)
    
    # دریافت آمار آفریقا
    stats = rag_engine.get_africa_stats()
    print("\n📊 آمار داده‌های آفریقا:")
    print(f"   - تصاویر آفریقایی: {stats['total_images']}")
    print(f"   - کشورهای مختلف: {stats['unique_countries']}")
    print(f"   - پوشش جغرافیایی: {stats['coverage_percentage']:.1f}%")
    
    # توزیع کشورها
    print("\n🗺️ توزیع کشورهای برتر:")
    for country, count in list(stats['country_distribution'].items())[:10]:
        print(f"   - {country}: {count} تصویر")
    
    # تست با پرسش نمونه
    query = "lion"  # ابتدا با "lion" تست کنید
    result = rag_engine.process_query(query)
    
    # نمایش نتایج
    print("\n" + "="*60)
    print(f"نتایج برای: {query}")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"✅ تعداد نتایج: {result['results_count']}")
        if result['map_path']:
            print(f"🗺️ نقشه ایجاد شد: {result['map_path']}")
        print("\n📄 گزارش:")
        print(result['report'])
        
        print("\n🔍 نمونه نتایج:")
        for i, sample in enumerate(result['sample_results']):
            print(f"{i+1}. {sample['title']} - {sample['country']}")
            print(f"   توضیح: {sample['blip_description']}")
            print(f"   امتیاز: {sample['score']}")
            print()
    else:
        print(result['message'])

if __name__ == "__main__":
    main()