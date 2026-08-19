import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

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
            'congo', 'cote d\'ivoire', 'djibouti', 'egypt', 'equatorial guinea',
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
            'republic of congo': "Congo"
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
    """موتور RAG بهبودیافته با فیلتر جغرافیایی"""
    
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
                    'full_text': f"{title} {description} {location} {tags}",
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