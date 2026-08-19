from africa_geo_filter import AfricaGeoFilter, EnhancedRAGEngine
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import folium
import os
import re

class RAGEngine:
    def __init__(self, excel_path: str, npz_path: str):
        # Load data
        self.excel_data = pd.read_excel(excel_path)
        self.npz_data = np.load(npz_path)
        
        # Merge data
        self.images_data = self._merge_data()
        
        # LLM settings
        self.llm_client = None
        
        print(f"✅ RAG Engine loaded: {len(self.images_data)} images")
    
    def _extract_country_from_text(self, text: str) -> str:
        """Extract country name from text using African countries list"""
        if pd.isna(text) or not isinstance(text, str):
            return "Unknown"
            
        text = text.lower()
        
        # Comprehensive list of African countries
        african_countries = {
            'south africa': 'South Africa', 'nigeria': 'Nigeria', 'egypt': 'Egypt',
            'ethiopia': 'Ethiopia', 'kenya': 'Kenya', 'tanzania': 'Tanzania',
            'uganda': 'Uganda', 'morocco': 'Morocco', 'ghana': 'Ghana',
            'madagascar': 'Madagascar', 'zimbabwe': 'Zimbabwe', 'zambia': 'Zambia',
            'botswana': 'Botswana', 'namibia': 'Namibia', 'senegal': 'Senegal',
            'cameroon': 'Cameroon', 'ivory coast': 'Ivory Coast', 'tunisia': 'Tunisia',
            'libya': 'Libya', 'sudan': 'Sudan', 'algeria': 'Algeria', 'angola': 'Angola',
            'mozambique': 'Mozambique', 'rwanda': 'Rwanda', 'somalia': 'Somalia',
            'chad': 'Chad', 'mali': 'Mali', 'niger': 'Niger', 'malawi': 'Malawi'
        }
        
        for country_key, country_name in african_countries.items():
            if country_key in text:
                return country_name
        
        return "Unknown"
    
    # در کلاس RAGEngine، تابع _merge_data با نسخه بهبودیافته جایگزین شود
    def _merge_data(self) -> List[Dict]:
        """ادغام داده‌ها با فیلتر آفریقا"""
        merged_data = []
        geo_filter = AfricaGeoFilter()
        
        for i, filename in enumerate(self.npz_data['filenames']):
            excel_row = self.excel_data[self.excel_data['File Name'] == filename]
            
            if not excel_row.empty:
                row = excel_row.iloc[0]
                
                # استخراج تمام فیلدها
                title = str(row.get('Name', '')) if not pd.isna(row.get('Name')) else ""
                description = str(row.get('Description', '')) if not pd.isna(row.get('Description')) else ""
                location = str(row.get('Location', '')) if not pd.isna(row.get('Location')) else ""
                tags = str(row.get('Tags', '')) if not pd.isna(row.get('Tags')) else ""
                country = str(row.get('Country', '')) if not pd.isna(row.get('Country')) else ""
                
                lat, lon = row.get('Latitude', 0), row.get('Longitude', 0)
                
                # فیلتر آفریقا
                if not geo_filter.is_in_africa(lat, lon):
                    continue
                    
                # استخراج کشور
                if not country or country == "Unknown":
                    search_text = f"{title} {description} {location} {tags}"
                    country = geo_filter.extract_country_advanced(search_text)
                
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
                    'full_text': f"{title} {description} {location} {tags}"
                })
        
        return merged_data

    def search_animal(self, animal_name: str) -> List[Dict]:
        """Search for specific animal in data"""
        animal_name = animal_name.lower()
        results = []
        
        for item in self.images_data:
            # Search in all text fields
            search_text = f"{item['title']} {item['blip_description']} {item['description']} {item['keywords']}".lower()
            
            if animal_name in search_text:
                # Calculate similarity score
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
        
        # Sort by score
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _extract_context(self, text: str, animal: str) -> str:
        """Extract sentence where animal is mentioned"""
        sentences = text.split('.')
        for sentence in sentences:
            if animal in sentence:
                return sentence.strip() + '.'
        return ""
    
    def generate_map(self, results: List[Dict], output_path: str = "animal_map.html"):
        """Generate interactive map"""
        # Filter results with valid coordinates
        valid_results = [r for r in results if r['latitude'] and r['longitude'] and 
                        not np.isnan(r['latitude']) and not np.isnan(r['longitude'])]
        
        if not valid_results:
            print("⚠️ No results with valid geographic coordinates found")
            return None
            
        # Create base map
        avg_lat = np.mean([r['latitude'] for r in valid_results])
        avg_lon = np.mean([r['longitude'] for r in valid_results])
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)
        
        # Add markers for each result
        for result in valid_results:
            popup_text = f"""
            <b>{result['title']}</b><br>
            <i>{result['blip_description']}</i><br>
            کشور: {result['country']}<br>
            امتیاز: {result['score']}<br>
            """
            
            folium.Marker(
                [result['latitude'], result['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=result['title'],
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # Save map
        m.save(output_path)
        print(f"🗺️ Map created: {output_path}")
        return output_path
    
    def generate_report(self, animal: str, results: List[Dict]) -> str:
        """Generate analysis report"""
        if not results:
            return f"هیچ اطلاعاتی درباره '{animal}' در داده‌ها یافت نشد."
        
        # Analyze data - filter invalid values
        countries = [r['country'] for r in results 
                    if r['country'] != 'Unknown' 
                    and not pd.isna(r['country']) 
                    and isinstance(r['country'], str)]
        
        country_counts = {}
        for country in countries:
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # Generate report
        report = f"""
# گزارش تحلیل: {animal}

## 📊 خلاصه آماری
- تعداد مشاهدات: {len(results)}
- کشورهای مشاهده شده: {len(set(countries))}
- بیشترین مشاهده در: {max(country_counts, key=country_counts.get) if country_counts else 'نامشخص'}

## 🗺️ پراکنش جغرافیایی
"""
        
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {country}: {count} مشاهده\n"
        
        report += f"""
## 📝 نمونه‌های کلیدی
"""
        
        for i, result in enumerate(results[:5]):
            report += f"""
### نمونه {i+1}
- **موقعیت**: {result['country']} (عرض: {result['latitude']}, طول: {result['longitude']})
- **توضیح خودکار**: {result['blip_description']}
- **تطابق**: {result['search_match'] or 'توضیح کامل در تصویر'}
"""
        
        # Filter valid countries for recommendations section
        valid_countries = [str(country) for country in country_counts.keys() 
                          if country and not pd.isna(country)]
        
        report += f"""
## 💡 توصیه‌های حفاظتی
بر اساس مشاهدات، این گونه بیشتر در مناطق {', '.join(valid_countries[:3]) if valid_countries else 'مختلف'} مشاهده شده است.
پیشنهاد می‌شود برنامه‌های حفاظتی در این مناطق متمرکز شود.
"""
        
        return report
    
    def process_query(self, animal_query: str):
        """Process user query completely"""
        print(f"🔍 Searching: {animal_query}")
        
        # Search in data
        results = self.search_animal(animal_query)
        
        if not results:
            return {
                'status': 'not_found',
                'message': f'No information found about "{animal_query}".'
            }
        
        # Generate map
        map_path = self.generate_map(results, f"{animal_query.replace(' ', '_')}_map.html")
        
        # Generate report
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
    # File paths
    EXCEL_PATH = "African.xls"
    NPZ_PATH = "results/image_embeddings.npz"
    
    if not os.path.exists(EXCEL_PATH) or not os.path.exists(NPZ_PATH):
        print("❌ Data files not found!")
        return
    
    # Create RAG engine
    rag_engine = RAGEngine(EXCEL_PATH, NPZ_PATH)
    
    # Test with sample query
    query = "lion"  # First test with "lion"
    result = rag_engine.process_query(query)
    
    # Display results
    print("\n" + "="*50)
    print(f"Results for: {query}")
    print("="*50)
    
    if result['status'] == 'success':
        print(f"✅ Number of results: {result['results_count']}")
        if result['map_path']:
            print(f"🗺️ Map created: {result['map_path']}")
        print("\n📄 Report:")
        print(result['report'])
        
        print("\n🔍 Sample results:")
        for i, sample in enumerate(result['sample_results']):
            print(f"{i+1}. {sample['title']} - {sample['country']}")
            print(f"   Description: {sample['blip_description']}")
            print()
    else:
        print(result['message'])

if __name__ == "__main__":
    main()