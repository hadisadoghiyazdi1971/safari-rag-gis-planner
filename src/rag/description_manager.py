import json
import os
import pandas as pd
from typing import Dict, List, Optional

class DescriptionManager:
    """مدیریت ذخیره‌سازی و بازیابی توضیحات BLIP و LLM"""
    
    def __init__(self, storage_file: str = "descriptions_database.json"):
        self.storage_file = storage_file
        self.descriptions_db = self._load_database()
    
    def _load_database(self) -> Dict:
        """بارگذاری دیتابیس از فایل"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading database: {e}")
                return {}
        return {}
    
    def _save_database(self):
        """ذخیره دیتابیس در فایل"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.descriptions_db, f, ensure_ascii=False, indent=2)
            print(f"💾 Database saved: {len(self.descriptions_db)} entries")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def get_description(self, filename: str) -> Dict:
        """دریافت توضیحات برای یک فایل"""
        return self.descriptions_db.get(filename, {
            'blip_description': '',
            'llm_description': '',
            'has_llm': False,
            'timestamp': ''
        })
    
    def save_blip_description(self, filename: str, blip_desc: str):
        """ذخیره توضیحات BLIP"""
        if filename not in self.descriptions_db:
            self.descriptions_db[filename] = {
                'blip_description': blip_desc,
                'llm_description': '',
                'has_llm': False,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        else:
            self.descriptions_db[filename]['blip_description'] = blip_desc
        
        self._save_database()
    
    def save_llm_description(self, filename: str, llm_desc: str):
        """ذخیره توضیحات LLM"""
        if filename not in self.descriptions_db:
            self.descriptions_db[filename] = {
                'blip_description': '',
                'llm_description': llm_desc,
                'has_llm': True,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        else:
            self.descriptions_db[filename]['llm_description'] = llm_desc
            self.descriptions_db[filename]['has_llm'] = True
        
        self._save_database()
    
    def get_files_without_llm(self, filenames: List[str]) -> List[str]:
        """دریافت فایل‌هایی که LLM ندارند"""
        return [f for f in filenames if not self.descriptions_db.get(f, {}).get('has_llm', False)]
    
    def get_stats(self) -> Dict:
        """آمار دیتابیس"""
        total = len(self.descriptions_db)
        with_llm = sum(1 for item in self.descriptions_db.values() if item.get('has_llm', False))
        
        return {
            'total_files': total,
            'with_llm': with_llm,
            'without_llm': total - with_llm,
            'llm_coverage': (with_llm / total * 100) if total > 0 else 0
        }
    
    def export_to_csv(self, output_file: str = "descriptions_export.csv"):
        """خروجی به CSV برای تحلیل"""
        data = []
        for filename, desc_data in self.descriptions_db.items():
            data.append({
                'filename': filename,
                'blip_description': desc_data.get('blip_description', ''),
                'llm_description': desc_data.get('llm_description', ''),
                'has_llm': desc_data.get('has_llm', False),
                'timestamp': desc_data.get('timestamp', '')
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"📊 Exported {len(data)} entries to {output_file}")