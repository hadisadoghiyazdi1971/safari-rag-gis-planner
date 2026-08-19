import faiss
import numpy as np
import pandas as pd
import pickle
import os

class VectorStore:
    def __init__(self, index_dir="./vector_store"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index = None
        self.metadata = []
    
    def create_from_processed_data(self, npz_path: str, excel_path: str):
        """ایجاد ایندکس از داده‌های پردازش‌شده"""
        # 1. بارگذاری نتایج پردازش تصاویر
        data = np.load(npz_path)
        filenames = data['filenames']
        texts = data['texts'] 
        vectors = data['vectors']
        
        # 2. بارگذاری اکسل
        excel_data = pd.read_excel(excel_path)
        
        # 3. ایجاد ایندکس FAISS
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        
        # 4. ایجاد متادیتا
        for i, filename in enumerate(filenames):
            # پیدا کردن ردیف مربوطه در اکسل
            excel_row = excel_data[excel_data['filename'] == filename]
            
            if not excel_row.empty:
                row = excel_row.iloc[0]
                self.metadata.append({
                    'id': i,
                    'filename': filename,
                    'blip_text': texts[i],
                    'latitude': row.get('latitude', 0),
                    'longitude': row.get('longitude', 0),
                    'title': row.get('title', ''),
                    'description': row.get('description', ''),
                    'keywords': row.get('keywords', '')
                })
        
        print(f"✅ ایندکس ایجاد شد: {len(self.metadata)} تصویر")
    
    def search_similar(self, query_vector: np.ndarray, k: int = 10):
        """جستجوی ساده مشابهت‌ها"""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    'score': float(score),
                    'filename': meta['filename'],
                    'blip_text': meta['blip_text'],
                    'latitude': meta['latitude'],
                    'longitude': meta['longitude'],
                    'title': meta['title']
                })
        
        return results

    def save(self):
        """ذخیره ساده"""
        faiss.write_index(self.index, f"{self.index_dir}/vectors.index")
        with open(f"{self.index_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self):
        """بارگذاری ساده"""
        self.index = faiss.read_index(f"{self.index_dir}/vectors.index")
        with open(f"{self.index_dir}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

def main():
    # استفاده ساده
    vector_store = VectorStore()
    
    # ایجاد ایندکس از داده‌های موجود
    vector_store.create_from_processed_data(
        npz_path="./results/image_embeddings.npz",
        excel_path="./African.xls"  # همان فایل اکسل اصلی
    )
    
    vector_store.save()
    print("✅ ذخیره برداری آماده است!")

if __name__ == "__main__":
    main()