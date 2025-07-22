import os
from text_analyzer import TextAnalyzer
from rapidfuzz import fuzz

class VideoAnalyzer:
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.text_analyzer = TextAnalyzer()
        self.processed_texts = []
        
    def get_frame_files(self):
        """Frame dosyalarını sıralı şekilde al"""
        frames = []
        print(f"\nFrame klasörü: {self.frames_dir}")
        try:
            files = os.listdir(self.frames_dir)
            print(f"Toplam dosya sayısı: {len(files)}")
            
            for file in files:
                if file.startswith('frame_') and file.endswith('.jpg'):
                    frames.append(os.path.join(self.frames_dir, file))
            
            print(f"Bulunan frame sayısı: {len(frames)}")
            if frames:
                print(f"İlk frame: {frames[0]}")
                print(f"Son frame: {frames[-1]}")
            
            return sorted(frames)
        except Exception as e:
            print(f"Frame dosyaları listelenirken hata: {e}")
            return []
    
    def process_frames(self):
        """Tüm frame'leri işle"""
        frames = self.get_frame_files()
        if not frames:
            print("İşlenecek frame bulunamadı!")
            return []
            
        current_texts = []
        total_frames = len(frames)
        processed_count = 0  # İşlenen metin sayısı
        
        print(f"\nToplam {total_frames} frame işlenecek...")
        print("Bu işlem biraz zaman alabilir, lütfen bekleyin...")
        
        for i, frame_path in enumerate(frames, 1):
            print(f"\rFrame işleniyor: {i}/{total_frames} ({(i/total_frames)*100:.1f}%)", end="")
            
            # Frame'deki metinleri al
            frame_texts = self.text_analyzer.process_frame(frame_path)
            
            # Frame'de metin bulunduysa işle
            if frame_texts:
                print(f"\nFrame {i}: {len(frame_texts)} metin bulundu:")
                for ft in frame_texts[:3]:  # İlk 3 metni göster
                    print(f"  - {ft['text']}")
            
            for text_info in frame_texts:
                text = text_info['text']
                coords = text_info['coords']
                
                # Gelişmiş benzer metin kontrolü ve gruplama
                is_similar = False
                best_match = None
                best_score = 0
                
                for idx, existing in enumerate(current_texts):
                    if self.text_analyzer.compare_texts(text, existing['text']):
                        is_similar = True
                        # En iyi eşleşmeyi bul
                        ratio = fuzz.ratio(text.lower(), existing['text'].lower())
                        if ratio > best_score:
                            best_score = ratio
                            best_match = idx
                
                if not is_similar:
                    # Yeni benzersiz metin ekle
                    text_entry = {
                        'text': text,
                        'frame': frame_path,
                        'coords': coords,
                        'occurrences': 1,
                        'variations': [text]
                    }
                    current_texts.append(text_entry)
                    self.processed_texts.append(text_entry)
                elif best_match is not None:
                    # Mevcut metni güncelle ve varyasyonları kaydet
                    current_texts[best_match]['occurrences'] += 1
                    if text not in current_texts[best_match]['variations']:
                        current_texts[best_match]['variations'].append(text)
        
        print(f"\n\nToplam {len(self.processed_texts)} benzersiz metin bulundu.")
        return current_texts

    def generate_report(self):
        """İşlenmiş metinlerden rapor oluştur"""
        if not self.processed_texts:
            return "Henüz metin işlenmemiş."
        
        report = "📺 Video Metin Raporu:\n\n"
        
        # Metinleri koordinatlarına göre grupla
        top_texts = []    # Üst bölge metinleri
        bottom_texts = [] # Alt bölge metinleri
        other_texts = []  # Diğer bölge metinleri
        
        for text_info in self.processed_texts:
            try:
                coords = text_info['coords']
                text = text_info['text']
                
                # Koordinat formatını kontrol et
                if isinstance(coords, list) and len(coords) > 0:
                    if isinstance(coords[0], list):
                        y_coord = coords[0][1]  # [[x1,y1], [x2,y2], ...] formatı
                    else:
                        y_coord = coords[1]     # [x1,y1,x2,y2,...] formatı
                else:
                    y_coord = 300  # Varsayılan olarak orta bölgeye koy
                
                text_info_with_variations = {
                    'text': text,
                    'variations': text_info.get('variations', [text]),
                    'occurrences': text_info.get('occurrences', 1)
                }
                
                if y_coord < 200:  # Üst bölge
                    top_texts.append(text_info_with_variations)
                elif y_coord > 400:  # Alt bölge
                    bottom_texts.append(text_info_with_variations)
                else:  # Orta bölge
                    other_texts.append(text_info_with_variations)
            except Exception as e:
                print(f"Metin koordinat hatası: {text} - {str(e)}")
                other_texts.append({'text': text, 'variations': [text], 'occurrences': 1})  # Hata durumunda orta bölgeye koy
        
        if top_texts:
            report += "🔸 Üst Bölge Metinleri:\n"
            for text_info in top_texts:
                report += f"  - {text_info['text']}"
                if len(text_info['variations']) > 1:
                    report += f" ({text_info['occurrences']} kez, varyasyonlar: {', '.join(text_info['variations'])})"
                report += "\n"
            report += "\n"
            
        if bottom_texts:
            report += "🔸 Alt Bölge Metinleri (Altyazılar):\n"
            for text_info in bottom_texts:
                report += f"  - {text_info['text']}"
                if len(text_info['variations']) > 1:
                    report += f" ({text_info['occurrences']} kez, varyasyonlar: {', '.join(text_info['variations'])})"
                report += "\n"
            report += "\n"
            
        if other_texts:
            report += "🔸 Diğer Metinler:\n"
            for text_info in other_texts:
                report += f"  - {text_info['text']}"
                if len(text_info['variations']) > 1:
                    report += f" ({text_info['occurrences']} kez, varyasyonlar: {', '.join(text_info['variations'])})"
                report += "\n"
        
        return report
