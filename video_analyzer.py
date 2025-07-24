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
                completed_sentences = []
                for ft in frame_texts:
                    if ft.get('text'):
                        sentences = self.text_analyzer.sentence_buffer.add_text(ft['text'])
                        if sentences:
                            for sentence in sentences:
                                if self.text_analyzer.is_valid_sentence(sentence):
                                    completed_sentences.append({
                                        'text': sentence,
                                        'coords': ft['coords'],
                                        'entities': self.text_analyzer.extract_entities(sentence)
                                    })

                if completed_sentences:
                    print(f"\nFrame {i}: {len(completed_sentences)} cümle tamamlandı:")
                    for cs in completed_sentences[:2]:  # İlk 2 cümleyi göster
                        print(f"  - {cs['text']}")

                current_texts.extend(completed_sentences)

        print(f"\n\nToplam {len(current_texts)} benzersiz cümle bulundu.")
        return current_texts

    def generate_report(self):
        """İşlenmiş metinlerden rapor oluştur"""
        if not self.processed_texts:
            return "Henüz metin işlenmemiş."

        report = "📺 Video Metin Raporu:\n\n"

        # Tüm metinleri birleştir (özet için)
        all_text = ""

        # Metinleri koordinatlarına göre grupla
        top_texts = []    # Üst bölge metinleri
        bottom_texts = [] # Alt bölge metinleri
        other_texts = []  # Diğer bölge metinleri

        for text_info in self.processed_texts:
            try:
                coords = text_info['coords']
                text = text_info['text']
                timestamp = text_info.get('timestamp', 0)  # Zaman damgası

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
                    'occurrences': text_info.get('occurrences', 1),
                    'timestamp': timestamp
                }

                if y_coord < 200:  # Üst bölge
                    top_texts.append(text_info_with_variations)
                elif y_coord > 400:  # Alt bölge
                    bottom_texts.append(text_info_with_variations)
                else:  # Orta bölge
                    other_texts.append(text_info_with_variations)
            except Exception as e:
                print(f"Metin koordinat hatası: {text} - {str(e)}")

        def format_texts_with_timestamp(texts):
            formatted = ""
            for text_info in texts:
                timestamp = text_info.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                formatted += f"  - [{minutes}:{seconds:02d}] {text_info['text']}\n"
            return formatted

        if top_texts:
            report += "🔸 Üst Bölge Metinleri:\n"
            report += format_texts_with_timestamp(top_texts)
            report += "\n"

        if bottom_texts:
            report += "🔸 Alt Bölge Metinleri (Altyazılar):\n"
            report += format_texts_with_timestamp(bottom_texts)
            report += "\n"

        if other_texts:
            report += "🔸 Diğer Metinler:\n"
            report += format_texts_with_timestamp(other_texts)

        return report