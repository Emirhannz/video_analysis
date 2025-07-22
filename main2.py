import cv2
from video_analyzer import VideoAnalyzer
import os
import numpy as np
from typing import List, Dict
from rapidfuzz import fuzz

class VideoFrameAnalyzer(VideoAnalyzer):
    def __init__(self, video_path: str):
        super().__init__(None)  # frames_dir artık kullanılmayacak
        self.video_path = video_path
        self.processed_texts = []
        
    def get_video_frames(self) -> List[Dict]:
        """Videodan frame'leri doğrudan oku"""
        frames = []
        print(f"\nVideo dosyası açılıyor: {self.video_path}")
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Video dosyası açılamadı!")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            print(f"Video Bilgileri:")
            print(f"- Toplam frame: {total_frames}")
            print(f"- FPS: {fps}")
            print(f"- Süre: {duration:.2f} saniye")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Her saniyede bir frame al (fps'e göre atlama yap)
                if frame_count % int(fps) == 0:
                    timestamp = frame_count / fps
                    frames.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'frame_no': frame_count
                    })
                    print(f"\rFrame okunuyor: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")
                
                frame_count += 1
            
            cap.release()
            print(f"\nToplam {len(frames)} frame okundu.")
            return frames
            
        except Exception as e:
            print(f"Video okuma hatası: {e}")
            return []
    
    def process_video(self):
        """Video frame'lerini işle"""
        frames = self.get_video_frames()
        if not frames:
            print("İşlenecek frame bulunamadı!")
            return []
            
        current_texts = []
        total_frames = len(frames)
        
        print(f"\nToplam {total_frames} frame işlenecek...")
        print("Bu işlem biraz zaman alabilir, lütfen bekleyin...")
        
        for i, frame_data in enumerate(frames, 1):
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            frame_no = frame_data['frame_no']
            
            print(f"\rFrame işleniyor: {i}/{total_frames} ({(i/total_frames)*100:.1f}%) - {timestamp:.2f}s", end="")
            
            # Frame'deki metinleri al
            frame_texts = self.text_analyzer.process_frame_array(frame)
            
            # Frame'de metin bulunduysa işle
            if frame_texts:
                print(f"\nFrame {frame_no} ({timestamp:.2f}s): {len(frame_texts)} metin bulundu:")
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
                        ratio = fuzz.ratio(text.lower(), existing['text'].lower())
                        if ratio > best_score:
                            best_score = ratio
                            best_match = idx
                
                if not is_similar:
                    # Yeni benzersiz metin ekle
                    text_entry = {
                        'text': text,
                        'timestamp': timestamp,
                        'frame_no': frame_no,
                        'coords': coords,
                        'occurrences': 1,
                        'variations': [text]
                    }
                    current_texts.append(text_entry)
                    self.processed_texts.append(text_entry)
                elif best_match is not None:
                    # Mevcut metni güncelle
                    current_texts[best_match]['occurrences'] += 1
                    if text not in current_texts[best_match]['variations']:
                        current_texts[best_match]['variations'].append(text)
        
        print(f"\n\nToplam {len(self.processed_texts)} benzersiz metin bulundu.")
        return current_texts

def main():
    # Video dosyasının yolu (örnek)
    video_path = "ornekvideo4.mp4"  # videoyu buraya koyun
    
    if not os.path.exists(video_path):
        print(f"HATA: Video dosyası bulunamadı: {video_path}")
        return
    
    # Video analiz nesnesini oluştur
    analyzer = VideoFrameAnalyzer(video_path)
    
    print("Video işleniyor...")
    analyzer.process_video()
    
    print("\nRapor oluşturuluyor...")
    report = analyzer.generate_report()
    
    print("\n" + "="*50 + "\n")
    print(report)
    
    # Raporu dosyaya kaydet
    with open('video_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\nRapor 'video_report.txt' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
