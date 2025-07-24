import cv2
from video_analyzer import VideoAnalyzer
import os
import numpy as np
from typing import List, Dict
from rapidfuzz import fuzz
from optical_flow_tracker import OpticalFlowTracker
import torch  # GPU kontrolü için PyTorch eklendi
import spacy
import easyocr

class VideoFrameAnalyzer(VideoAnalyzer):
    def __init__(self, video_path: str):
        super().__init__(None)  # frames_dir artık kullanılmayacak
        self.video_path = video_path
        self.processed_texts = []
        self.optical_flow_tracker = OpticalFlowTracker()
        
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

            # Optical flow ile akan yazıları tespit et ve birleştir
            flowing_texts = self.optical_flow_tracker.process_frame(frame, frame_texts)

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
                                        'timestamp': timestamp,
                                        'frame_no': frame_no,
                                        'coords': ft['coords'],
                                        'entities': self.text_analyzer.extract_entities(sentence)
                                    })

                if completed_sentences:
                    print(f"\nFrame {frame_no} ({timestamp:.2f}s): {len(completed_sentences)} cümle tamamlandı:")
                    for cs in completed_sentences[:2]:  # İlk 2 cümleyi göster
                        print(f"  - {cs['text']}")

                current_texts.extend(completed_sentences)

        # Add processed sentences to self.processed_texts
        self.processed_texts.extend(current_texts)

        print(f"\n\nToplam {len(current_texts)} benzersiz cümle bulundu.")
        return current_texts

def main():
    # Video dosyasının yolu (örnek)
    video_path = "ornekvideo7.mp4"  # videoyu buraya koyun

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

    # Raporu 'rapor' klasörüne kaydet
    rapor_klasoru = "rapor"
    os.makedirs(rapor_klasoru, exist_ok=True)
    video_baslik = os.path.splitext(os.path.basename(video_path))[0]
    rapor_dosyasi = os.path.join(rapor_klasoru, f"{video_baslik}.txt")

    with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nRapor '{rapor_dosyasi}' dosyasına kaydedildi.")

# GPU'nun aktif olup olmadığını kontrol et ve yazdır
def test_gpu_usage():
    print("\nGPU Kullanılabilirlik Testi:")
    if torch.cuda.is_available():
        print("PyTorch GPU kullanabiliyor.")
        try:
            # Basit bir tensör işlemi yaparak GPU'yu test et
            x = torch.rand(3, 3).cuda()
            print("PyTorch GPU testi başarılı: Tensor işlemi tamamlandı.")
        except Exception as e:
            print(f"PyTorch GPU testi başarısız: {e}")
    else:
        print("PyTorch GPU kullanamıyor.")

    # OpenCV'nin CUDA desteğini kontrol et
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("OpenCV CUDA desteği aktif.")
        else:
            print("OpenCV CUDA desteği pasif.")
    except Exception as e:
        print(f"OpenCV CUDA kontrolü başarısız: {e}")

# SpaCy modelini manuel olarak yükle
try:
    print("SpaCy modeli yükleniyor: tr_core_news_trf")
    nlp = spacy.load("tr_core_news_trf")
    print("SpaCy modeli başarıyla yüklendi.")
except Exception as e:
    print(f"SpaCy modeli yüklenirken hata: {e}")

# EasyOCR'nin GPU kullanıp kullanmadığını test et
try:
    print("\nEasyOCR GPU Testi:")
    reader = easyocr.Reader(['tr'], gpu=torch.cuda.is_available())
    print("EasyOCR başarıyla yüklendi ve GPU kullanımı test edildi.")
except Exception as e:
    print(f"EasyOCR GPU testi başarısız: {e}")

if __name__ == "__main__":
    test_gpu_usage()
    main()