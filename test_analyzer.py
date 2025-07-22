from text_analyzer import TextAnalyzer
import os

def test_single_frame():
    try:
        # Test için TextAnalyzer oluştur
        analyzer = TextAnalyzer()
        
        # Frame klasörünün yolu
        frames_dir = os.path.join(os.path.dirname(__file__), 'frames')
        
        # Frame klasörü kontrolü
        if not os.path.exists(frames_dir):
            print(f"HATA: Frame klasörü bulunamadı: {frames_dir}")
            return
            
        # Test için ilk frame'i al
        test_frame = os.path.join(frames_dir, 'frame_0006.jpg')
        
        # Frame dosyası kontrolü
        if not os.path.exists(test_frame):
            print(f"HATA: Test frame dosyası bulunamadı: {test_frame}")
            return
            
        # Frame boyutu kontrolü
        import cv2
        image = cv2.imread(test_frame)
        if image is None:
            print(f"HATA: Frame dosyası okunamadı: {test_frame}")
            return
        
        print(f"Test frame: {test_frame}")
        print(f"Frame boyutu: {image.shape}")
        print("OCR işlemi başlıyor...")
        
        # Frame'i işle
        results = analyzer.process_frame(test_frame)
        
        print("\nBulunan metinler:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. Metin: {result['text']}")
                print(f"   Koordinatlar: {result['coords']}")
                print(f"   Güven skoru: {result['confidence']}")
        else:
            print("Hiç metin bulunamadı!")
            
    except Exception as e:
        print(f"HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_frame()
