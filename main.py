from video_analyzer import VideoAnalyzer
import os

def main():
    # Frame klasörünün yolu
    frames_dir = os.path.join(os.path.dirname(__file__), 'frames')
    
    # Video analiz nesnesini oluştur
    analyzer = VideoAnalyzer(frames_dir)
    
    print("Frame'ler işleniyor...")
    analyzer.process_frames()
    
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
