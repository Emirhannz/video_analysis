from paddleocr import PaddleOCR
import os
from rapidfuzz import fuzz
import re

class TextAnalyzer:
    def __init__(self):
        self.ocr = PaddleOCR()
        self.tr_char_map = {
            '|': 'i',
            'l': 'ı',
            'I': 'ı',
            'i': 'i',
            'O': 'o',
            'B': 'b',
            'P': 'p',
            'T': 't',
            'R': 'r'
        }
        # Yaygın yazım varyasyonları
        self.common_variations = {
            'HABER': ['HAPER', 'HAVER', 'HABR', 'HBER'],
            'TRT': ['TRD', 'TRP', 'TRT'],
            'KANAL': ['KANL', 'KNAL', 'KANAAL'],
            'STAR': ['STR', 'ISTAR', 'ESTAR'],
            'SHOW': ['SOW', 'ŞOV', 'SHOW'],
            'TV': ['TW', 'Tv', 'TV']
        }
    
    def normalize_text(self, text):
        """Metni normalize et ve temizle"""
        # Küçük harfe çevir
        text = text.lower()
        # Türkçe karakterleri düzelt
        for wrong, correct in self.tr_char_map.items():
            text = text.lower().replace(wrong.lower(), correct.lower())
        # Özel karakterleri ve fazla boşlukları temizle
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def find_common_variations(self, text):
        """Metindeki yaygın varyasyonları bul ve düzelt"""
        normalized = text.upper()
        for correct, variations in self.common_variations.items():
            for var in variations:
                if var in normalized:
                    normalized = normalized.replace(var, correct)
        return normalized

    def compare_texts(self, text1, text2, threshold=75):
        """Gelişmiş metin karşılaştırma"""
        # İlk olarak temel normalizasyon
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Yaygın varyasyonları kontrol et
        var1 = self.find_common_variations(text1)
        var2 = self.find_common_variations(text2)
        
        # Farklı karşılaştırma yöntemlerini dene
        ratio = fuzz.ratio(norm1, norm2)
        partial_ratio = fuzz.partial_ratio(norm1, norm2)
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2)
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2)
        
        # Yaygın varyasyonlar için ek kontrol
        variation_match = (var1 == var2)
        
        # En yüksek eşleşme skorunu al
        max_ratio = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # Eğer yaygın varyasyon eşleşmesi varsa veya benzerlik skoru yüksekse
        return variation_match or max_ratio > threshold

    def fix_turkish_chars(self, text):
        """Türkçe karakterleri düzelt"""
        for wrong, correct in self.tr_char_map.items():
            text = text.replace(wrong, correct)
        return text
    
    def process_frame_array(self, frame_array):
        """Numpy array olarak frame'i işle"""
        texts = []
        try:
            result = self.ocr.ocr(frame_array)
            if not result:
                return []
                
            for page_result in result:
                if isinstance(page_result, dict):
                    found_texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    boxes = page_result.get('rec_boxes', [])
                    
                    for i in range(len(found_texts)):
                        if i < len(scores) and i < len(boxes):
                            text = str(found_texts[i])
                            score = float(scores[i])
                            box = boxes[i]
                            
                            fixed_text = self.fix_turkish_chars(text)
                            if fixed_text.strip() and score > 0.5:
                                texts.append({
                                    'text': fixed_text,
                                    'coords': box.tolist() if hasattr(box, 'tolist') else box,
                                    'confidence': score
                                })
        except Exception as e:
            print(f"HATA: Frame işlenirken hata oluştu: {str(e)}")
        
        return texts
        
    def process_frame(self, frame_path):
        if not os.path.exists(frame_path):
            print(f"HATA: Frame bulunamadı: {frame_path}")
            return []
            
        texts = []
        try:
            result = self.ocr.ocr(frame_path)
            if not result:
                return []
                
            for page_result in result:
                if isinstance(page_result, dict):
                    found_texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    boxes = page_result.get('rec_boxes', [])
                    
                    for i in range(len(found_texts)):
                        if i < len(scores) and i < len(boxes):
                            text = str(found_texts[i])
                            score = float(scores[i])
                            box = boxes[i]
                            
                            fixed_text = self.fix_turkish_chars(text)
                            if fixed_text.strip() and score > 0.5:
                                texts.append({
                                    'text': fixed_text,
                                    'coords': box.tolist() if hasattr(box, 'tolist') else box,
                                    'confidence': score
                                })
        except Exception as e:
            print(f"HATA: {str(e)}")
        
        return texts
    
    def compare_texts(self, text1, text2, threshold=90):
        return fuzz.ratio(text1, text2) > threshold
