from easyocr import Reader
import os
from rapidfuzz import fuzz
import re
import numpy as np
from sentence_buffer import SentenceBuffer
import spacy

class TextAnalyzer:
    def __init__(self):
        # EasyOCR ayarları - varsayılan
        self.ocr = Reader(['tr'], gpu=True)
        self.sentence_buffer = SentenceBuffer()

        # SpaCy Türkçe Transformer modelini yükle
        try:
            self.nlp = spacy.load("tr_core_news_trf")
        except Exception as e:
            print(f"SpaCy modeli yüklenirken hata: {e}")
            self.nlp = None

        # Türkçe dil özellikleri
        self.verb_suffixes = ['yor', 'dı', 'di', 'du', 'dü', 'acak', 'ecek', 'mış', 'miş', 'muş', 'müş', 'ti', 'di']
        self.common_nouns = ['türkiye', 'ankara', 'istanbul', 'cumhurbaşkanı', 'bakan', 'meclis', 'hükümet']
        self.stop_words = ['ve', 'veya', 'ile', 'için', 'gibi', 'kadar', 'daha', 'en', 'çok', 'bir']

        self.tr_char_map = {
            # Temel Türkçe karakter düzeltmeleri
            'I': 'ı',
            'İ': 'i',
            'i': 'i',
            'ı': 'ı',
            'Ğ': 'ğ',
            'Ü': 'ü',
            'Ş': 'ş',
            'Ö': 'ö',
            'Ç': 'ç',
            # OCR hatalarını düzelt
            '|': 'i',
            'l': 'ı',
            'O': 'o',
            'B': 'b',
            'P': 'p',
            'T': 't',
            'R': 'r',
            # Sık karşılaşılan OCR hataları
            '1': 'ı',
            '0': 'o',
            '€': 'e',
            '¢': 'c',
            'ã': 'a',
            'õ': 'o',
            'ñ': 'n',
            'í': 'i',
            'ó': 'o',
            'é': 'e',
            'á': 'a',
            'ú': 'u'
        }

    def is_valid_sentence(self, text: str) -> bool:
        """Metnin geçerli bir cümle olup olmadığını kontrol et"""
        if not text or len(text) < 3:
            return False

        # SpaCy ile analiz
        if self.nlp:
            doc = self.nlp(text)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            has_noun = any(token.pos_ == "NOUN" for token in doc)
            return has_verb or has_noun  # Hem fiil hem isim yerine biri yeterli

        # Basit kontrol (SpaCy yoksa)
        words = text.lower().split()
        has_verb = any(word.endswith(tuple(self.verb_suffixes)) for word in words)
        has_noun = any(word in self.common_nouns for word in words)
        return has_verb or has_noun  # Hem fiil hem isim yerine biri yeterli

    def extract_entities(self, text: str) -> dict:
        """Metindeki varlıkları (kişi, yer, kurum vs.) çıkar"""
        entities = {
            'PERSON': [],
            'LOC': [],
            'ORG': []
        }

        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)

        return entities

    def normalize_text(self, text):
        """Metni normalize et ve temizle"""
        if not text:
            return ""

        # Metindeki özel boşluk karakterlerini normal boşluğa çevir
        text = text.replace('\u200b', ' ')  # Zero-width space
        text = text.replace('\xa0', ' ')    # Non-breaking space
        text = text.replace('\t', ' ')      # Tab
        text = text.replace('\n', ' ')      # Newline

        # Özel karakterleri temizle
        text = text.replace('"', '')  # Çift tırnak
        text = text.replace("'", '')  # Tek tırnak

        # Önce tüm metni küçük harfe çevir
        text = text.lower()

        # Türkçe karakterleri düzelt (büyük/küçük harf duyarlı)
        normalized = text
        for wrong, correct in self.tr_char_map.items():
            normalized = normalized.replace(wrong.lower(), correct)
            normalized = normalized.replace(wrong.upper(), correct)

        # Kelimeleri ayır ve tekrar birleştir (boşlukları düzenle)
        words = []
        current_word = ""
        for char in normalized:
            if char.isalnum() or char in "çğıöşü":
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if char.isspace():
                    if not words or not words[-1].isspace():
                        words.append(' ')
                else:
                    words.append(char)
        if current_word:
            words.append(current_word)

        # Kelimeleri birleştir
        normalized = ''.join(words)

        # Çoklu boşlukları tekil boşluğa çevir
        normalized = re.sub(r'\s+', ' ', normalized)

        # Noktalama işaretlerini koru ama fazla olanları temizle
        normalized = re.sub(r'[.]{2,}', '.', normalized)  # ... -> .
        normalized = re.sub(r'[!]{2,}', '!', normalized)  # !!! -> !
        normalized = re.sub(r'[?]{2,}', '?', normalized)  # ??? -> ?

        # Noktalama işaretlerinden sonra boşluk ekle
        normalized = re.sub(r'([.,!?])([^\s])', r'\1 \2', normalized)

        # Baştaki ve sondaki boşlukları temizle
        return normalized.strip()
    
    def find_common_variations(self, text):
        """Metindeki yaygın varyasyonları bul ve düzelt"""
        normalized = text.upper()
        for correct, variations in self.common_variations.items():
            for var in variations:
                if var in normalized:
                    normalized = normalized.replace(var, correct)
        return normalized

    def compare_texts(self, text1, text2, coords1=None, coords2=None, threshold=65):
        """Gelişmiş metin karşılaştırma"""
        # İlk olarak temel normalizasyon
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Metin çok kısaysa eşleşme skorunu yükselt
        if len(norm1) <= 3 or len(norm2) <= 3:
            threshold = 90
        
        # Yaygın varyasyonları kontrol et
        var1 = self.find_common_variations(text1)
        var2 = self.find_common_variations(text2)
        
        # Farklı karşılaştırma yöntemlerini dene
        ratio = fuzz.ratio(norm1, norm2)
        partial_ratio = fuzz.partial_ratio(norm1, norm2)
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2)
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2)
        
        # En yüksek eşleşme skorunu al
        max_ratio = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # Yaygın varyasyonlar için ek kontrol
        variation_match = (var1 == var2)
        
        # Koordinat bazlı karşılaştırma
        coord_match = False
        if coords1 is not None and coords2 is not None:
            try:
                # Koordinatları numpy array'e çevir
                c1 = np.array(coords1)
                c2 = np.array(coords2)
                
                # Koordinatlar liste içinde liste formatındaysa düzelt
                if len(c1.shape) > 2:
                    c1 = c1.reshape(-1, 2)
                if len(c2.shape) > 2:
                    c2 = c2.reshape(-1, 2)
                
                # Merkezleri hesapla
                center1 = np.mean(c1, axis=0)
                center2 = np.mean(c2, axis=0)
                
                # İki merkez arasındaki mesafe
                distance = np.linalg.norm(center1 - center2)
                
                # Eğer mesafe 50 pikselden azsa, aynı bölgede sayılır
                coord_match = distance < 50
                
                # Aynı bölgedeyse ve benzerlik skoru düşükse, skoru artır
                if coord_match and max_ratio > 40:
                    max_ratio += 20
            except:
                pass
        
        # Sonuç hesaplama:
        # 1. Tam eşleşme varsa
        if ratio == 100:
            return True
        # 2. Yaygın varyasyon eşleşmesi ve yeterli benzerlik varsa
        if variation_match and max_ratio > threshold - 10:
            return True
        # 3. Aynı bölgede ve yeterli benzerlik varsa
        if coord_match and max_ratio > threshold - 5:
            return True
        # 4. Çok yüksek benzerlik varsa
        return max_ratio > threshold

    def fix_turkish_chars(self, text):
        """Türkçe karakterleri düzelt"""
        for wrong, correct in self.tr_char_map.items():
            text = text.replace(wrong, correct)
        return text
    
    def process_frame_array(self, frame_array):
        """Numpy array olarak frame'i işle"""
        texts = []
        try:
            result = self.ocr.readtext(frame_array)
            for (bbox, text, confidence) in result:
                # Önce Türkçe karakterleri düzelt
                fixed_text = self.fix_turkish_chars(text)
                # Sonra metni tamamen normalize et
                normalized_text = self.normalize_text(fixed_text)

                if normalized_text.strip() and confidence > 0.5:
                    texts.append({
                        'text': normalized_text,
                        'coords': bbox,
                        'confidence': confidence
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
            result = self.ocr.readtext(frame_path)
            for (bbox, text, confidence) in result:
                # Önce Türkçe karakterleri düzelt
                fixed_text = self.fix_turkish_chars(text)
                # Sonra metni tamamen normalize et
                normalized_text = self.normalize_text(fixed_text)

                if normalized_text.strip() and confidence > 0.5:
                    texts.append({
                        'text': normalized_text,
                        'coords': bbox,
                        'confidence': confidence
                    })
        except Exception as e:
            print(f"HATA: {str(e)}")

        return texts
    
    def compare_texts(self, text1, text2, coords1=None, coords2=None, threshold=65):
        """Gelişmiş metin karşılaştırma"""
        # İlk olarak temel normalizasyon
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Metin çok kısaysa eşleşme skorunu yükselt
        if len(norm1) <= 3 or len(norm2) <= 3:
            threshold = 90
        
        # Yaygın varyasyonları kontrol et
        var1 = self.find_common_variations(text1)
        var2 = self.find_common_variations(text2)
        
        # Farklı karşılaştırma yöntemlerini dene
        ratio = fuzz.ratio(norm1, norm2)
        partial_ratio = fuzz.partial_ratio(norm1, norm2)
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2)
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2)
        
        # En yüksek eşleşme skorunu al
        max_ratio = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # Yaygın varyasyonlar için ek kontrol
        variation_match = (var1 == var2)
        
        # Koordinat bazlı karşılaştırma
        coord_match = False
        if coords1 is not None and coords2 is not None:
            try:
                import numpy as np
                # Koordinatları numpy array'e çevir
                c1 = np.array(coords1)
                c2 = np.array(coords2)
                
                # Koordinatlar liste içinde liste formatındaysa düzelt
                if len(c1.shape) > 2:
                    c1 = c1.reshape(-1, 2)
                if len(c2.shape) > 2:
                    c2 = c2.reshape(-1, 2)
                
                # Merkezleri hesapla
                center1 = np.mean(c1, axis=0)
                center2 = np.mean(c2, axis=0)
                
                # İki merkez arasındaki mesafe
                distance = np.linalg.norm(center1 - center2)
                
                # Eğer mesafe 100 pikselden azsa, yakın bölgede sayılır
                coord_match = distance < 100
                
                # Aynı bölgedeyse ve benzerlik skoru düşükse, skoru artır
                if coord_match and max_ratio > 40:
                    max_ratio += 20
            except Exception as e:
                print(f"Koordinat karşılaştırma hatası: {str(e)}")
                pass
        
        # Sonuç hesaplama:
        # 1. Tam eşleşme varsa
        if ratio == 100:
            return True
        # 2. Yaygın varyasyon eşleşmesi ve yeterli benzerlik varsa
        if variation_match and max_ratio > threshold - 10:
            return True
        # 3. Aynı bölgede ve yeterli benzerlik varsa
        if coord_match and max_ratio > threshold - 5:
            return True
        # 4. Çok yüksek benzerlik varsa
        return max_ratio > threshold