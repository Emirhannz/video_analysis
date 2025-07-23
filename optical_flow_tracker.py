import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class OpticalFlowTracker:
    def __init__(self):
        # Optical flow parametreleri
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        # Aktif metin takibi için buffer
        self.active_text_buffers = {}
        self.last_frame_gray = None
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame'i optical flow için hazırla"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    def _is_sentence_complete(self, text: str) -> bool:
        """Cümlenin bitip bitmediğini kontrol et"""
        # Noktalama işaretleri ve özel durumlar
        end_markers = ['.', '!', '?', '...']
        for marker in end_markers:
            if text.strip().endswith(marker):
                return True
        return False
    
    def _merge_text_boxes(self, box1: List[float], box2: List[float], flow: np.ndarray) -> Optional[List[float]]:
        """İki metin kutusunu optical flow'a göre birleştir"""
        try:
            # Liste formatındaki koordinatları numpy array'e çevir
            box1_arr = np.array(box1).reshape(-1, 2)
            box2_arr = np.array(box2).reshape(-1, 2)
            
            # Kutuların merkezlerini hesapla
            center1 = np.mean(box1_arr, axis=0)
            center2 = np.mean(box2_arr, axis=0)
            
            # Flow vektörünü al
            h, w = flow.shape[:2]
            y, x = int(center1[1]), int(center1[0])
            if 0 <= y < h and 0 <= x < w:
                displacement = flow[y, x]
                predicted_center = center1 + displacement
                
                # Tahmin edilen merkez ile gerçek merkez arasındaki mesafe
                distance = np.linalg.norm(predicted_center - center2)
                
                # Eğer mesafe makul ise kutuları birleştir
                if distance < 50:  # Bu eşik değeri ayarlanabilir
                    merged_box = np.vstack((box1, box2))
                    return merged_box.tolist()
            
            return None
        except Exception as e:
            print(f"Kutu birleştirme hatası: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray, texts: List[Dict]) -> List[Dict]:
        """Frame'deki metinleri optical flow ile takip et ve akan yazıları birleştir"""
        # Frame'i hazırla
        current_frame_gray = self._preprocess_frame(frame)
        
        if self.last_frame_gray is None:
            self.last_frame_gray = current_frame_gray
            return texts
            
        # Optical flow hesapla
        flow = cv2.calcOpticalFlowFarneback(
            self.last_frame_gray,
            current_frame_gray,
            None,
            **self.flow_params
        )
        
        # Yeni ve aktif metinleri işle
        new_texts = []
        for text_info in texts:
            text = text_info['text']
            coords = text_info['coords']
            
            # Koordinatları doğru formata çevir
            if isinstance(coords, list):
                # Eğer coords bir liste ise ve içinde listeler varsa
                if coords and isinstance(coords[0], list):
                    coords = [coord for sublist in coords for coord in sublist]  # Düzleştir
            
            # Her aktif buffer için kontrol et
            merged = False
            buffers_to_remove = []
            
            for buffer_id, buffer in self.active_text_buffers.items():
                merged_box = self._merge_text_boxes(buffer['coords'], coords, flow)
                
                if merged_box is not None:
                    # Metinleri birleştir
                    merged_text = f"{buffer['text']} {text}"
                    buffer['text'] = merged_text
                    buffer['coords'] = merged_box
                    buffer['last_update'] = 0  # Reset counter
                    merged = True
                    
                    # Eğer cümle tamamlandıysa
                    if self._is_sentence_complete(merged_text):
                        new_texts.append({
                            'text': merged_text,
                            'coords': merged_box,
                            'is_flowing': True
                        })
                        buffers_to_remove.append(buffer_id)
                    break
            
            # Buffer'ları temizle
            for buffer_id in buffers_to_remove:
                del self.active_text_buffers[buffer_id]
            
            # Eğer birleştirilemedi ise yeni buffer oluştur
            if not merged:
                buffer_id = len(self.active_text_buffers)
                self.active_text_buffers[buffer_id] = {
                    'text': text,
                    'coords': coords,
                    'last_update': 0
                }
        
        # Aktif buffer'ları güncelle ve timeout olanları temizle
        active_buffers = {}
        for buffer_id, buffer in self.active_text_buffers.items():
            buffer['last_update'] += 1
            if buffer['last_update'] < 10:  # 10 frame timeout
                active_buffers[buffer_id] = buffer
            elif not self._is_sentence_complete(buffer['text']):
                # Timeout olan ama bitmemiş cümleleri ekle
                new_texts.append({
                    'text': buffer['text'],
                    'coords': buffer['coords'],
                    'is_flowing': True
                })
        
        self.active_text_buffers = active_buffers
        self.last_frame_gray = current_frame_gray
        
        return new_texts
