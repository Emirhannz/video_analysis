class SentenceBuffer:
    def __init__(self):
        self.buffer = ""
        self.sentence_end_markers = ['.', '!', '?']
        self.max_buffer_size = 1000  # Karakter limiti
        
    def add_text(self, text: str) -> list:
        """
        Metni buffer'a ekler ve tamamlanmış cümleleri döndürür
        """
        self.buffer += " " + text.strip()
        self.buffer = self.buffer.strip()
        
        completed_sentences = []
        while True:
            # Tamamlanmış cümle var mı kontrol et
            end_index = -1
            for marker in self.sentence_end_markers:
                idx = self.buffer.find(marker)
                if idx != -1 and (end_index == -1 or idx < end_index):
                    end_index = idx + 1
            
            # Tamamlanmış cümle varsa ayır
            if end_index != -1:
                sentence = self.buffer[:end_index].strip()
                if sentence:
                    completed_sentences.append(sentence)
                self.buffer = self.buffer[end_index:].strip()
            else:
                break
                
        # Buffer çok uzunsa zorla böl
        if len(self.buffer) > self.max_buffer_size:
            last_space = self.buffer.rfind(' ', 0, self.max_buffer_size)
            if last_space != -1:
                sentence = self.buffer[:last_space].strip()
                if sentence:
                    completed_sentences.append(sentence + "...")
                self.buffer = self.buffer[last_space:].strip()
            
        return completed_sentences
        
    def get_pending(self) -> str:
        """Henüz tamamlanmamış metni döndürür"""
        return self.buffer
        
    def clear(self):
        """Buffer'ı temizler"""
        pending = self.buffer
        self.buffer = ""
        return pending if pending else None
