"""File Reading Tool"""

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_file(self):
        """Read file contents"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
