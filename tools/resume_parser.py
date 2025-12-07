"""Resume Parsing Tool"""

class ResumeParser:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def extract_skills(self):
        """Extract skills from resume"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple skill extraction
            skills = []
            common_skills = ['Python', 'Java', 'JavaScript', 'SQL', 'Machine Learning', 
                           'Data Analysis', 'Project Management', 'Leadership']
            
            for skill in common_skills:
                if skill.lower() in content.lower():
                    skills.append(skill)
            
            return skills if skills else ["Skills not clearly specified"]
        except:
            return ["Error reading resume"]
