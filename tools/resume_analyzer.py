"""
CognifyX Resume Intelligence Engine
Analyzes resume data from CSV, Excel, and PDF formats
Fully dynamic - adapts to any resume dataset structure
"""

import pandas as pd
import numpy as np
from datetime import datetime
from agents.planner_agent import PlannerAgent
from agents.worker_agent import WorkerAgent
from agents.reviewer_agent import ReviewerAgent
import os
import re
import glob
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

class ResumeAnalyzer:
    """
    Universal Resume Analysis Engine
    Handles CSV/Excel data and PDF resume parsing
    """
    
    def __init__(self, data_source, planner_model="llama3", worker_model="mistral", reviewer_model="qwen2.5"):
        self.data_source = data_source
        self.data = None
        self.data_type = None  # 'csv', 'excel', 'pdf', or 'folder'
        
        # Initialize agents
        self.planner = PlannerAgent(model=planner_model)
        self.worker = WorkerAgent(model=worker_model)
        self.reviewer = ReviewerAgent(model=reviewer_model)
        
        # Auto-detected fields
        self.text_fields = []
        self.categorical_fields = []
        self.numeric_fields = []
        
    def load_and_preprocess(self):
        """Load resume data from various sources"""
        try:
            # Determine data type
            if self.data_source.endswith('.csv'):
                self.data_type = 'csv'
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(self.data_source, encoding=encoding, on_bad_lines='skip')
                        break
                    except:
                        continue
                        
            elif self.data_source.endswith(('.xlsx', '.xls')):
                self.data_type = 'excel'
                self.data = pd.read_excel(self.data_source)
                
            elif os.path.isdir(self.data_source):
                self.data_type = 'folder'
                self.data = self._load_from_folder()
            
            if self.data is None or len(self.data) == 0:
                raise ValueError("Unable to load resume data")
            
            # Auto-detect fields
            self._detect_fields()
            
            return {
                'success': True,
                'data_type': self.data_type,
                'total_resumes': len(self.data),
                'columns': len(self.data.columns),
                'text_fields': len(self.text_fields),
                'categorical_fields': len(self.categorical_fields)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_from_folder(self):
        """Load resumes from folder of PDFs"""
        resumes = []
        
        # Find all PDF files recursively
        pdf_files = glob.glob(os.path.join(self.data_source, '**', '*.pdf'), recursive=True)
        
        if not pdf_files:
            return pd.DataFrame({'file_name': [], 'content': [], 'category': []})
        
        # Limit to first 500 PDFs for performance
        pdf_files = pdf_files[:500]
        
        for pdf_path in pdf_files:
            try:
                # Extract category from folder structure
                category = os.path.basename(os.path.dirname(pdf_path))
                
                # Extract text from PDF
                text = self._extract_pdf_text(pdf_path)
                
                if text and len(text.strip()) > 50:  # Valid content
                    resumes.append({
                        'file_name': os.path.basename(pdf_path),
                        'content': text,
                        'category': category,
                        'file_path': pdf_path
                    })
            except Exception as e:
                # Skip problematic PDFs
                continue
        
        return pd.DataFrame(resumes)
    
    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        
        # Try pdfplumber first (more accurate)
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages[:5]:  # First 5 pages only
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
            except:
                pass
        
        # Fallback to PyPDF2
        if PDF_AVAILABLE and not text:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages[:5]:  # First 5 pages only
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except:
                pass
        
        return text
    
    def _detect_fields(self):
        """Auto-detect resume fields"""
        for col in self.data.columns:
            # Skip if mostly null
            if self.data[col].isnull().sum() > len(self.data) * 0.9:
                continue
            
            # Text fields (long text)
            if self.data[col].dtype == 'object':
                avg_length = self.data[col].astype(str).str.len().mean()
                if avg_length > 100:  # Long text
                    self.text_fields.append(col)
                else:  # Categorical
                    self.categorical_fields.append(col)
            
            # Numeric fields
            elif pd.api.types.is_numeric_dtype(self.data[col]):
                self.numeric_fields.append(col)
    
    def extract_skills(self):
        """Extract and analyze skills from resumes - fully dynamic"""
        if not self.text_fields and len(self.data.columns) == 0:
            return {
                'available': False,
                'message': 'No text data found for skill extraction'
            }
        
        try:
            # Comprehensive skill categories
            tech_skills = [
                # Programming Languages
                'Python', 'Java', 'JavaScript', 'C++', 'C#', 'C', 'PHP', 'Ruby', 'Go', 'Rust',
                'Swift', 'Kotlin', 'TypeScript', 'Scala', 'Perl', 'R', 'MATLAB', 'VBA',
                # Databases
                'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'SQLite', 'Redis', 'Cassandra',
                'DynamoDB', 'NoSQL', 'Database', 'DB2',
                # Data Science & AI
                'Machine Learning', 'Deep Learning', 'AI', 'Data Science', 'NLP', 'Computer Vision',
                'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib',
                'Seaborn', 'SciPy', 'Statistics', 'Data Analysis', 'Data Mining', 'Big Data',
                'Hadoop', 'Spark', 'Data Visualization', 'Tableau', 'Power BI', 'Excel',
                # Web Development
                'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring',
                'ASP.NET', 'Web Development', 'REST API', 'GraphQL', 'Microservices',
                # Cloud & DevOps
                'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Cloud', 'Jenkins', 'CI/CD',
                'DevOps', 'Terraform', 'Ansible', 'Linux', 'Unix', 'Shell Scripting',
                # Other Technical
                'Git', 'GitHub', 'Agile', 'Scrum', 'JIRA', 'SAP', 'ERP', 'CRM', 'Salesforce',
                'Testing', 'QA', 'Automation', 'Selenium', 'JUnit', 'API', 'Networking',
                'Security', 'Blockchain', 'IoT', 'Mobile Development', 'Android', 'iOS'
            ]
            
            soft_skills = [
                'Leadership', 'Communication', 'Teamwork', 'Problem Solving', 'Team Building',
                'Project Management', 'Critical Thinking', 'Analytical', 'Strategic Planning',
                'Presentation', 'Collaboration', 'Mentoring', 'Coaching', 'Negotiation',
                'Time Management', 'Organizational', 'Decision Making', 'Innovation',
                'Creativity', 'Adaptability', 'Conflict Resolution', 'Client Relations',
                'Customer Service', 'Sales', 'Marketing', 'Business Development'
            ]
            
            # Extract skills from ALL text columns (dynamic)
            skill_counts = {skill: 0 for skill in tech_skills + soft_skills}
            
            # Search in text fields if available
            search_cols = self.text_fields if self.text_fields else []
            
            # If no dedicated text fields, search in all string columns
            if not search_cols:
                for col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        search_cols.append(col)
            
            for text_col in search_cols:
                text_data = self.data[text_col].astype(str).str.lower()
                for skill in tech_skills + soft_skills:
                    # Use word boundaries for better matching
                    pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                    skill_counts[skill] += text_data.str.contains(pattern, case=False, na=False, regex=True).sum()
            
            # Get top skills
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
            top_skills = [(skill, count) for skill, count in top_skills if count > 0][:30]
            
            # Categorize skills
            tech_top = [(s, c) for s, c in top_skills if s in tech_skills]
            soft_top = [(s, c) for s, c in top_skills if s in soft_skills]
            
            # Category analysis (if category column exists)
            category_skills = {}
            cat_cols = [col for col in self.data.columns if 'category' in col.lower() or 'role' in col.lower() or 'job' in col.lower()]
            if cat_cols:
                cat_col = cat_cols[0]
                categories = self.data[cat_col].value_counts().head(10)
                category_skills = {cat: int(count) for cat, count in categories.items()}
            
            return {
                'available': True,
                'total_skills_found': len(top_skills),
                'top_technical_skills': tech_top[:15],
                'top_soft_skills': soft_top[:15],
                'all_top_skills': top_skills[:20],
                'skill_distribution': {
                    'technical': len(tech_top),
                    'soft': len(soft_top),
                    'total_unique': len(top_skills)
                },
                'category_distribution': category_skills,
                'insights': f"Analyzed {len(self.data)} resumes and found {len(top_skills)} distinct skills. "
                           f"Top technical skill: {tech_top[0][0] if tech_top else 'N/A'} ({tech_top[0][1] if tech_top else 0} resumes). "
                           f"Most common soft skill: {soft_top[0][0] if soft_top else 'N/A'} ({soft_top[0][1] if soft_top else 0} resumes). "
                           f"Searched across {len(search_cols)} text columns."
            }
            
        except Exception as e:
            return {
                'available': False,
                'message': f'Skill extraction failed: {str(e)}'
            }
    
    def analyze_experience(self):
        """Analyze experience levels"""
        # Look for experience-related columns
        exp_cols = [col for col in self.data.columns if any(word in col.lower() for word in ['experience', 'year', 'work'])]
        
        if not exp_cols:
            return {
                'available': False,
                'message': 'No experience data found'
            }
        
        try:
            exp_col = exp_cols[0]
            
            # If numeric, use directly
            if pd.api.types.is_numeric_dtype(self.data[exp_col]):
                exp_values = self.data[exp_col].dropna()
            else:
                # Try to extract numbers from text
                exp_values = self.data[exp_col].astype(str).str.extract(r'(\d+)')[0].astype(float).dropna()
            
            if len(exp_values) == 0:
                return {'available': False, 'message': 'No valid experience data'}
            
            # Categorize experience
            categories = {
                'Entry Level (0-2 years)': (exp_values <= 2).sum(),
                'Mid Level (3-5 years)': ((exp_values > 2) & (exp_values <= 5)).sum(),
                'Senior (6-10 years)': ((exp_values > 5) & (exp_values <= 10)).sum(),
                'Expert (10+ years)': (exp_values > 10).sum()
            }
            
            return {
                'available': True,
                'experience_column': exp_col,
                'average_experience': float(exp_values.mean()),
                'median_experience': float(exp_values.median()),
                'min_experience': float(exp_values.min()),
                'max_experience': float(exp_values.max()),
                'distribution': categories,
                'insights': f"Average experience: {exp_values.mean():.1f} years. "
                           f"Most candidates are {max(categories, key=categories.get)}."
            }
            
        except Exception as e:
            return {
                'available': False,
                'message': f'Experience analysis failed: {str(e)}'
            }
    
    def analyze_education(self):
        """Analyze education qualifications"""
        # Look for education columns
        edu_cols = [col for col in self.data.columns if any(word in col.lower() for word in ['education', 'degree', 'qualification', 'university', 'college'])]
        
        if not edu_cols:
            return {
                'available': False,
                'message': 'No education data found'
            }
        
        try:
            edu_col = edu_cols[0]
            edu_data = self.data[edu_col].astype(str).str.lower()
            
            # Education levels
            levels = {
                'PhD/Doctorate': edu_data.str.contains('phd|doctorate|doctoral', case=False, na=False).sum(),
                'Masters/MS/MBA': edu_data.str.contains('master|ms|mba|m\\.s', case=False, na=False).sum(),
                'Bachelors/BS': edu_data.str.contains('bachelor|bs|b\\.s|btech|be\\b', case=False, na=False).sum(),
                'Diploma/Certificate': edu_data.str.contains('diploma|certificate', case=False, na=False).sum()
            }
            
            total_classified = sum(levels.values())
            
            return {
                'available': True,
                'education_column': edu_col,
                'total_resumes': len(self.data),
                'classified': total_classified,
                'distribution': levels,
                'insights': f"Analyzed education for {len(self.data)} resumes. "
                           f"Most common qualification: {max(levels, key=levels.get)} ({levels[max(levels, key=levels.get)]} candidates)."
            }
            
        except Exception as e:
            return {
                'available': False,
                'message': f'Education analysis failed: {str(e)}'
            }
    
    def candidate_ranking(self):
        """Rank candidates based on multiple criteria"""
        if len(self.data) == 0:
            return {'available': False, 'message': 'No data to rank'}
        
        try:
            # Create scoring system
            scores = pd.DataFrame(index=self.data.index)
            scores['total_score'] = 0
            
            # Score based on text length (proxy for detail)
            if self.text_fields:
                text_col = self.text_fields[0]
                scores['detail_score'] = self.data[text_col].astype(str).str.len() / 1000
                scores['total_score'] += scores['detail_score']
            
            # Score based on categorical diversity
            if self.categorical_fields:
                for cat_col in self.categorical_fields[:3]:
                    # Higher score for less common categories
                    value_counts = self.data[cat_col].value_counts()
                    scores[f'{cat_col}_score'] = self.data[cat_col].map(lambda x: 1 / (value_counts.get(x, 1)))
                    scores['total_score'] += scores[f'{cat_col}_score']
            
            # Get top candidates
            top_indices = scores['total_score'].nlargest(10).index
            top_candidates = self.data.loc[top_indices].copy()
            top_candidates['score'] = scores.loc[top_indices, 'total_score']
            
            return {
                'available': True,
                'total_candidates': len(self.data),
                'scoring_criteria': list(scores.columns),
                'top_10_candidates': top_candidates.to_dict('records'),
                'insights': f"Ranked {len(self.data)} candidates. "
                           f"Top candidate scored {scores['total_score'].max():.2f}. "
                           f"Average score: {scores['total_score'].mean():.2f}"
            }
            
        except Exception as e:
            return {
                'available': False,
                'message': f'Ranking failed: {str(e)}'
            }
    
    def generate_hiring_recommendations(self, skills_data, experience_data, education_data, ranking_data):
        """Generate intelligent hiring recommendations based on analysis"""
        recommendations = []
        
        # Skill-based recommendations
        if skills_data.get('available'):
            top_tech = skills_data.get('top_technical_skills', [])[:3]
            if top_tech:
                recommendations.append({
                    'category': 'High-Demand Skills',
                    'priority': 'HIGH',
                    'insight': f"Focus hiring on candidates with {', '.join([s[0] for s in top_tech])}. These are the most common skills in your talent pool.",
                    'action': f"Create job postings targeting {top_tech[0][0]} experts - {top_tech[0][1]} qualified candidates available"
                })
        
        # Experience-based recommendations
        if experience_data.get('available'):
            avg_exp = experience_data.get('average_experience', 0)
            distribution = experience_data.get('distribution', {})
            max_category = max(distribution, key=distribution.get) if distribution else None
            
            if max_category:
                recommendations.append({
                    'category': 'Experience Level',
                    'priority': 'MEDIUM',
                    'insight': f"Most candidates are {max_category} with average {avg_exp:.1f} years experience.",
                    'action': f"Optimize compensation packages for {max_category} professionals"
                })
        
        # Education-based recommendations
        if education_data.get('available'):
            edu_dist = education_data.get('distribution', {})
            top_qual = max(edu_dist, key=edu_dist.get) if edu_dist else None
            
            if top_qual:
                recommendations.append({
                    'category': 'Education Quality',
                    'priority': 'MEDIUM',
                    'insight': f"Majority hold {top_qual} - {edu_dist[top_qual]} candidates available.",
                    'action': f"Fast-track screening for {top_qual} holders to reduce time-to-hire"
                })
        
        # Category-based recommendations
        if skills_data.get('category_distribution'):
            cat_dist = skills_data['category_distribution']
            top_roles = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_roles:
                recommendations.append({
                    'category': 'Talent Pool Distribution',
                    'priority': 'HIGH',
                    'insight': f"Largest talent pools: {', '.join([f'{r[0]} ({r[1]})' for r in top_roles])}",
                    'action': f"Prioritize recruitment for {top_roles[0][0]} roles where supply is highest"
                })
        
        return recommendations
    
    def generate_resume_report(self):
        """Generate comprehensive resume analysis report with actionable insights"""
        load_result = self.load_and_preprocess()
        
        if not load_result.get('success'):
            return {
                'error': load_result.get('error'),
                'available': False
            }
        
        # Run all analyses
        skills = self.extract_skills()
        experience = self.analyze_experience()
        education = self.analyze_education()
        ranking = self.candidate_ranking()
        
        # Generate intelligent recommendations
        recommendations = self.generate_hiring_recommendations(skills, experience, education, ranking)
        
        # Build executive summary with actionable insights
        summary_parts = [
            "=" * 80,
            "COGNIFYX RESUME INTELLIGENCE REPORT",
            "=" * 80,
            f"\n📊 DATASET OVERVIEW",
            f"   Source: {self.data_source}",
            f"   Total Resumes: {len(self.data):,}",
            f"   Data Type: {self.data_type.upper()}",
            f"   Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        # Skills insights
        if skills.get('available'):
            summary_parts.extend([
                f"\n💻 SKILLS ANALYSIS",
                f"   {skills.get('insights', 'Not available')}",
                f"   Total Unique Skills: {skills['skill_distribution']['total_unique']}",
                f"   Technical Skills Found: {skills['skill_distribution']['technical']}",
                f"   Soft Skills Found: {skills['skill_distribution']['soft']}",
                f"\n   🔥 Top 5 Technical Skills:"
            ])
            for skill, count in skills.get('top_technical_skills', [])[:5]:
                percentage = (count / len(self.data)) * 100
                summary_parts.append(f"      • {skill}: {count} resumes ({percentage:.1f}%)")
        
        # Experience insights
        if experience.get('available'):
            summary_parts.extend([
                f"\n💼 EXPERIENCE ANALYSIS",
                f"   {experience.get('insights', 'Not available')}",
                f"   Average: {experience['average_experience']:.1f} years",
                f"   Range: {experience['min_experience']:.0f} - {experience['max_experience']:.0f} years",
                f"   Median: {experience['median_experience']:.1f} years"
            ])
        
        # Education insights
        if education.get('available'):
            summary_parts.extend([
                f"\n🎓 EDUCATION ANALYSIS",
                f"   {education.get('insights', 'Not available')}",
                f"   Distribution:"
            ])
            for qual, count in sorted(education['distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.data)) * 100
                summary_parts.append(f"      • {qual}: {count} ({percentage:.1f}%)")
        
        # Candidate ranking insights
        if ranking.get('available'):
            summary_parts.extend([
                f"\n🏆 CANDIDATE RANKING",
                f"   {ranking.get('insights', 'Not available')}",
                f"   Total Evaluated: {ranking['total_candidates']:,}",
                f"   Scoring Criteria: {', '.join(ranking['scoring_criteria'][:3])}"
            ])
        
        # Strategic recommendations
        if recommendations:
            summary_parts.extend([
                f"\n⚡ STRATEGIC RECOMMENDATIONS",
                f"   Generated {len(recommendations)} actionable insights:\n"
            ])
            for rec in recommendations:
                summary_parts.extend([
                    f"   [{rec['priority']}] {rec['category']}",
                    f"      💡 {rec['insight']}",
                    f"      ✅ {rec['action']}\n"
                ])
        
        # Use cases and value proposition
        summary_parts.extend([
            f"\n🎯 KEY USE CASES FOR THIS ANALYSIS:",
            f"   1. Talent Pool Optimization - Identify high-supply skill areas",
            f"   2. Compensation Planning - Align offers with experience levels",
            f"   3. Job Description Refinement - Focus on in-demand skills",
            f"   4. Recruitment Strategy - Target abundant candidate categories",
            f"   5. Screening Automation - Prioritize top-scoring candidates",
            f"\n💼 BUSINESS VALUE:",
            f"   • Reduce time-to-hire by targeting abundant talent pools",
            f"   • Optimize recruitment ROI by focusing on qualified candidates",
            f"   • Data-driven hiring decisions backed by {len(self.data):,} resume analysis",
            f"   • Competitive intelligence on skill trends in the market",
            "=" * 80
        ])
        
        summary_context = "\n".join(summary_parts)
        
        return {
            'available': True,
            'load_info': load_result,
            'skills_analysis': skills,
            'experience_analysis': experience,
            'education_analysis': education,
            'candidate_ranking': ranking,
            'recommendations': recommendations,
            'executive_summary': summary_context,
            'generated_at': datetime.now().isoformat(),
            'use_cases': [
                'Talent Pool Analysis',
                'Skills Gap Identification',
                'Hiring Strategy Optimization',
                'Compensation Benchmarking',
                'Candidate Screening Automation'
            ]
        }
