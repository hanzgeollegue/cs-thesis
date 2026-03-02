"""
Deterministic Natural Language Generation for Resume Analysis

This module provides fast, deterministic text generation for candidate analyses
and pairwise comparisons using pure Python templates and rule-based logic.
No LLMs are used - all text is generated from structured data and templates.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of pairwise candidate comparison."""
    candidate_a_id: str
    candidate_b_id: str
    winner: Optional[str]  # "a", "b", or None for tie
    margin: float  # Score difference
    comparison_text: str
    key_differences: List[str]


class FactExtractor:
    """Extracts auditable facts from candidate data for transparency."""
    
    def extract_facts(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured facts from candidate data for audit trail.
        
        Args:
            candidate_data: Candidate resume data with scores and parsed content
            jd_criteria: Job description criteria
            
        Returns:
            Dictionary with extracted facts for transparency
        """
        try:
            scores = candidate_data.get('scores', {})
            parsed = candidate_data.get('parsed', {})
            meta = candidate_data.get('meta', {})
            
            facts = {
                # Score breakdown
                'scores': {
                    'final_score': float(scores.get('final_score', 0.0)),
                    'final_score_percentage': round(float(scores.get('final_score', 0.0)) * 100, 1),
                    'coverage': float(scores.get('coverage', 0.0)),
                    'has_match_skills': bool(scores.get('has_match_skills', False)),
                    'has_match_experience': bool(scores.get('has_match_experience', False))
                },
                
                # Skills analysis
                'skills': {
                    'matched_required': scores.get('matched_required_skills', []),
                    'matched_nice': scores.get('matched_nice_skills', []),
                    'missing_required': scores.get('missing_skills', []),
                    'total_required': len(jd_criteria.get('must_have_skills', [])),
                    'total_nice': len(jd_criteria.get('nice_to_have_skills', []))
                },
                
                # Experience summary
                'experience': self._extract_experience_facts(parsed),
                
                # Education summary
                'education': self._extract_education_facts(parsed),
                
                # Projects summary
                'projects': self._extract_projects_facts(parsed),
                
                # Metadata
                'metadata': {
                    'candidate_name': self._extract_candidate_name(candidate_data),
                    'source_file': meta.get('source_file', 'Unknown'),
                    'parsing_quality': self._assess_parsing_quality(parsed)
                }
            }
            
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return {'error': str(e)}
    
    def _extract_experience_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract experience-related facts."""
        experience_items = parsed.get('experience', [])
        
        if not experience_items:
            return {'count': 0, 'total_years': 0, 'top_roles': [], 'has_relevant': False}
        
        # Calculate total years (rough estimate)
        total_years = 0
        top_roles = []
        
        for exp in experience_items[:5]:  # Top 5 roles
            title = exp.get('title', '')
            company = exp.get('company', '')
            if title:
                top_roles.append(f"{title} at {company}" if company else title)
        
        # Estimate years from role count and typical durations
        if len(experience_items) >= 3:
            total_years = len(experience_items) * 2  # Rough estimate
        elif len(experience_items) >= 1:
            total_years = len(experience_items) * 1.5
        
        return {
            'count': len(experience_items),
            'total_years': total_years,
            'top_roles': top_roles,
            'has_relevant': len(experience_items) > 0
        }
    
    def _extract_education_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract education-related facts."""
        education_items = parsed.get('education', [])
        
        if not education_items:
            return {'count': 0, 'degrees': [], 'has_relevant': False}
        
        degrees = []
        for edu in education_items:
            degree = edu.get('degree', '')
            school = edu.get('school', '')
            if degree:
                degrees.append(f"{degree} from {school}" if school else degree)
        
        return {
            'count': len(education_items),
            'degrees': degrees,
            'has_relevant': len(education_items) > 0
        }
    
    def _extract_projects_facts(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Extract project-related facts."""
        projects = parsed.get('projects', [])
        
        if not projects:
            return {'count': 0, 'top_projects': [], 'technologies': []}
        
        top_projects = []
        all_technologies = []
        
        for project in projects[:3]:  # Top 3 projects
            name = project.get('name', '')
            if name:
                top_projects.append(name)
            
            techs = project.get('technologies', [])
            if techs:
                all_technologies.extend(techs)
        
        # Deduplicate technologies
        unique_technologies = list(dict.fromkeys(all_technologies))
        
        return {
            'count': len(projects),
            'top_projects': top_projects,
            'technologies': unique_technologies
        }
    
    def _extract_candidate_name(self, candidate_data: Dict[str, Any]) -> str:
        """Extract candidate name from various possible locations."""
        # Try parsed data first
        parsed = candidate_data.get('parsed', {})
        if isinstance(parsed, dict):
            name = (parsed.get('candidate_name') or 
                   parsed.get('name') or 
                   parsed.get('parsed_name'))
            if name:
                return str(name).strip()
        
        # Try meta data
        meta = candidate_data.get('meta', {})
        if isinstance(meta, dict):
            name = meta.get('candidate_name') or meta.get('name')
            if name:
                return str(name).strip()
        
        # Fallback to filename
        source_file = meta.get('source_file', 'Unknown.pdf')
        base_name = source_file.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        return base_name.title()
    
    def _assess_parsing_quality(self, parsed: Dict[str, Any]) -> str:
        """Assess the quality of resume parsing."""
        has_experience = len(parsed.get('experience', [])) > 0
        has_skills = len(parsed.get('skills', [])) > 0
        has_education = len(parsed.get('education', [])) > 0
        
        if has_experience and has_skills and has_education:
            return 'excellent'
        elif has_experience and has_skills:
            return 'good'
        elif has_experience or has_skills:
            return 'fair'
        else:
            return 'poor'


class CandidateAnalyzer:
    """Generates per-candidate analysis summaries using templates and rules."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def generate_analysis(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> str:
        """
        Generate a 2-4 sentence analysis summary for a candidate.
        
        Args:
            candidate_data: Candidate resume data with scores and parsed content
            jd_criteria: Job description criteria
            
        Returns:
            Natural language analysis text
        """
        try:
            facts = FactExtractor().extract_facts(candidate_data, jd_criteria)
            scores = facts['scores']
            skills = facts['skills']
            experience = facts['experience']
            
            # Determine score tier
            score_pct = scores['final_score_percentage']
            tier = self._get_score_tier(score_pct)
            
            # Generate analysis based on tier and facts
            analysis_parts = []
            
            # Opening sentence - score and overall assessment
            opening = self._generate_opening_sentence(score_pct, tier, skills)
            analysis_parts.append(opening)
            
            # Skills assessment
            skills_analysis = self._generate_skills_analysis(skills, tier)
            if skills_analysis:
                analysis_parts.append(skills_analysis)
            
            # Experience assessment
            experience_analysis = self._generate_experience_analysis(experience, tier)
            if experience_analysis:
                analysis_parts.append(experience_analysis)
            
            # Gap analysis or recommendations
            gap_analysis = self._generate_gap_analysis(skills, experience, tier)
            if gap_analysis:
                analysis_parts.append(gap_analysis)
            
            # Join parts and clean up
            analysis = '. '.join(analysis_parts)
            if not analysis.endswith('.'):
                analysis += '.'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return f"Analysis unavailable due to processing error: {str(e)}"
    
    def _get_score_tier(self, score_pct: float) -> str:
        """Determine score tier for template selection."""
        if score_pct >= 80:
            return 'excellent'
        elif score_pct >= 60:
            return 'strong'
        elif score_pct >= 40:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_opening_sentence(self, score_pct: float, tier: str, skills: Dict[str, Any]) -> str:
        """Generate the opening assessment sentence."""
        matched_count = len(skills['matched_required'])
        total_required = skills['total_required']
        
        if tier == 'excellent':
            if score_pct >= 95:
                return f"Exceptional match with {score_pct:.0f}% alignment and {matched_count}/{total_required} required skills"
            else:
                return f"Strong candidate with {score_pct:.0f}% match and {matched_count}/{total_required} required skills"
        elif tier == 'strong':
            return f"Good candidate with {score_pct:.0f}% match and {matched_count}/{total_required} required skills"
        elif tier == 'moderate':
            return f"Moderate candidate with {score_pct:.0f}% match and {matched_count}/{total_required} required skills"
        else:
            return f"Limited match with {score_pct:.0f}% alignment and {matched_count}/{total_required} required skills"
    
    def _generate_skills_analysis(self, skills: Dict[str, Any], tier: str) -> str:
        """Generate skills-focused analysis."""
        matched_required = skills['matched_required']
        missing_required = skills['missing_required']
        matched_nice = skills['matched_nice']
        
        if not matched_required and not matched_nice:
            return "No matching skills identified in resume"
        
        analysis_parts = []
        
        # Required skills
        if matched_required:
            if len(matched_required) == 1:
                analysis_parts.append(f"Has required skill: {matched_required[0]}")
            elif len(matched_required) <= 3:
                analysis_parts.append(f"Required skills: {', '.join(matched_required)}")
            else:
                top_skills = matched_required[:3]
                analysis_parts.append(f"Key required skills: {', '.join(top_skills)} (and {len(matched_required)-3} more)")
        
        # Nice-to-have skills
        if matched_nice:
            if len(matched_nice) == 1:
                analysis_parts.append(f"Bonus skill: {matched_nice[0]}")
            else:
                analysis_parts.append(f"Additional skills: {', '.join(matched_nice[:3])}")
        
        # Missing critical skills
        if missing_required and tier in ['weak', 'moderate']:
            if len(missing_required) == 1:
                analysis_parts.append(f"Missing critical skill: {missing_required[0]}")
            else:
                analysis_parts.append(f"Missing skills: {', '.join(missing_required[:3])}")
        
        return '; '.join(analysis_parts)
    
    def _generate_experience_analysis(self, experience: Dict[str, Any], tier: str) -> str:
        """Generate experience-focused analysis."""
        if not experience['has_relevant']:
            return "No work experience provided"
        
        count = experience['count']
        top_roles = experience['top_roles']
        
        if count == 1:
            return f"Experience: {top_roles[0]}"
        elif count <= 3:
            return f"Experience includes: {'; '.join(top_roles)}"
        else:
            return f"Extensive experience: {'; '.join(top_roles[:2])} (and {count-2} more roles)"
    
    def _generate_gap_analysis(self, skills: Dict[str, Any], experience: Dict[str, Any], tier: str) -> str:
        """Generate gap analysis or recommendations."""
        missing_required = skills['missing_required']
        
        if tier == 'excellent':
            return "Strong overall fit with minor skill gaps"
        elif tier == 'strong':
            if missing_required:
                return f"Good fit but needs: {', '.join(missing_required[:2])}"
            else:
                return "Solid candidate with room for growth"
        elif tier == 'moderate':
            if missing_required:
                return f"Potential candidate requiring: {', '.join(missing_required[:2])}"
            else:
                return "Moderate fit with development potential"
        else:
            if missing_required:
                return f"Significant skill gaps in: {', '.join(missing_required[:2])}"
            else:
                return "Limited match requiring substantial development"
    
    def _initialize_templates(self) -> Dict[str, Any]:
        """Initialize template library for different scenarios."""
        return {
            'score_tiers': {
                'excellent': {
                    'tone': 'positive',
                    'focus': 'strengths',
                    'gaps': 'minor'
                },
                'strong': {
                    'tone': 'positive',
                    'focus': 'balanced',
                    'gaps': 'addressable'
                },
                'moderate': {
                    'tone': 'neutral',
                    'focus': 'potential',
                    'gaps': 'significant'
                },
                'weak': {
                    'tone': 'cautious',
                    'focus': 'gaps',
                    'gaps': 'major'
                }
            }
        }


class PairwiseComparator:
    """Generates pairwise candidate comparisons."""
    
    def compare(self, candidate_a: Dict[str, Any], candidate_b: Dict[str, Any], 
                jd_criteria: Dict[str, Any]) -> ComparisonResult:
        """
        Generate pairwise comparison between two candidates.
        
        Args:
            candidate_a: First candidate data
            candidate_b: Second candidate data
            jd_criteria: Job description criteria
            
        Returns:
            ComparisonResult with analysis and key differences
        """
        try:
            facts_a = FactExtractor().extract_facts(candidate_a, jd_criteria)
            facts_b = FactExtractor().extract_facts(candidate_b, jd_criteria)
            
            score_a = facts_a['scores']['final_score_percentage']
            score_b = facts_b['scores']['final_score_percentage']
            
            # Determine winner and margin
            margin = abs(score_a - score_b)
            if score_a > score_b:
                winner = 'a'
            elif score_b > score_a:
                winner = 'b'
            else:
                winner = None
            
            # Get candidate names for use in comparisons
            name_a = facts_a['metadata']['candidate_name']
            name_b = facts_b['metadata']['candidate_name']
            
            # Generate comparison text
            comparison_text = self._generate_comparison_text(
                candidate_a, candidate_b, facts_a, facts_b, winner, margin
            )
            
            # Extract key differences
            key_differences = self._extract_key_differences(facts_a, facts_b, name_a, name_b)
            
            return ComparisonResult(
                candidate_a_id=str(candidate_a.get('id', 'A')),
                candidate_b_id=str(candidate_b.get('id', 'B')),
                winner=winner,
                margin=margin,
                comparison_text=comparison_text,
                key_differences=key_differences
            )
            
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return ComparisonResult(
                candidate_a_id=str(candidate_a.get('id', 'A')),
                candidate_b_id=str(candidate_b.get('id', 'B')),
                winner=None,
                margin=0.0,
                comparison_text=f"Comparison unavailable: {str(e)}",
                key_differences=[]
            )
    
    def _generate_comparison_text(self, candidate_a: Dict[str, Any], candidate_b: Dict[str, Any],
                                 facts_a: Dict[str, Any], facts_b: Dict[str, Any],
                                 winner: Optional[str], margin: float) -> str:
        """Generate natural language comparison text."""
        name_a = facts_a['metadata']['candidate_name']
        name_b = facts_b['metadata']['candidate_name']
        score_a = facts_a['scores']['final_score_percentage']
        score_b = facts_b['scores']['final_score_percentage']
        
        # Opening comparison
        if winner == 'a':
            opening = f"{name_a} edges {name_b} ({score_a:.0f}% vs {score_b:.0f}%)"
        elif winner == 'b':
            opening = f"{name_b} edges {name_a} ({score_b:.0f}% vs {score_a:.0f}%)"
        else:
            opening = f"{name_a} and {name_b} are closely matched ({score_a:.0f}% vs {score_b:.0f}%)"
        
        # Skills comparison
        skills_comparison = self._compare_skills(facts_a['skills'], facts_b['skills'], name_a, name_b)
        
        # Experience comparison
        exp_comparison = self._compare_experience(facts_a['experience'], facts_b['experience'], name_a, name_b)
        
        # Combine parts
        parts = [opening]
        if skills_comparison:
            parts.append(skills_comparison)
        if exp_comparison:
            parts.append(exp_comparison)
        
        return '. '.join(parts) + '.'
    
    def _compare_skills(self, skills_a: Dict[str, Any], skills_b: Dict[str, Any], 
                        name_a: str = "Candidate A", name_b: str = "Candidate B") -> str:
        """Compare skills between candidates."""
        matched_a = skills_a['matched_required']
        matched_b = skills_b['matched_required']
        
        if len(matched_a) > len(matched_b):
            return f"{name_a} has stronger required skills"
        elif len(matched_b) > len(matched_a):
            return f"{name_b} has stronger required skills"
        else:
            # Same count, compare specific skills
            unique_a = set(matched_a) - set(matched_b)
            unique_b = set(matched_b) - set(matched_a)
            
            if unique_a and unique_b:
                return f"{name_a} has {', '.join(list(unique_a)[:2])} while {name_b} has {', '.join(list(unique_b)[:2])}"
            elif unique_a:
                return f"{name_a} has additional skills: {', '.join(list(unique_a)[:2])}"
            elif unique_b:
                return f"{name_b} has additional skills: {', '.join(list(unique_b)[:2])}"
            else:
                return "Both have similar skill sets"
    
    def _compare_experience(self, exp_a: Dict[str, Any], exp_b: Dict[str, Any],
                            name_a: str = "Candidate A", name_b: str = "Candidate B") -> str:
        """Compare experience between candidates."""
        count_a = exp_a['count']
        count_b = exp_b['count']
        
        if count_a > count_b:
            return f"{name_a} has more extensive experience ({count_a} vs {count_b} roles)"
        elif count_b > count_a:
            return f"{name_b} has more extensive experience ({count_b} vs {count_a} roles)"
        else:
            return "Both have similar experience levels"
    
    def _extract_key_differences(self, facts_a: Dict[str, Any], facts_b: Dict[str, Any],
                                  name_a: str = "Candidate A", name_b: str = "Candidate B") -> List[str]:
        """Extract key differences for bullet points."""
        differences = []
        
        # Score difference
        score_diff = abs(facts_a['scores']['final_score_percentage'] - facts_b['scores']['final_score_percentage'])
        if score_diff > 5:
            if facts_a['scores']['final_score_percentage'] > facts_b['scores']['final_score_percentage']:
                differences.append(f"{name_a} scores {score_diff:.0f}% higher overall")
            else:
                differences.append(f"{name_b} scores {score_diff:.0f}% higher overall")
        
        # Skills differences
        skills_a = set(facts_a['skills']['matched_required'])
        skills_b = set(facts_b['skills']['matched_required'])
        
        unique_a = skills_a - skills_b
        unique_b = skills_b - skills_a
        
        if unique_a:
            differences.append(f"{name_a} has unique skills: {', '.join(list(unique_a)[:2])}")
        if unique_b:
            differences.append(f"{name_b} has unique skills: {', '.join(list(unique_b)[:2])}")
        
        # Experience differences
        exp_a = facts_a['experience']['count']
        exp_b = facts_b['experience']['count']
        
        if abs(exp_a - exp_b) > 1:
            if exp_a > exp_b:
                differences.append(f"{name_a} has {exp_a - exp_b} more experience roles")
            else:
                differences.append(f"{name_b} has {exp_b - exp_a} more experience roles")
        
        return differences[:3]  # Limit to top 3 differences


# Convenience functions for easy integration
def generate_candidate_analysis(candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> str:
    """Generate analysis text for a single candidate."""
    analyzer = CandidateAnalyzer()
    return analyzer.generate_analysis(candidate_data, jd_criteria)


def generate_candidate_facts(candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Extract facts for a single candidate."""
    extractor = FactExtractor()
    return extractor.extract_facts(candidate_data, jd_criteria)


def generate_pairwise_comparison(candidate_a: Dict[str, Any], candidate_b: Dict[str, Any], 
                                jd_criteria: Dict[str, Any]) -> ComparisonResult:
    """Generate pairwise comparison between two candidates."""
    comparator = PairwiseComparator()
    return comparator.compare(candidate_a, candidate_b, jd_criteria)
