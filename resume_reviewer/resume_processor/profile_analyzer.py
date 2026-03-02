"""
Profile analyzer for detecting seniority, career progression, and specializations.
Provides context-aware insights for personalized NLG.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from .nlg_config import get_domain_config, DomainConfig

logger = logging.getLogger(__name__)


class ProfileAnalyzer:
    """Analyzes candidate profiles for context-aware NLG."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        
        # Seniority detection patterns
        self.seniority_patterns = {
            'intern': [
                r'\bintern\b', r'\binternship\b', r'\btrainee\b', r'\bapprentice\b'
            ],
            'junior': [
                r'\bjunior\b', r'\bentry\b', r'\bassociate\b', r'\bassistant\b',
                r'\bnew grad\b', r'\brecent grad\b', r'\bgraduate\b'
            ],
            'mid': [
                r'\bmid\b', r'\bmiddle\b', r'\bintermediate\b', r'\bdeveloper\b',
                r'\bengineer\b', r'\banalyst\b', r'\bspecialist\b'
            ],
            'senior': [
                r'\bsenior\b', r'\bprincipal\b', r'\blead\b', r'\barchitect\b',
                r'\bexpert\b', r'\bconsultant\b'
            ],
            'lead': [
                r'\blead\b', r'\bteam lead\b', r'\btech lead\b', r'\bmanager\b',
                r'\bhead of\b', r'\bdirector\b'
            ],
            'staff': [
                r'\bstaff\b', r'\bdistinguished\b', r'\bfellow\b', r'\bchief\b',
                r'\bvp\b', r'\bvice president\b'
            ]
        }
        
        # Technical specialization patterns
        self.tech_specializations = {
            'backend': [
                r'\bbackend\b', r'\bserver\b', r'\bapi\b', r'\bmicroservices\b',
                r'\bdatabase\b', r'\bsql\b', r'\bpostgresql\b', r'\bmysql\b',
                r'\bmongodb\b', r'\bredis\b', r'\bnode\.js\b', r'\bjava\b',
                r'\bpython\b', r'\bgo\b', r'\brust\b', r'\bc\+\+\b'
            ],
            'frontend': [
                r'\bfrontend\b', r'\bfront-end\b', r'\bui\b', r'\bux\b',
                r'\buser interface\b', r'\buser experience\b', r'\bweb\b',
                r'\bhtml\b', r'\bcss\b', r'\bjavascript\b', r'\btypescript\b',
                r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bsvelte\b'
            ],
            'devops': [
                r'\bdevops\b', r'\bdev-ops\b', r'\bci/cd\b', r'\bdeployment\b',
                r'\bkubernetes\b', r'\bdocker\b', r'\baws\b', r'\bazure\b',
                r'\bgcp\b', r'\bterraform\b', r'\bansible\b', r'\bjenkins\b',
                r'\bmonitoring\b', r'\blogging\b'
            ],
            'data': [
                r'\bdata\b', r'\banalytics\b', r'\bmachine learning\b', r'\bml\b',
                r'\bai\b', r'\bartificial intelligence\b', r'\bstatistics\b',
                r'\bpython\b', r'\br\b', r'\bsql\b', r'\btableau\b', r'\bpower bi\b',
                r'\btensorflow\b', r'\bpytorch\b', r'\bscikit-learn\b'
            ],
            'mobile': [
                r'\bmobile\b', r'\bios\b', r'\bandroid\b', r'\bswift\b',
                r'\bkotlin\b', r'\bflutter\b', r'\breact native\b', r'\bionic\b'
            ],
            'security': [
                r'\bsecurity\b', r'\bcybersecurity\b', r'\bpenetration\b',
                r'\bvulnerability\b', r'\bthreat\b', r'\brisk\b', r'\bcompliance\b'
            ]
        }
        
        # Leadership indicators
        self.leadership_indicators = [
            r'\bled\b', r'\bmanaged\b', r'\bmentored\b', r'\bcoached\b',
            r'\btrained\b', r'\bguided\b', r'\bdirected\b', r'\boversaw\b',
            r'\bsupervised\b', r'\bcoordinated\b', r'\bfacilitated\b',
            r'\bteam of\b', r'\bgroup of\b', r'\bdepartment\b'
        ]
        
        # Rare qualification patterns
        self.rare_qualifications = [
            r'\bphd\b', r'\bdoctorate\b', r'\bpatent\b', r'\bpublication\b',
            r'\bpublished\b', r'\bresearch\b', r'\bthesis\b', r'\bdissertation\b',
            r'\bconference\b', r'\bspeaker\b', r'\bkeynote\b', r'\baward\b',
            r'\bhonor\b', r'\bdistinction\b', r'\bexcellence\b'
        ]
    
    def detect_seniority(self, candidate_data: Dict[str, Any]) -> str:
        """
        Detect seniority level from candidate data.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            Detected seniority level (intern/junior/mid/senior/lead/staff)
        """
        try:
            parsed = candidate_data.get('parsed', {})
            experience_items = parsed.get('experience', [])
            
            # Extract all text for analysis
            all_text = []
            
            # Add experience titles and descriptions
            for exp in experience_items:
                title = exp.get('title', '')
                company = exp.get('company', '')
                bullets = exp.get('bullets', [])
                
                if title:
                    all_text.append(title)
                if company:
                    all_text.append(company)
                if bullets:
                    all_text.extend(bullets)
            
            # Add education
            education_items = parsed.get('education', [])
            for edu in education_items:
                degree = edu.get('degree', '')
                school = edu.get('school', '')
                if degree:
                    all_text.append(degree)
                if school:
                    all_text.append(school)
            
            # Combine all text
            combined_text = ' '.join(all_text).lower()
            
            # Score each seniority level
            seniority_scores = {}
            for level, patterns in self.seniority_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, combined_text, re.IGNORECASE)
                    score += len(matches)
                seniority_scores[level] = score
            
            # Find highest scoring level
            if not any(seniority_scores.values()):
                # Fallback: estimate from experience count
                exp_count = len(experience_items)
                if exp_count == 0:
                    return 'intern'
                elif exp_count <= 2:
                    return 'junior'
                elif exp_count <= 4:
                    return 'mid'
                else:
                    return 'senior'
            
            best_level = max(seniority_scores, key=seniority_scores.get)
            
            # Refine based on experience count
            exp_count = len(experience_items)
            if best_level == 'intern' and exp_count > 1:
                return 'junior'
            elif best_level == 'junior' and exp_count > 3:
                return 'mid'
            elif best_level == 'mid' and exp_count > 5:
                return 'senior'
            
            return best_level
            
        except Exception as e:
            logger.warning(f"Error detecting seniority: {e}")
            return 'mid'  # Safe fallback
    
    def assess_career_progression(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess career progression trajectory.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            Dictionary with progression analysis
        """
        try:
            parsed = candidate_data.get('parsed', {})
            experience_items = parsed.get('experience', [])
            
            if len(experience_items) < 2:
                return {
                    'trajectory': 'insufficient_data',
                    'velocity': 0,
                    'progression_type': 'unknown',
                    'leadership_growth': False
                }
            
            # Analyze title progression
            titles = [exp.get('title', '').lower() for exp in experience_items]
            
            # Detect progression patterns
            progression_score = 0
            leadership_growth = False
            
            for i in range(len(titles) - 1):
                current_title = titles[i]
                next_title = titles[i + 1]
                
                # Check for seniority increase
                if self._is_seniority_increase(current_title, next_title):
                    progression_score += 1
                
                # Check for leadership growth
                if self._has_leadership_growth(current_title, next_title):
                    leadership_growth = True
            
            # Determine trajectory
            if progression_score >= 2:
                trajectory = 'ascending'
            elif progression_score == 1:
                trajectory = 'moderate_growth'
            else:
                trajectory = 'lateral'
            
            # Calculate velocity (promotions per year, rough estimate)
            total_years = len(experience_items) * 2  # Rough estimate
            velocity = progression_score / max(total_years, 1)
            
            return {
                'trajectory': trajectory,
                'velocity': velocity,
                'progression_type': 'technical' if not leadership_growth else 'leadership',
                'leadership_growth': leadership_growth,
                'promotion_count': progression_score
            }
            
        except Exception as e:
            logger.warning(f"Error assessing career progression: {e}")
            return {
                'trajectory': 'unknown',
                'velocity': 0,
                'progression_type': 'unknown',
                'leadership_growth': False
            }
    
    def identify_specializations(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify technical specializations from candidate data.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            Dictionary with specialization analysis
        """
        try:
            parsed = candidate_data.get('parsed', {})
            
            # Extract all text
            all_text = []
            
            # Add experience
            for exp in parsed.get('experience', []):
                title = exp.get('title', '')
                bullets = exp.get('bullets', [])
                if title:
                    all_text.append(title)
                if bullets:
                    all_text.extend(bullets)
            
            # Add projects
            for project in parsed.get('projects', []):
                name = project.get('name', '')
                summary = project.get('summary', '')
                technologies = project.get('technologies', [])
                if name:
                    all_text.append(name)
                if summary:
                    all_text.append(summary)
                if technologies:
                    all_text.extend(technologies)
            
            # Add skills
            skills = parsed.get('skills', [])
            if skills:
                # Handle both string and dict skill formats
                for skill in skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            all_text.append(skill_name)
                    elif isinstance(skill, str):
                        all_text.append(skill)
            
            # Combine and analyze
            combined_text = ' '.join(all_text).lower()
            
            # Score each specialization
            specialization_scores = {}
            for spec, patterns in self.tech_specializations.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, combined_text, re.IGNORECASE)
                    score += len(matches)
                specialization_scores[spec] = score
            
            # Find primary and secondary specializations
            sorted_specs = sorted(specialization_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Gate specialization output by evidence threshold
            SPECIALIZATION_MIN_CONFIDENCE = 0.5  # Require at least 50% confidence
            SPECIALIZATION_MIN_SIGNALS = 2  # Require at least 2 distinct signals
            
            total_score = sum(specialization_scores.values())
            max_score = max(specialization_scores.values()) if specialization_scores else 0
            confidence = min(max_score / 5.0, 1.0)  # Normalize confidence
            
            # Count distinct signals (non-zero scores)
            distinct_signals = len([score for score in specialization_scores.values() if score > 0])
            
            # Only return specialization if thresholds met
            if confidence < SPECIALIZATION_MIN_CONFIDENCE or distinct_signals < SPECIALIZATION_MIN_SIGNALS:
                return {
                    'primary': None,
                    'secondary': [],
                    'type': 'unknown',
                    'scores': specialization_scores,
                    'confidence': 0.0,
                    'signals': distinct_signals
                }
            
            primary_spec = sorted_specs[0][0] if sorted_specs[0][1] > 0 else 'general'
            secondary_specs = [spec for spec, score in sorted_specs[1:3] if score > 0]
            
            # Determine if generalist or specialist
            if total_score == 0:
                specialization_type = 'unknown'
            elif max_score / total_score > 0.6:
                specialization_type = 'specialist'
            else:
                specialization_type = 'generalist'
            
            return {
                'primary': primary_spec,
                'secondary': secondary_specs,
                'type': specialization_type,
                'scores': specialization_scores,
                'confidence': confidence,
                'signals': distinct_signals  # Always expose signals used
            }
            
        except Exception as e:
            logger.warning(f"Error identifying specializations: {e}")
            return {
                'primary': None,
                'secondary': [],
                'type': 'unknown',
                'scores': {},
                'confidence': 0.0,
                'signals': 0
            }
    
    def find_rare_qualifications(self, candidate_data: Dict[str, Any]) -> List[str]:
        """
        Find rare qualifications and achievements.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            List of rare qualifications found
        """
        try:
            parsed = candidate_data.get('parsed', {})
            
            # Extract all text
            all_text = []
            
            # Add all sections
            for section in ['experience', 'education', 'projects', 'skills']:
                items = parsed.get(section, [])
                for item in items:
                    if isinstance(item, dict):
                        for value in item.values():
                            if isinstance(value, str):
                                all_text.append(value)
                            elif isinstance(value, list):
                                all_text.extend([str(v) for v in value])
                    elif isinstance(item, str):
                        all_text.append(item)
            
            # Combine and search
            combined_text = ' '.join(all_text).lower()
            
            # Find rare qualifications
            found_qualifications = []
            for pattern in self.rare_qualifications:
                matches = re.findall(pattern, combined_text, re.IGNORECASE)
                if matches:
                    found_qualifications.extend(matches)
            
            # Remove duplicates and return
            return list(set(found_qualifications))
            
        except Exception as e:
            logger.warning(f"Error finding rare qualifications: {e}")
            return []
    
    def detect_leadership_experience(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect leadership experience and team management.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            Dictionary with leadership analysis
        """
        try:
            parsed = candidate_data.get('parsed', {})
            experience_items = parsed.get('experience', [])
            
            leadership_indicators = []
            team_sizes = []
            
            for exp in experience_items:
                title = exp.get('title', '')
                bullets = exp.get('bullets', [])
                
                # Check title for leadership
                if any(re.search(pattern, title, re.IGNORECASE) for pattern in self.leadership_indicators):
                    leadership_indicators.append(title)
                
                # Check bullets for leadership and team size
                for bullet in bullets:
                    # Look for team size indicators
                    team_match = re.search(r'team of (\d+)', bullet, re.IGNORECASE)
                    if team_match:
                        team_sizes.append(int(team_match.group(1)))
                    
                    # Look for leadership indicators
                    if any(re.search(pattern, bullet, re.IGNORECASE) for pattern in self.leadership_indicators):
                        leadership_indicators.append(bullet)
            
            # Analyze leadership level
            if not leadership_indicators:
                leadership_level = 'none'
            elif len(leadership_indicators) >= 3:
                leadership_level = 'extensive'
            elif len(leadership_indicators) >= 1:
                leadership_level = 'some'
            else:
                leadership_level = 'minimal'
            
            # Calculate average team size
            avg_team_size = sum(team_sizes) / len(team_sizes) if team_sizes else 0
            
            return {
                'level': leadership_level,
                'indicators': leadership_indicators,
                'team_sizes': team_sizes,
                'avg_team_size': avg_team_size,
                'has_management': len(team_sizes) > 0
            }
            
        except Exception as e:
            logger.warning(f"Error detecting leadership experience: {e}")
            return {
                'level': 'unknown',
                'indicators': [],
                'team_sizes': [],
                'avg_team_size': 0,
                'has_management': False
            }
    
    def _is_seniority_increase(self, current_title: str, next_title: str) -> bool:
        """Check if there's a seniority increase between titles."""
        seniority_order = ['intern', 'junior', 'mid', 'senior', 'lead', 'staff']
        
        current_level = self._extract_seniority_level(current_title)
        next_level = self._extract_seniority_level(next_title)
        
        if current_level in seniority_order and next_level in seniority_order:
            return seniority_order.index(next_level) > seniority_order.index(current_level)
        
        return False
    
    def _extract_seniority_level(self, title: str) -> Optional[str]:
        """Extract seniority level from title."""
        for level, patterns in self.seniority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    return level
        return None
    
    def _has_leadership_growth(self, current_title: str, next_title: str) -> bool:
        """Check if there's leadership growth between titles."""
        current_leadership = any(re.search(pattern, current_title, re.IGNORECASE) 
                               for pattern in self.leadership_indicators)
        next_leadership = any(re.search(pattern, next_title, re.IGNORECASE) 
                            for pattern in self.leadership_indicators)
        
        return not current_leadership and next_leadership
    
    def analyze_profile_context(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive profile analysis for NLG context.
        
        Args:
            candidate_data: Candidate resume data
            
        Returns:
            Complete profile context for NLG
        """
        try:
            seniority = self.detect_seniority(candidate_data)
            progression = self.assess_career_progression(candidate_data)
            specializations = self.identify_specializations(candidate_data)
            rare_qualifications = self.find_rare_qualifications(candidate_data)
            leadership = self.detect_leadership_experience(candidate_data)
            
            return {
                'seniority': seniority,
                'progression': progression,
                'specializations': specializations,
                'rare_qualifications': rare_qualifications,
                'leadership': leadership,
                'technical_depth': self._assess_technical_depth(candidate_data),
                'career_stage': self._determine_career_stage(seniority, progression)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing profile context: {e}")
            return {
                'seniority': 'mid',
                'progression': {'trajectory': 'unknown'},
                'specializations': {'primary': 'general'},
                'rare_qualifications': [],
                'leadership': {'level': 'none'},
                'technical_depth': 'moderate',
                'career_stage': 'mid'
            }
    
    def _assess_technical_depth(self, candidate_data: Dict[str, Any]) -> str:
        """Assess technical depth from projects and experience."""
        try:
            parsed = candidate_data.get('parsed', {})
            projects = parsed.get('projects', [])
            experience = parsed.get('experience', [])
            
            # Count technical indicators
            technical_indicators = 0
            
            # Check projects for technical complexity
            for project in projects:
                technologies = project.get('technologies', [])
                if len(technologies) >= 3:
                    technical_indicators += 1
                
                summary = project.get('summary', '')
                if any(word in summary.lower() for word in ['architecture', 'scalable', 'distributed', 'microservices']):
                    technical_indicators += 1
            
            # Check experience for technical depth
            for exp in experience:
                bullets = exp.get('bullets', [])
                for bullet in bullets:
                    if any(word in bullet.lower() for word in ['architected', 'designed', 'implemented', 'optimized']):
                        technical_indicators += 1
            
            if technical_indicators >= 4:
                return 'high'
            elif technical_indicators >= 2:
                return 'moderate'
            else:
                return 'basic'
                
        except Exception as e:
            logger.warning(f"Error assessing technical depth: {e}")
            return 'moderate'
    
    def _determine_career_stage(self, seniority: str, progression: Dict[str, Any]) -> str:
        """Determine overall career stage."""
        if seniority in ['intern', 'junior']:
            return 'early'
        elif seniority in ['mid'] and progression.get('trajectory') == 'ascending':
            return 'growing'
        elif seniority in ['senior', 'lead', 'staff']:
            return 'established'
        else:
            return 'mid'
