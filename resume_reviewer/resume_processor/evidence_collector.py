"""
Evidence Collection System for Resume Analysis

This module collects and ranks evidence from multiple sources to support
evidence-based analysis generation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """Represents a single piece of evidence."""
    source: str  # 'ce_pair', 'skill_inference', 'text_match'
    skill: str
    confidence: float  # 0.0 to 1.0
    text: str  # Evidence text or context
    section: Optional[str] = None  # Which resume section
    company: Optional[str] = None  # Company context if available
    role: Optional[str] = None  # Role context if available
    micro_quote: Optional[str] = None  # 4-8 word quote if available


class EvidenceCollector:
    """Collects and ranks evidence from multiple sources."""
    
    def __init__(self):
        self.confidence_threshold = 0.6  # Minimum confidence for inclusion
    
    def collect_evidence(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> List[EvidenceItem]:
        """
        Collect evidence from multiple sources:
        1. CE pairs (query-context matches with scores)
        2. Skill inference (section-specific with confidence)
        3. Direct text matches from parsed sections
        
        Returns ranked evidence items with source, confidence, text
        """
        evidence_pool = []
        
        try:
            # 1. Collect CE evidence (if available)
            ce_evidence = self._collect_ce_evidence(candidate_data, jd_criteria)
            evidence_pool.extend(ce_evidence)
            
            # 2. Collect skill inference evidence
            skill_evidence = self._collect_skill_inference_evidence(candidate_data, jd_criteria)
            evidence_pool.extend(skill_evidence)
            
            # 3. Collect direct text matches
            text_evidence = self._collect_text_match_evidence(candidate_data, jd_criteria)
            evidence_pool.extend(text_evidence)
            
            # Filter by confidence threshold
            filtered_evidence = [e for e in evidence_pool if e.confidence >= self.confidence_threshold]
            
            # Sort by confidence (highest first)
            filtered_evidence.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Collected {len(filtered_evidence)} evidence items above threshold {self.confidence_threshold}")
            return filtered_evidence
            
        except Exception as e:
            logger.error(f"Error collecting evidence: {e}")
            return []
    
    def _collect_ce_evidence(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> List[EvidenceItem]:
        """Collect evidence from cross-encoder pairs."""
        evidence = []
        
        try:
            # Check if CE scores are available in candidate data
            scores = candidate_data.get('scores', {})
            ce_score = scores.get('cross_encoder', 0.0)
            ce_raw_score = scores.get('cross_encoder_raw', 0.0)
            ce_confidence = scores.get('cross_encoder_confidence', 0.0)
            
            # Use CE confidence if available, otherwise use raw score
            confidence_score = ce_confidence if ce_confidence > 0 else ce_raw_score
            
            if confidence_score > 0.5:  # Lower threshold to capture more evidence
                # Extract context from parsed data
                parsed = candidate_data.get('parsed', {})
                experience = parsed.get('experience', [])
                
                if experience:
                    # Use top experience item for context
                    top_exp = experience[0]
                    company = top_exp.get('company', '')
                    role = top_exp.get('title', '')
                    description = top_exp.get('description', '')
                    
                    # Find best matching skill from required skills
                    required_skills = jd_criteria.get('must_have_skills', [])
                    if required_skills:
                        # Try to find the skill that best matches the experience
                        best_skill = self._find_best_matching_skill(required_skills, description, role)
                        
                        # Create evidence text from experience context
                        evidence_text = self._create_evidence_text_from_experience(role, company, description)
                        
                        evidence.append(EvidenceItem(
                            source='ce_pair',
                            skill=best_skill,
                            confidence=min(0.95, confidence_score),
                            text=evidence_text,
                            section='experience',
                            company=company,
                            role=role,
                            micro_quote=self._extract_micro_quote_from_context(evidence_text)
                        ))
                        
                        logger.debug(f"Created CE evidence for {best_skill}: {evidence_text}")
            
        except Exception as e:
            logger.warning(f"Error collecting CE evidence: {e}")
        
        return evidence
    
    def _find_best_matching_skill(self, required_skills: List[str], description: str, role: str) -> str:
        """Find the skill that best matches the experience context."""
        if not required_skills:
            return "relevant experience"
        
        # Simple matching - look for skill mentions in description/role
        context_text = f"{role} {description}".lower()
        
        for skill in required_skills:
            if skill.lower() in context_text:
                return skill
        
        # Return first required skill if no match found
        return required_skills[0]
    
    def _create_evidence_text_from_experience(self, role: str, company: str, description: str) -> str:
        """Create evidence text from experience context."""
        if description:
            # Use description if available
            return f"{role} experience at {company}: {description[:100]}..."
        else:
            # Fallback to role and company
            return f"{role} experience at {company}"
    
    def _collect_skill_inference_evidence(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> List[EvidenceItem]:
        """Collect evidence from skill inference system."""
        evidence = []
        
        try:
            # Import skill inference if available and enabled
            try:
                from .config import INFER_SKILLS
                if not INFER_SKILLS:
                    return evidence  # Skip skill inference if disabled
                
                from .skill_inference import infer_required_skills
                
                parsed = candidate_data.get('parsed', {})
                required_skills = jd_criteria.get('must_have_skills', [])
                
                if required_skills and parsed:
                    skill_details = infer_required_skills(required_skills, parsed)
                    
                    for detail in skill_details:
                        if detail.get('confidence', 0.0) >= self.confidence_threshold:
                            evidence.append(EvidenceItem(
                                source='skill_inference',
                                skill=detail.get('skill', ''),
                                confidence=detail.get('confidence', 0.0),
                                text=f"Found in {detail.get('section', 'unknown')} section",
                                section=detail.get('section'),
                                micro_quote=self._extract_micro_quote(self._format_evidence_text(detail.get('evidence', [])))
                            ))
                            
            except ImportError:
                logger.debug("Skill inference module not available")
                
        except Exception as e:
            logger.warning(f"Error collecting skill inference evidence: {e}")
        
        return evidence
    
    def _collect_text_match_evidence(self, candidate_data: Dict[str, Any], jd_criteria: Dict[str, Any]) -> List[EvidenceItem]:
        """Collect evidence from direct text matches."""
        evidence = []
        
        try:
            parsed = candidate_data.get('parsed', {})
            required_skills = jd_criteria.get('must_have_skills', [])
            
            if not required_skills or not parsed:
                return evidence
            
            # Check each section for skill mentions
            sections = {
                'experience': parsed.get('experience', []),
                'projects': parsed.get('projects', []),
                'skills': parsed.get('skills', [])
            }
            
            for section_name, section_data in sections.items():
                if not section_data:
                    continue
                
                # Convert section data to text
                section_text = self._section_to_text(section_data)
                
                for skill in required_skills:
                    if self._skill_mentioned_in_text(skill, section_text):
                        # Extract context around the skill mention
                        context = self._extract_skill_context(skill, section_text)
                        
                        # Use the extracted context for more meaningful evidence text
                        evidence_text = context if context else f"Mentioned in {section_name}"
                        
                        evidence.append(EvidenceItem(
                            source='text_match',
                            skill=skill,
                            confidence=0.6,  # Lower confidence for text matches
                            text=evidence_text,
                            section=section_name,
                            micro_quote=self._extract_micro_quote_from_context(context)
                        ))
                        
        except Exception as e:
            logger.warning(f"Error collecting text match evidence: {e}")
        
        return evidence
    
    def get_best_evidence_for_skill(self, skill: str, evidence_pool: List[EvidenceItem]) -> Optional[EvidenceItem]:
        """Get single best evidence item for a skill."""
        skill_evidence = [e for e in evidence_pool if e.skill.lower() == skill.lower()]
        
        if not skill_evidence:
            return None
        
        # Sort by confidence and return best
        skill_evidence.sort(key=lambda x: x.confidence, reverse=True)
        return skill_evidence[0]
    
    def assess_data_quality(self, candidate_data: Dict[str, Any], evidence_pool: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Assess overall data quality:
        - Parsing completeness (has experience/skills sections)
        - CE evidence count and average confidence
        - Skill inference confidence levels
        
        Returns quality assessment
        """
        try:
            parsed = candidate_data.get('parsed', {})
            
            # Check parsing completeness
            has_experience = bool(parsed.get('experience'))
            has_skills = bool(parsed.get('skills'))
            has_projects = bool(parsed.get('projects'))
            
            # Count evidence by source
            ce_evidence = [e for e in evidence_pool if e.source == 'ce_pair']
            skill_evidence = [e for e in evidence_pool if e.source == 'skill_inference']
            text_evidence = [e for e in evidence_pool if e.source == 'text_match']
            
            # Check CE data availability from scores and actual CE pairs
            scores = candidate_data.get('scores', {})
            ce_score = scores.get('cross_encoder', 0.0)
            ce_raw_score = scores.get('cross_encoder_raw', 0.0)
            ce_confidence = scores.get('cross_encoder_confidence', 0.0)
            ce_pairs = scores.get('ce_pairs', [])
            has_ce_data = (ce_score > 0 or ce_raw_score > 0 or ce_confidence > 0) and len(ce_pairs) > 0
            
            # Calculate average confidence
            avg_confidence = sum(e.confidence for e in evidence_pool) / len(evidence_pool) if evidence_pool else 0.0
            
            # Determine quality level
            quality_score = 0
            if has_experience:
                quality_score += 1
            if has_skills:
                quality_score += 1
            if has_projects:
                quality_score += 1
            if len(ce_evidence) > 0:
                quality_score += 1
            if avg_confidence > 0.7:
                quality_score += 1
            if has_ce_data:
                quality_score += 1
            
            if quality_score >= 4:
                quality_level = 'high'
            elif quality_score >= 2:
                quality_level = 'medium'
            else:
                quality_level = 'low'
            
            return {
                'quality_level': quality_level,
                'quality_score': quality_score,
                'parsing_completeness': {
                    'has_experience': has_experience,
                    'has_skills': has_skills,
                    'has_projects': has_projects
                },
                'evidence_counts': {
                    'ce_pairs': len(ce_evidence),
                    'skill_inference': len(skill_evidence),
                    'text_matches': len(text_evidence),
                    'total': len(evidence_pool)
                },
                'average_confidence': avg_confidence,
                'needs_caveat': quality_level in ['low', 'medium'] or not has_ce_data,
                'ce_data_availability': {
                    'has_ce_data': has_ce_data,
                    'ce_score': ce_score,
                    'ce_raw_score': ce_raw_score,
                    'ce_confidence': ce_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {
                'quality_level': 'low',
                'quality_score': 0,
                'parsing_completeness': {'has_experience': False, 'has_skills': False, 'has_projects': False},
                'evidence_counts': {'ce_pairs': 0, 'skill_inference': 0, 'text_matches': 0, 'total': 0},
                'average_confidence': 0.0,
                'needs_caveat': True
            }
    
    def select_best_concrete_example(self, evidence_pool: List[EvidenceItem], jd_criteria: Dict[str, Any]) -> Optional[EvidenceItem]:
        """
        Select single best example that:
        1. Ties to a critical JD requirement
        2. Has highest confidence score
        3. Has rich context (role + outcome)
        4. Optionally has extractable micro-quote
        
        Returns best evidence item for concrete example
        """
        if not evidence_pool:
            return None
        
        # Filter for critical skills (required skills)
        required_skills = set(jd_criteria.get('must_have_skills', []))
        critical_evidence = [e for e in evidence_pool if e.skill in required_skills]
        
        if not critical_evidence:
            # Fall back to any evidence if no critical skills found
            critical_evidence = evidence_pool
        
        # Sort by confidence, then by source priority (CE > skill_inference > text_match)
        source_priority = {'ce_pair': 3, 'skill_inference': 2, 'text_match': 1}
        
        critical_evidence.sort(key=lambda x: (x.confidence, source_priority.get(x.source, 0)), reverse=True)
        
        return critical_evidence[0]
    
    def calculate_statement_confidence(self, evidence_item: EvidenceItem) -> float:
        """
        Calculate confidence for making a claim:
        - CE pair score > 0.7: high (0.85+)
        - Skill inference > 0.6: medium (0.6-0.85)
        - Text match only: low (< 0.6)
        
        Only include claims with confidence >= 0.6
        """
        base_confidence = evidence_item.confidence
        
        # Adjust based on source
        if evidence_item.source == 'ce_pair' and base_confidence > 0.7:
            return min(0.95, base_confidence + 0.1)
        elif evidence_item.source == 'skill_inference' and base_confidence > 0.6:
            return base_confidence
        elif evidence_item.source == 'text_match':
            return max(0.0, base_confidence - 0.1)
        
        return base_confidence
    
    # Helper methods
    
    def _section_to_text(self, section_data: List[Any]) -> str:
        """Convert section data to searchable text."""
        if not section_data:
            return ""
        
        text_parts = []
        for item in section_data:
            if isinstance(item, dict):
                # Extract relevant text fields
                for field in ['title', 'company', 'description', 'bullets', 'summary']:
                    value = item.get(field, '')
                    if value:
                        if isinstance(value, list):
                            text_parts.extend(str(v) for v in value)
                        else:
                            text_parts.append(str(value))
            else:
                text_parts.append(str(item))
        
        return ' '.join(text_parts)
    
    def _skill_mentioned_in_text(self, skill: str, text: str) -> bool:
        """Check if skill is mentioned in text (case-insensitive)."""
        if not skill or not text:
            return False
        
        # Simple substring match (can be enhanced with better NLP)
        return skill.lower() in text.lower()
    
    def _extract_skill_context(self, skill: str, text: str, context_chars: int = 80) -> str:
        """Extract meaningful context around skill mention."""
        if not skill or not text:
            return ""
        
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        index = text_lower.find(skill_lower)
        if index == -1:
            return ""
        
        # Try to extract a complete sentence or phrase
        # Find sentence boundaries
        start = max(0, index - context_chars)
        end = min(len(text), index + len(skill) + context_chars)
        
        # Adjust to sentence boundaries if possible
        sentence_start = text.rfind('.', start, index)
        if sentence_start > start - 20:  # If we're close to a sentence start
            start = sentence_start + 1
        
        sentence_end = text.find('.', index + len(skill), end)
        if sentence_end > 0 and sentence_end < end + 20:  # If we're close to a sentence end
            end = sentence_end + 1
        
        context = text[start:end].strip()
        
        # Clean up the context to be more concise and natural
        context = context.strip('.,;:')
        
        # Remove common prefixes
        prefixes_to_remove = ['and ', 'with ', 'using ', 'via ', 'through ']
        for prefix in prefixes_to_remove:
            if context.lower().startswith(prefix):
                context = context[len(prefix):]
        
        # Remove common suffixes
        suffixes_to_remove = [' and', ' with', ' using', ' via', ' through']
        for suffix in suffixes_to_remove:
            if context.lower().endswith(suffix):
                context = context[:-len(suffix)]
        
        # If context is too long, try to extract just the key phrase around the skill
        if len(context) > 80:
            # Try to find a shorter, more relevant phrase around the skill mention
            words = context.split()
            skill_words = skill.split()
            
            # Find the skill in the context
            skill_start = -1
            for i in range(len(words) - len(skill_words) + 1):
                if words[i:i+len(skill_words)] == skill_words:
                    skill_start = i
                    break
            
            if skill_start >= 0:
                # Extract 3-5 words before and after the skill
                start_idx = max(0, skill_start - 3)
                end_idx = min(len(words), skill_start + len(skill_words) + 3)
                context = ' '.join(words[start_idx:end_idx])
            else:
                # Fallback: take the middle portion
                if len(words) > 12:
                    start_idx = max(0, len(words) // 2 - 4)
                    end_idx = min(len(words), start_idx + 8)
                    context = ' '.join(words[start_idx:end_idx])
        
        return context
    
    def _extract_micro_quote(self, evidence_list: List[str]) -> Optional[str]:
        """Extract 4-8 word micro-quote from evidence."""
        if not evidence_list:
            return None
        
        # Take first evidence item and extract short quote
        evidence_text = evidence_list[0] if evidence_list else ""
        words = evidence_text.split()
        
        if len(words) <= 8:
            return evidence_text
        else:
            # Take first 6 words
            return ' '.join(words[:6])
    
    def _extract_micro_quote_from_context(self, context: str) -> Optional[str]:
        """Extract micro-quote from context text."""
        if not context:
            return None
        
        words = context.split()
        if len(words) <= 8:
            return context
        else:
            # Take middle portion if context is long
            start = len(words) // 2 - 3
            end = start + 6
            return ' '.join(words[max(0, start):min(len(words), end)])
    
    def _format_evidence_text(self, evidence: Any) -> str:
        """
        Format evidence text from various data types.
        """
        if isinstance(evidence, str):
            return evidence
        elif isinstance(evidence, list):
            # Handle list of strings or dicts
            formatted_parts = []
            for item in evidence:
                if isinstance(item, str):
                    formatted_parts.append(item)
                elif isinstance(item, dict):
                    # Extract text from dict
                    text = item.get('text', '') or item.get('description', '') or str(item)
                    if text:
                        formatted_parts.append(text)
            return ' '.join(formatted_parts)
        elif isinstance(evidence, dict):
            # Extract text from dict
            return evidence.get('text', '') or evidence.get('description', '') or str(evidence)
        else:
            return str(evidence) if evidence else ''
