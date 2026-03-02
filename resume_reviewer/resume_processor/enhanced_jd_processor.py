"""
Enhanced Job Description Processor

This module handles the preprocessing of job descriptions that may contain
natural language sentences instead of clean skill tokens. It extracts
canonical skill tokens, creates JD summaries, and enforces strict schema validation.
"""

import logging
import re
from typing import Any, Dict, List, Set, Tuple

from .text_processor import SkillTaxonomy

logger = logging.getLogger(__name__)


class EnhancedJDProcessor:
    """Enhanced JD processor that handles natural language inputs and extracts clean skill tokens."""
    
    def __init__(self):
        self.skill_taxonomy = SkillTaxonomy()
        
        # Canonical skill tokens that we expect to extract
        self.canonical_skills = {
            'css', 'git', 'html', 'java', 'javascript', 'linux', 'nodejs', 'python', 'react'
        }
        
        # Additional skills from taxonomy
        self.canonical_skills.update(self.skill_taxonomy.skill_mapping.keys())
        
        # Logging configuration
        self.debug_logging = True
    
    def process_jd_criteria(self, raw_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw JD criteria and extract clean skill tokens from natural language.
        
        Args:
            raw_criteria: Raw JD criteria that may contain sentences in skill arrays
            
        Returns:
            Processed JD criteria with clean skill tokens and JD summary
        """
        try:
            logger.info("Starting enhanced JD processing")
            
            # Create a copy to avoid modifying the original
            processed_criteria = raw_criteria.copy()
            
            # Process must-have skills
            must_have_result = self._process_skill_field(
                raw_criteria.get('must_have_skills', []),
                'must_have_skills'
            )
            processed_criteria['must_have_skills'] = must_have_result['clean_skills']
            processed_criteria['must_have_skills_debug'] = must_have_result
            
            # Process nice-to-have skills
            nice_to_have_result = self._process_skill_field(
                raw_criteria.get('nice_to_have_skills', []),
                'nice_to_have_skills'
            )
            processed_criteria['nice_to_have_skills'] = nice_to_have_result['clean_skills']
            processed_criteria['nice_to_have_skills_debug'] = nice_to_have_result
            
            # Process industry experience
            industry_result = self._process_skill_field(
                raw_criteria.get('industry_experience', []),
                'industry_experience'
            )
            processed_criteria['industry_experience'] = industry_result['clean_skills']
            processed_criteria['industry_experience_debug'] = industry_result
            
            # Create JD summary from all text content
            jd_summary = self._create_jd_summary(raw_criteria)
            processed_criteria['jd_summary'] = jd_summary
            
            # Validate schema
            self._validate_schema(processed_criteria)
            
            logger.info(f"JD processing completed. Extracted {len(processed_criteria['must_have_skills'])} must-have skills")
            
            return processed_criteria
            
        except Exception as e:
            logger.error(f"Error processing JD criteria: {e}")
            # Return original criteria with empty skills as fallback
            fallback_criteria = raw_criteria.copy()
            fallback_criteria['must_have_skills'] = []
            fallback_criteria['nice_to_have_skills'] = []
            fallback_criteria['industry_experience'] = []
            fallback_criteria['jd_summary'] = self._create_jd_summary(raw_criteria)
            return fallback_criteria
    
    def _process_skill_field(self, raw_skills: List[str], field_name: str) -> Dict[str, Any]:
        """
        Process a skill field that may contain sentences or clean tokens.
        
        IMPORTANT: User-provided skills are PRESERVED as-is. We only attempt extraction
        from inputs that look like full natural language sentences (with verbs, etc.).
        Short skill phrases like "system monitoring" or "project management" are kept intact.
        
        Args:
            raw_skills: List of skills that may be sentences or tokens
            field_name: Name of the field for logging
            
        Returns:
            Dictionary with clean skills and debug information
        """
        if not isinstance(raw_skills, list):
            return {
                'clean_skills': [],
                'skills_before': [],
                'skills_after': [],
                'rejected_items': [],
                'extracted_from_sentences': []
            }
        
        clean_skills = []
        rejected_items = []
        extracted_from_sentences = []
        
        logger.info(f"Processing {field_name}: {len(raw_skills)} items")
        
        for item in raw_skills:
            if not isinstance(item, str):
                rejected_items.append(f"Non-string item: {item}")
                continue
            
            item = item.strip()
            if not item:
                continue
            
            # PRIORITY: Preserve user input as-is unless it's clearly a natural language sentence
            # A "sentence" must have: >5 words AND contain verb-like patterns
            if self._is_natural_language_sentence(item):
                # Only extract skills from actual sentences (not short skill phrases)
                extracted_skills = self._extract_skills_from_sentence(item)
                if extracted_skills:
                    clean_skills.extend(extracted_skills)
                    extracted_from_sentences.append({
                        'sentence': item,
                        'extracted_skills': extracted_skills
                    })
                    logger.debug(f"Extracted from sentence: '{item}' -> {extracted_skills}")
                else:
                    # Even if extraction fails, keep the original if it's short
                    if len(item.split()) <= 4:
                        normalized = self.skill_taxonomy.normalize_skill(item)
                        clean_skills.append(normalized)
                        logger.debug(f"Preserved user input (extraction failed): '{item}' -> '{normalized}'")
                    else:
                        rejected_items.append(f"No skills found in: '{item}'")
            else:
                # Short phrases and single words: PRESERVE as user input
                # Normalize but DO NOT reject based on canonical list
                normalized = self.skill_taxonomy.normalize_skill(item)
                clean_skills.append(normalized)
                logger.debug(f"Preserved user skill: '{item}' -> '{normalized}'")
        
        # Remove duplicates while preserving order
        clean_skills = list(dict.fromkeys(clean_skills))
        
        result = {
            'clean_skills': clean_skills,
            'skills_before': raw_skills,
            'skills_after': clean_skills,
            'rejected_items': rejected_items,
            'extracted_from_sentences': extracted_from_sentences
        }
        
        if self.debug_logging:
            logger.info(f"{field_name} processing results:")
            logger.info(f"  Clean skills: {clean_skills}")
            logger.info(f"  Rejected items: {rejected_items}")
            logger.info(f"  Extracted from sentences: {len(extracted_from_sentences)}")
        
        return result
    
    def _is_natural_language_sentence(self, text: str) -> bool:
        """
        Determine if text is a natural language sentence vs. a skill phrase.
        
        Skill phrases to PRESERVE: "system monitoring", "project management", "React.js"
        Sentences to EXTRACT from: "Must have experience with Python and React"
        
        Returns True only for actual sentences, not short skill phrases.
        """
        if not text:
            return False
        
        words = text.split()
        
        # Short phrases (4 words or less) are skill phrases, not sentences
        if len(words) <= 4:
            return False
        
        text_lower = text.lower()
        
        # Must contain sentence-like patterns (verbs, requirements language)
        sentence_indicators = [
            'must have', 'should have', 'need to', 'able to',
            'experience with', 'experience in', 'knowledge of',
            'proficiency in', 'familiar with', 'working with',
            'required', 'preferred', 'understanding of',
            'we are', 'looking for', 'responsible for',
            'will be', 'you will', 'the candidate'
        ]
        
        has_sentence_pattern = any(indicator in text_lower for indicator in sentence_indicators)
        
        # Also check for sentence-ending punctuation with sufficient length
        has_sentence_structure = len(words) > 6 and text.rstrip()[-1] in '.!?'
        
        return has_sentence_pattern or has_sentence_structure
    
    def _is_clean_skill_token(self, item: str) -> bool:
        """
        Check if an item is a clean skill token (no spaces, punctuation, or periods).
        
        Args:
            item: String to check
            
        Returns:
            True if item is a clean skill token
        """
        # Clean tokens should not contain:
        # - Spaces (except for multi-word skills like "sql server")
        # - Punctuation (except for special cases like "c++", "c#")
        # - Periods
        # - Multiple words (except known multi-word skills)
        
        if not item or len(item) > 50:  # Too long to be a skill token
            return False
        
        # Check for periods (definitely not a clean token)
        if '.' in item:
            return False
        
        # Check for multiple spaces (likely a sentence)
        if '  ' in item or item.count(' ') > 2:
            return False
        
        # Check for sentence-like patterns
        sentence_indicators = [
            'we are', 'we\'re', 'looking for', 'should be', 'must have',
            'experience with', 'knowledge of', 'proficiency in', 'familiar with',
            'comfortable with', 'working with', 'using', 'for', 'and', 'or'
        ]
        
        item_lower = item.lower()
        for indicator in sentence_indicators:
            if indicator in item_lower:
                return False
        
        # Check if it's a known multi-word skill
        known_multi_word_skills = {
            'sql server', 'asp.net', 'c++', 'c#', 'node.js', 'react native',
            'spring boot', 'ruby on rails', 'chart.js', 'ci/cd', 'chart.js'
        }
        
        if item_lower in known_multi_word_skills:
            return True
        
        # Single word or simple phrases are likely clean tokens
        if len(item.split()) <= 2 and not any(p in item for p in ['.', '!', '?', ',', ';', ':']):
            return True
        
        return False
    
    def _extract_skills_from_sentence(self, sentence: str) -> List[str]:
        """
        Extract canonical skill tokens from a sentence.
        
        Args:
            sentence: Sentence that may contain skill mentions
            
        Returns:
            List of canonical skill tokens found in the sentence
        """
        if not sentence:
            return []
        
        # Use the skill taxonomy to extract skills
        extracted_skills = self.skill_taxonomy.extract_skills_from_text(sentence)
        
        # Filter to only include canonical skills
        canonical_skills = []
        for skill in extracted_skills:
            if skill in self.canonical_skills:
                canonical_skills.append(skill)
        
        return canonical_skills
    
    def _create_jd_summary(self, criteria: Dict[str, Any]) -> str:
        """
        Create a JD summary from all text content for CE and semantic matching.
        
        Args:
            criteria: JD criteria dictionary
            
        Returns:
            Natural language JD summary
        """
        try:
            summary_parts = []
            
            # Job title and seniority
            title = criteria.get('position_title', '').strip()
            seniority = criteria.get('seniority_level', '').strip()
            if title:
                if seniority:
                    summary_parts.append(f"We are seeking a {seniority} {title}.")
                else:
                    summary_parts.append(f"We are seeking a {title}.")
            
            # Experience requirements
            exp_years = criteria.get('experience_min_years', 0)
            if exp_years > 0:
                summary_parts.append(f"The role requires {exp_years}+ years of experience.")
            
            # Education
            education = criteria.get('education_requirements', '').strip()
            if education:
                summary_parts.append(f"Education requirements: {education}")
            
            # Location
            location = criteria.get('location', '').strip()
            if location:
                summary_parts.append(f"Location: {location}")
            
            # Collect all text content for skills and responsibilities
            all_text_content = []
            
            # Add original skill content (sentences)
            must_have = criteria.get('must_have_skills', [])
            if must_have:
                all_text_content.extend([str(item) for item in must_have])
            
            nice_to_have = criteria.get('nice_to_have_skills', [])
            if nice_to_have:
                all_text_content.extend([str(item) for item in nice_to_have])
            
            industry = criteria.get('industry_experience', [])
            if industry:
                all_text_content.extend([str(item) for item in industry])
            
            # Add responsibilities (preserve as separate sentences)
            responsibilities = criteria.get('responsibilities', [])
            if responsibilities:
                # Responsibilities are descriptive text - join them properly as sentences
                responsibilities_text = '. '.join(str(item).strip() for item in responsibilities if str(item).strip())
                # Ensure each responsibility ends with a period
                if responsibilities_text and not responsibilities_text.endswith('.'):
                    responsibilities_text += '.'
                if responsibilities_text:
                    summary_parts.append(f"Key responsibilities: {responsibilities_text}")
            
            # Add keywords
            keywords = criteria.get('keywords', [])
            if keywords:
                all_text_content.extend([str(item) for item in keywords])
            
            # Combine remaining text content (skills, keywords)
            if all_text_content:
                combined_text = ' '.join(all_text_content)
                # Clean up the text
                combined_text = re.sub(r'\s+', ' ', combined_text).strip()
                summary_parts.append(combined_text)
            
            # Join all parts
            jd_summary = ' '.join(summary_parts)
            
            # Clean up
            jd_summary = re.sub(r'\s+', ' ', jd_summary).strip()
            
            return jd_summary
            
        except Exception as e:
            logger.error(f"Error creating JD summary: {e}")
            return str(criteria.get('position_title', ''))
    
    def _validate_schema(self, criteria: Dict[str, Any]) -> None:
        """
        Validate that the processed criteria conforms to the expected schema.
        
        Args:
            criteria: Processed JD criteria to validate
        """
        try:
            # Check that skills arrays contain only clean tokens
            for field in ['must_have_skills', 'nice_to_have_skills', 'industry_experience']:
                skills = criteria.get(field, [])
                if not isinstance(skills, list):
                    logger.warning(f"{field} is not a list: {type(skills)}")
                    continue
                
                for skill in skills:
                    if not isinstance(skill, str):
                        logger.warning(f"Non-string skill in {field}: {skill}")
                    elif not self._is_clean_skill_token(skill):
                        logger.warning(f"Non-clean skill token in {field}: '{skill}'")
                    elif skill not in self.canonical_skills:
                        logger.warning(f"Unknown skill token in {field}: '{skill}'")
            
            # Check that jd_summary exists
            if 'jd_summary' not in criteria:
                logger.warning("jd_summary field missing from processed criteria")
            
            logger.info("Schema validation completed")
            
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
    
    def get_processing_stats(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about the JD processing.
        
        Args:
            criteria: Processed JD criteria
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_must_have_skills': len(criteria.get('must_have_skills', [])),
            'total_nice_to_have_skills': len(criteria.get('nice_to_have_skills', [])),
            'total_industry_experience': len(criteria.get('industry_experience', [])),
            'jd_summary_length': len(criteria.get('jd_summary', '')),
            'has_debug_info': 'must_have_skills_debug' in criteria
        }
        
        # Add debug statistics if available
        if 'must_have_skills_debug' in criteria:
            debug = criteria['must_have_skills_debug']
            stats['must_have_rejected_items'] = len(debug.get('rejected_items', []))
            stats['must_have_extracted_from_sentences'] = len(debug.get('extracted_from_sentences', []))
        
        if 'nice_to_have_skills_debug' in criteria:
            debug = criteria['nice_to_have_skills_debug']
            stats['nice_to_have_rejected_items'] = len(debug.get('rejected_items', []))
            stats['nice_to_have_extracted_from_sentences'] = len(debug.get('extracted_from_sentences', []))
        
        return stats


def process_jd_criteria_enhanced(raw_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to process JD criteria with enhanced preprocessing.
    
    Args:
        raw_criteria: Raw JD criteria that may contain sentences in skill arrays
        
    Returns:
        Processed JD criteria with clean skill tokens and JD summary
    """
    processor = EnhancedJDProcessor()
    return processor.process_jd_criteria(raw_criteria)
