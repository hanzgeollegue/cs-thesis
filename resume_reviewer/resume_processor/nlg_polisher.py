"""
Grammar polisher for NLG text post-processing.
Provides lightweight rule-based grammar fixes and style improvements.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from .nlg_config import get_domain_config, DomainConfig

logger = logging.getLogger(__name__)


class GrammarPolisher:
    """Lightweight grammar and style polisher for NLG text."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.polish_log = []
        
        # Grammar rules
        self.singular_plural_rules = [
            (r'\b1 (\w+)s\b', r'1 \1'),  # "1 skills" -> "1 skill"
            (r'\b0 (\w+)s\b', r'0 \1'),  # "0 skills" -> "0 skill"
            (r'\b(\d+) skill\b', r'\1 skills'),  # "2 skill" -> "2 skills"
            (r'\b(\d+) year\b', r'\1 years'),  # "2 year" -> "2 years"
            (r'\b(\d+) role\b', r'\1 roles'),  # "2 role" -> "2 roles"
            (r'\b(\d+) project\b', r'\1 projects'),  # "2 project" -> "2 projects"
        ]
        
        # Tense consistency patterns
        self.tense_patterns = {
            'present': [
                r'\bdemonstrates?\b', r'\bshows?\b', r'\bexhibits?\b', r'\bdisplays?\b',
                r'\bpossesses?\b', r'\bhas\b', r'\bhave\b', r'\bis\b', r'\bare\b'
            ],
            'past': [
                r'\bdemonstrated\b', r'\bshowed\b', r'\bexhibited\b', r'\bdisplayed\b',
                r'\bpossessed\b', r'\bhad\b', r'\bwas\b', r'\bwere\b'
            ]
        }
        
        # Redundancy patterns
        self.redundancy_patterns = [
            (r'\b(\w+)\s+\1\b', r'\1'),  # "skill skill" -> "skill"
            (r'\bthe the\b', 'the'),  # "the the" -> "the"
            (r'\band and\b', 'and'),  # "and and" -> "and"
            (r'\bwith with\b', 'with'),  # "with with" -> "with"
            (r'\bin in\b', 'in'),  # "in in" -> "in"
        ]
        
        # Transition connectors
        self.transition_connectors = [
            'Additionally', 'However', 'Furthermore', 'Moreover', 'In contrast',
            'Specifically', 'Notably', 'Importantly', 'Significantly', 'Particularly'
        ]
    
    def apply_polish(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply all grammar and style polish to text.
        
        Args:
            text: Input text to polish
            
        Returns:
            Tuple of (polished_text, polish_log)
        """
        self.polish_log = []
        polished_text = text
        
        try:
            # Apply fixes in order
            polished_text = self.fix_singular_plural(polished_text)
            polished_text = self.fix_verb_tense(polished_text)
            polished_text = self.remove_redundancy(polished_text)
            polished_text = self.add_transitions(polished_text)
            polished_text = self.fix_sentence_structure(polished_text)
            polished_text = self.clean_punctuation(polished_text)
            
            return polished_text, self.polish_log
            
        except Exception as e:
            logger.warning(f"Error applying polish: {e}")
            return text, [f"Polish error: {str(e)}"]
    
    def fix_singular_plural(self, text: str) -> str:
        """Fix singular/plural mismatches."""
        original_text = text
        
        for pattern, replacement in self.singular_plural_rules:
            new_text = re.sub(pattern, replacement, text)
            if new_text != text:
                self.polish_log.append(f"Fixed singular/plural: {pattern}")
                text = new_text
        
        if text != original_text:
            self.polish_log.append("Applied singular/plural fixes")
        
        return text
    
    def fix_verb_tense(self, text: str) -> str:
        """Ensure verb tense consistency."""
        original_text = text
        
        # Detect dominant tense
        present_count = 0
        past_count = 0
        
        for pattern in self.tense_patterns['present']:
            present_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        for pattern in self.tense_patterns['past']:
            past_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # If mixed tense, prefer present tense for consistency
        if present_count > 0 and past_count > 0:
            # Convert past tense to present where appropriate
            tense_conversions = [
                (r'\bdemonstrated\b', 'demonstrates'),
                (r'\bshowed\b', 'shows'),
                (r'\bexhibited\b', 'exhibits'),
                (r'\bdisplayed\b', 'displays'),
                (r'\bpossessed\b', 'possesses'),
                (r'\bhad\b', 'has'),
                (r'\bwas\b', 'is'),
                (r'\bwere\b', 'are')
            ]
            
            for pattern, replacement in tense_conversions:
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if new_text != text:
                    self.polish_log.append(f"Fixed tense: {pattern} -> {replacement}")
                    text = new_text
        
        if text != original_text:
            self.polish_log.append("Applied tense consistency fixes")
        
        return text
    
    def remove_redundancy(self, text: str) -> str:
        """Remove redundant phrases and words."""
        original_text = text
        
        for pattern, replacement in self.redundancy_patterns:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                self.polish_log.append(f"Removed redundancy: {pattern}")
                text = new_text
        
        # Remove duplicate sentences (within 2-sentence window)
        sentences = text.split('. ')
        cleaned_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Check if this sentence is too similar to recent ones
            is_duplicate = False
            for j in range(max(0, i-2), i):
                if j < len(cleaned_sentences):
                    similarity = self._calculate_similarity(sentence, cleaned_sentences[j])
                    if similarity > 0.8:  # 80% similarity threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                cleaned_sentences.append(sentence)
            else:
                self.polish_log.append(f"Removed duplicate sentence: {sentence[:50]}...")
        
        if len(cleaned_sentences) < len(sentences):
            text = '. '.join(cleaned_sentences)
            self.polish_log.append("Removed duplicate sentences")
        
        if text != original_text:
            self.polish_log.append("Applied redundancy removal")
        
        return text
    
    def add_transitions(self, text: str) -> str:
        """Add transition phrases between sentences where appropriate."""
        original_text = text
        
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return text
        
        # Check if transitions are needed
        transition_count = 0
        for connector in self.transition_connectors:
            transition_count += len(re.findall(connector, text, re.IGNORECASE))
        
        # Add transitions if needed (target: 1 transition per 3 sentences)
        target_transitions = len(sentences) // 3
        if transition_count < target_transitions:
            # Add transitions to appropriate sentences
            enhanced_sentences = []
            transition_index = 0
            
            for i, sentence in enumerate(sentences):
                if i > 0 and i % 3 == 0 and transition_index < len(self.transition_connectors):
                    # Add transition, but avoid awkward combinations
                    connector = self.transition_connectors[transition_index]
                    
                    # Avoid "Additionally, Absent" or similar awkward combinations
                    if sentence.lower().startswith(('absent', 'missing', 'lacks', 'no ')):
                        # Use a different connector for gap statements
                        if 'Additionally' in connector:
                            connector = 'However,'
                        elif 'Furthermore' in connector:
                            connector = 'Also,'
                    
                    enhanced_sentences.append(f"{connector}, {sentence}")
                    transition_index += 1
                    self.polish_log.append(f"Added transition: {connector}")
                else:
                    enhanced_sentences.append(sentence)
            
            text = '. '.join(enhanced_sentences)
        
        if text != original_text:
            self.polish_log.append("Applied transition improvements")
        
        return text
    
    def fix_sentence_structure(self, text: str) -> str:
        """Fix basic sentence structure issues."""
        original_text = text
        
        # Fix common sentence structure issues
        structure_fixes = [
            (r'\s+([,.!?])', r'\1'),  # Remove spaces before punctuation
            (r'([,.!?])([A-Z])', r'\1 \2'),  # Add space after punctuation
            (r'\s+', ' '),  # Normalize whitespace
            (r'^[a-z]', lambda m: m.group(0).upper()),  # Capitalize first letter
        ]
        
        for pattern, replacement in structure_fixes:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        # Ensure sentences end with proper punctuation
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        if text != original_text:
            self.polish_log.append("Applied sentence structure fixes")
        
        return text
    
    def clean_punctuation(self, text: str) -> str:
        """Clean up punctuation issues."""
        original_text = text
        
        # Fix punctuation issues
        punctuation_fixes = [
            (r'\.\.+', '.'),  # Multiple periods -> single period
            (r',,', ','),  # Double commas -> single comma
            (r'\s*,\s*', ', '),  # Normalize comma spacing
            (r'(?<!\d)\s*\.\s*(?!\d)', '. '),  # Normalize period spacing (but not in decimal numbers)
            (r'\s*;\s*', '; '),  # Normalize semicolon spacing
        ]
        
        for pattern, replacement in punctuation_fixes:
            text = re.sub(pattern, replacement, text)
        
        if text != original_text:
            self.polish_log.append("Applied punctuation cleanup")
        
        return text
    
    def _calculate_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate similarity between two sentences (simple word overlap)."""
        words1 = set(sentence1.lower().split())
        words2 = set(sentence2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def validate_readability(self, text: str) -> Dict[str, Any]:
        """Validate text readability and style."""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Check sentence length
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            
            # Check for overly long or short sentences
            min_words = self.config.grammar_rules.sentence_length['min_words']
            max_words = self.config.grammar_rules.sentence_length['max_words']
            
            long_sentences = [i for i, length in enumerate(sentence_lengths) if length > max_words]
            short_sentences = [i for i, length in enumerate(sentence_lengths) if length < min_words]
            
            # Check vocabulary diversity
            all_words = []
            for sentence in sentences:
                all_words.extend(sentence.lower().split())
            
            unique_words = set(all_words)
            vocabulary_diversity = len(unique_words) / len(all_words) if all_words else 0
            
            # Check transition presence
            transition_count = 0
            for connector in self.transition_connectors:
                transition_count += len(re.findall(connector, text, re.IGNORECASE))
            
            transition_frequency = transition_count / len(sentences) if sentences else 0
            required_frequency = self.config.grammar_rules.transitions['required_frequency']
            
            return {
                'avg_sentence_length': avg_length,
                'long_sentences': long_sentences,
                'short_sentences': short_sentences,
                'vocabulary_diversity': vocabulary_diversity,
                'transition_frequency': transition_frequency,
                'meets_length_requirements': not long_sentences and not short_sentences,
                'meets_transition_requirements': transition_frequency >= required_frequency,
                'overall_quality': 'good' if (not long_sentences and not short_sentences and 
                                            transition_frequency >= required_frequency) else 'needs_improvement'
            }
            
        except Exception as e:
            logger.warning(f"Error validating readability: {e}")
            return {
                'avg_sentence_length': 0,
                'long_sentences': [],
                'short_sentences': [],
                'vocabulary_diversity': 0,
                'transition_frequency': 0,
                'meets_length_requirements': False,
                'meets_transition_requirements': False,
                'overall_quality': 'error'
            }
