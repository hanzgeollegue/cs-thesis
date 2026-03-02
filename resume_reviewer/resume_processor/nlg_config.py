"""
Configuration loader for NLG domain rules.
Provides type-safe access to business rules and thresholds.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoreTierConfig:
    """Configuration for a score tier."""
    threshold: float
    tone: str
    focus: str
    gaps: str
    adjectives: List[str]
    reinforcing_words: List[str]


@dataclass
class SeniorityConfig:
    """Configuration for a seniority level."""
    experience_range: List[int]
    tone_adjustment: str
    emphasis: List[str]
    leadership_keywords: List[str]


@dataclass
class ContextWeights:
    """Weights for different role contexts."""
    skills_weight: float
    experience_weight: float
    projects_weight: float
    education_weight: float


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different levels."""
    high_confidence: float
    medium_confidence: float
    low_confidence: float


@dataclass
class TemplateVariants:
    """Number of variants for each template type."""
    opening_sentences: int
    skills_phrases: int
    experience_phrases: int
    gap_analysis: int
    transitions: int


@dataclass
class SpecialAchievements:
    """Thresholds for special achievements."""
    leadership_threshold: int
    rare_skill_threshold: float
    career_velocity_threshold: int
    certification_keywords: List[str]


@dataclass
class GrammarRules:
    """Grammar and style rules."""
    sentence_length: Dict[str, int]
    transitions: Dict[str, Any]


@dataclass
class DomainConfig:
    """Complete domain configuration."""
    version: str
    score_tiers: Dict[str, ScoreTierConfig]
    seniority_levels: Dict[str, SeniorityConfig]
    context_weights: Dict[str, ContextWeights]
    confidence_thresholds: ConfidenceThresholds
    template_variants: TemplateVariants
    special_achievements: SpecialAchievements
    grammar_rules: GrammarRules


class DomainConfigLoader:
    """Loads and validates domain configuration from JSON."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with config file path."""
        if config_path is None:
            # Default to same directory as this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, 'nlg_domain_config.json')
        
        self.config_path = config_path
        self._config: Optional[DomainConfig] = None
    
    def load_config(self) -> DomainConfig:
        """Load and parse the domain configuration."""
        if self._config is not None:
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            self._config = self._parse_config(raw_config)
            logger.info(f"Loaded NLG domain config v{self._config.version}")
            return self._config
            
        except FileNotFoundError:
            logger.error(f"Domain config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in domain config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading domain config: {e}")
            raise
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> DomainConfig:
        """Parse raw JSON config into typed dataclasses."""
        # Parse score tiers
        score_tiers = {}
        for tier_name, tier_data in raw_config['score_tiers'].items():
            score_tiers[tier_name] = ScoreTierConfig(
                threshold=tier_data['threshold'],
                tone=tier_data['tone'],
                focus=tier_data['focus'],
                gaps=tier_data['gaps'],
                adjectives=tier_data['adjectives'],
                reinforcing_words=tier_data['reinforcing_words']
            )
        
        # Parse seniority levels
        seniority_levels = {}
        for level_name, level_data in raw_config['seniority_levels'].items():
            seniority_levels[level_name] = SeniorityConfig(
                experience_range=level_data['experience_range'],
                tone_adjustment=level_data['tone_adjustment'],
                emphasis=level_data['emphasis'],
                leadership_keywords=level_data['leadership_keywords']
            )
        
        # Parse context weights
        context_weights = {}
        for context_name, weights_data in raw_config['context_weights'].items():
            context_weights[context_name] = ContextWeights(
                skills_weight=weights_data['skills_weight'],
                experience_weight=weights_data['experience_weight'],
                projects_weight=weights_data['projects_weight'],
                education_weight=weights_data['education_weight']
            )
        
        # Parse confidence thresholds
        confidence_thresholds = ConfidenceThresholds(
            high_confidence=raw_config['confidence_thresholds']['high_confidence'],
            medium_confidence=raw_config['confidence_thresholds']['medium_confidence'],
            low_confidence=raw_config['confidence_thresholds']['low_confidence']
        )
        
        # Parse template variants
        template_variants = TemplateVariants(
            opening_sentences=raw_config['template_variants']['opening_sentences'],
            skills_phrases=raw_config['template_variants']['skills_phrases'],
            experience_phrases=raw_config['template_variants']['experience_phrases'],
            gap_analysis=raw_config['template_variants']['gap_analysis'],
            transitions=raw_config['template_variants']['transitions']
        )
        
        # Parse special achievements
        special_achievements = SpecialAchievements(
            leadership_threshold=raw_config['special_achievements']['leadership_threshold'],
            rare_skill_threshold=raw_config['special_achievements']['rare_skill_threshold'],
            career_velocity_threshold=raw_config['special_achievements']['career_velocity_threshold'],
            certification_keywords=raw_config['special_achievements']['certification_keywords']
        )
        
        # Parse grammar rules
        grammar_rules = GrammarRules(
            sentence_length=raw_config['grammar_rules']['sentence_length'],
            transitions=raw_config['grammar_rules']['transitions']
        )
        
        return DomainConfig(
            version=raw_config['version'],
            score_tiers=score_tiers,
            seniority_levels=seniority_levels,
            context_weights=context_weights,
            confidence_thresholds=confidence_thresholds,
            template_variants=template_variants,
            special_achievements=special_achievements,
            grammar_rules=grammar_rules
        )
    
    def get_tier_config(self, score_pct: float) -> ScoreTierConfig:
        """Get score tier configuration for a given percentage."""
        config = self.load_config()
        
        # Find the appropriate tier
        for tier_name in ['excellent', 'strong', 'moderate', 'weak']:
            tier_config = config.score_tiers[tier_name]
            if score_pct >= tier_config.threshold:
                return tier_config
        
        # Fallback to weak tier
        return config.score_tiers['weak']
    
    def get_seniority_config(self, seniority_level: str) -> Optional[SeniorityConfig]:
        """Get seniority configuration for a given level."""
        config = self.load_config()
        return config.seniority_levels.get(seniority_level)
    
    def get_context_weights(self, role_type: str = 'technical_roles') -> ContextWeights:
        """Get context weights for a role type."""
        config = self.load_config()
        return config.context_weights.get(role_type, config.context_weights['technical_roles'])
    
    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level string for a given confidence score."""
        config = self.load_config()
        
        if confidence >= config.confidence_thresholds.high_confidence:
            return 'high'
        elif confidence >= config.confidence_thresholds.medium_confidence:
            return 'medium'
        else:
            return 'low'


# Global config loader instance
_config_loader = None


def get_domain_config() -> DomainConfig:
    """Get the global domain configuration instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = DomainConfigLoader()
    return _config_loader.load_config()


def get_config_loader() -> DomainConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = DomainConfigLoader()
    return _config_loader
