"""
Metadata tracking for NLG explainability and provenance.
Tracks which rules, templates, and inputs generated each part of the analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from .nlg_config import get_domain_config, DomainConfig

logger = logging.getLogger(__name__)


@dataclass
class SentenceMetadata:
    """Metadata for a single sentence in the analysis."""
    sentence_index: int
    template_id: str
    variant_index: int
    inputs_used: List[str]
    rules_applied: List[str]
    context_factors: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0


@dataclass
class AnalysisMetadata:
    """Complete metadata for an analysis."""
    version: str
    template_library_version: str
    domain_config_version: str
    sentences: List[SentenceMetadata] = field(default_factory=list)
    polish_operations: List[str] = field(default_factory=list)
    profile_context: Dict[str, Any] = field(default_factory=dict)
    batch_context: Dict[str, Any] = field(default_factory=dict)
    generation_timestamp: str = ""
    total_processing_time: float = 0.0


class ProvenanceTracker:
    """Tracks provenance and metadata for NLG generation."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.metadata = AnalysisMetadata(
            version="2.0",
            template_library_version="2.0",
            domain_config_version=self.config.version
        )
        self.current_sentence_index = 0
    
    def track_sentence(self, template_id: str, variant_index: int, 
                      inputs_used: List[str], rules_applied: List[str],
                      context_factors: Optional[Dict[str, Any]] = None,
                      confidence_score: float = 1.0) -> None:
        """Track metadata for a generated sentence."""
        sentence_meta = SentenceMetadata(
            sentence_index=self.current_sentence_index,
            template_id=template_id,
            variant_index=variant_index,
            inputs_used=inputs_used,
            rules_applied=rules_applied,
            context_factors=context_factors or {},
            confidence_score=confidence_score
        )
        
        self.metadata.sentences.append(sentence_meta)
        self.current_sentence_index += 1
    
    def track_polish_operation(self, operation: str) -> None:
        """Track a polish operation applied to the text."""
        self.metadata.polish_operations.append(operation)
    
    def set_profile_context(self, profile_context: Dict[str, Any]) -> None:
        """Set the profile context used for generation."""
        self.metadata.profile_context = profile_context
    
    def set_batch_context(self, batch_context: Dict[str, Any]) -> None:
        """Set the batch context used for generation."""
        self.metadata.batch_context = batch_context
    
    def set_context(self, candidate_id: str, job_description_id: str, 
                   batch_context: Optional[Dict[str, Any]] = None,
                   profile_context: Optional[Dict[str, Any]] = None) -> None:
        """Set the overall context for generation."""
        self.metadata.candidate_id = candidate_id
        self.metadata.job_description_id = job_description_id
        if batch_context:
            self.metadata.batch_context = batch_context
        if profile_context:
            self.metadata.profile_context = profile_context
    
    def set_generation_info(self, timestamp: str, processing_time: float) -> None:
        """Set generation timestamp and processing time."""
        self.metadata.generation_timestamp = timestamp
        self.metadata.total_processing_time = processing_time
    
    def set_processing_time(self, processing_time_ms: int) -> None:
        """Set processing time in milliseconds."""
        self.metadata.total_processing_time = processing_time_ms / 1000.0  # Convert to seconds
    
    def get_metadata(self) -> AnalysisMetadata:
        """Get the complete metadata."""
        return self.metadata
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata as dictionary for serialization."""
        return {
            'version': self.metadata.version,
            'template_library_version': self.metadata.template_library_version,
            'domain_config_version': self.metadata.domain_config_version,
            'sentences': [
                {
                    'sentence_index': s.sentence_index,
                    'template_id': s.template_id,
                    'variant_index': s.variant_index,
                    'inputs_used': s.inputs_used,
                    'rules_applied': s.rules_applied,
                    'context_factors': s.context_factors,
                    'confidence_score': s.confidence_score
                }
                for s in self.metadata.sentences
            ],
            'polish_operations': self.metadata.polish_operations,
            'profile_context': self.metadata.profile_context,
            'batch_context': self.metadata.batch_context,
            'generation_timestamp': self.metadata.generation_timestamp,
            'total_processing_time': self.metadata.total_processing_time
        }
    
    def get_explainability_summary(self) -> Dict[str, Any]:
        """Get a summary for explainability purposes."""
        if not self.metadata.sentences:
            return {'error': 'No sentences tracked'}
        
        # Analyze template usage
        template_usage = {}
        for sentence in self.metadata.sentences:
            template_id = sentence.template_id
            if template_id not in template_usage:
                template_usage[template_id] = 0
            template_usage[template_id] += 1
        
        # Analyze input usage
        input_usage = {}
        for sentence in self.metadata.sentences:
            for input_field in sentence.inputs_used:
                if input_field not in input_usage:
                    input_usage[input_field] = 0
                input_usage[input_field] += 1
        
        # Analyze rule usage
        rule_usage = {}
        for sentence in self.metadata.sentences:
            for rule in sentence.rules_applied:
                if rule not in rule_usage:
                    rule_usage[rule] = 0
                rule_usage[rule] += 1
        
        # Calculate confidence metrics
        confidence_scores = [s.confidence_score for s in self.metadata.sentences]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_sentences': len(self.metadata.sentences),
            'template_usage': template_usage,
            'input_usage': input_usage,
            'rule_usage': rule_usage,
            'polish_operations_count': len(self.metadata.polish_operations),
            'average_confidence': avg_confidence,
            'profile_context_available': bool(self.metadata.profile_context),
            'batch_context_available': bool(self.metadata.batch_context),
            'generation_time': self.metadata.total_processing_time
        }
    
    def validate_metadata(self) -> Dict[str, Any]:
        """Validate metadata completeness and consistency."""
        issues = []
        warnings = []
        
        # Check for required fields
        if not self.metadata.sentences:
            issues.append("No sentences tracked")
        
        if not self.metadata.version:
            issues.append("Missing version information")
        
        # Check sentence metadata
        for i, sentence in enumerate(self.metadata.sentences):
            if not sentence.template_id:
                issues.append(f"Sentence {i}: Missing template_id")
            
            if not sentence.inputs_used:
                warnings.append(f"Sentence {i}: No inputs tracked")
            
            if not sentence.rules_applied:
                warnings.append(f"Sentence {i}: No rules tracked")
            
            if sentence.confidence_score < 0 or sentence.confidence_score > 1:
                issues.append(f"Sentence {i}: Invalid confidence score {sentence.confidence_score}")
        
        # Check for consistency
        if self.metadata.sentences:
            expected_indices = list(range(len(self.metadata.sentences)))
            actual_indices = [s.sentence_index for s in self.metadata.sentences]
            if expected_indices != actual_indices:
                issues.append("Sentence indices are not sequential")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_sentences': len(self.metadata.sentences),
            'total_polish_operations': len(self.metadata.polish_operations)
        }
    
    def reset(self) -> None:
        """Reset the tracker for a new analysis."""
        self.metadata = AnalysisMetadata(
            version="2.0",
            template_library_version="2.0",
            domain_config_version=self.config.version
        )
        self.current_sentence_index = 0


class MetadataEmbedder:
    """Embeds metadata into analysis output structure."""
    
    @staticmethod
    def embed_metadata(analysis_text: str, bullets: List[str], 
                      facts: Dict[str, Any], metadata: AnalysisMetadata) -> Dict[str, Any]:
        """
        Embed metadata into the analysis output structure.
        
        Args:
            analysis_text: Generated analysis text
            bullets: Generated bullet points
            facts: Extracted facts
            metadata: Analysis metadata
            
        Returns:
            Complete analysis structure with embedded metadata
        """
        return {
            'text': analysis_text,
            'bullets': bullets,
            'facts': facts,
            'metadata': {
                'version': metadata.version,
                'template_library_version': metadata.template_library_version,
                'domain_config_version': metadata.domain_config_version,
                'sentences': [
                    {
                        'sentence_index': s.sentence_index,
                        'template_id': s.template_id,
                        'variant_index': s.variant_index,
                        'inputs_used': s.inputs_used,
                        'rules_applied': s.rules_applied,
                        'context_factors': s.context_factors,
                        'confidence_score': s.confidence_score
                    }
                    for s in metadata.sentences
                ],
                'polish_operations': metadata.polish_operations,
                'profile_context': metadata.profile_context,
                'batch_context': metadata.batch_context,
                'generation_timestamp': metadata.generation_timestamp,
                'total_processing_time': metadata.total_processing_time,
                'explainability_summary': ProvenanceTracker().get_explainability_summary()
            }
        }
    
    @staticmethod
    def extract_metadata_from_analysis(analysis_data: Dict[str, Any]) -> Optional[AnalysisMetadata]:
        """
        Extract metadata from analysis data structure.
        
        Args:
            analysis_data: Analysis data with embedded metadata
            
        Returns:
            Extracted metadata or None if not available
        """
        try:
            metadata_dict = analysis_data.get('metadata', {})
            if not metadata_dict:
                return None
            
            sentences = []
            for s_dict in metadata_dict.get('sentences', []):
                sentence_meta = SentenceMetadata(
                    sentence_index=s_dict.get('sentence_index', 0),
                    template_id=s_dict.get('template_id', ''),
                    variant_index=s_dict.get('variant_index', 0),
                    inputs_used=s_dict.get('inputs_used', []),
                    rules_applied=s_dict.get('rules_applied', []),
                    context_factors=s_dict.get('context_factors', {}),
                    confidence_score=s_dict.get('confidence_score', 1.0)
                )
                sentences.append(sentence_meta)
            
            metadata = AnalysisMetadata(
                version=metadata_dict.get('version', '2.0'),
                template_library_version=metadata_dict.get('template_library_version', '2.0'),
                domain_config_version=metadata_dict.get('domain_config_version', '2.0'),
                sentences=sentences,
                polish_operations=metadata_dict.get('polish_operations', []),
                profile_context=metadata_dict.get('profile_context', {}),
                batch_context=metadata_dict.get('batch_context', {}),
                generation_timestamp=metadata_dict.get('generation_timestamp', ''),
                total_processing_time=metadata_dict.get('total_processing_time', 0.0)
            )
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return None
