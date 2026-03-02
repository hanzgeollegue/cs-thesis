"""
Template library for NLG with micro-templates and variation pools.
Provides deterministic template selection and synonym rotation.
"""

import hashlib
import random
import logging
from typing import Dict, List, Any, Optional
from .nlg_config import get_domain_config, DomainConfig

logger = logging.getLogger(__name__)


class TemplateLibrary:
    """Library of micro-templates organized by section and context."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.templates = self._initialize_templates()
        self.synonyms = self._initialize_synonyms()
        self.transitions = self._initialize_transitions()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Initialize micro-template library."""
        return {
            'opening': {
                'excellent': {
                    'senior': [
                        "{adjective} match with {score}% alignment and {skills}",
                        "{adjective} fit showing {score}% compatibility across {skills}",
                        "Demonstrates {adjective} alignment at {score}% with {skills}",
                        "Shows {adjective} potential with {score}% match and {skills}",
                        "Exhibits {adjective} qualifications at {score}% with {skills}"
                    ],
                    'junior': [
                        "Strong candidate with {score}% match and {skills}",
                        "Good fit showing {score}% alignment and {skills}",
                        "Promising candidate with {score}% compatibility and {skills}",
                        "Solid potential at {score}% match with {skills}",
                        "Well-qualified with {score}% alignment and {skills}"
                    ],
                    'mid': [
                        "Strong candidate with {score}% match and {skills}",
                        "Good fit showing {score}% alignment and {skills}",
                        "Capable candidate with {score}% compatibility and {skills}",
                        "Solid match at {score}% with {skills}",
                        "Well-qualified with {score}% alignment and {skills}"
                    ]
                },
                'strong': {
                    'senior': [
                        "Good candidate with {score}% match and {skills}",
                        "Solid fit showing {score}% alignment and {skills}",
                        "Capable professional with {score}% compatibility and {skills}",
                        "Promising match at {score}% with {skills}",
                        "Well-qualified with {score}% alignment and {skills}"
                    ],
                    'junior': [
                        "Good candidate with {score}% match and {skills}",
                        "Solid potential showing {score}% alignment and {skills}",
                        "Promising candidate with {score}% compatibility and {skills}",
                        "Developing skills at {score}% match with {skills}",
                        "Good foundation with {score}% alignment and {skills}"
                    ],
                    'mid': [
                        "Good candidate with {score}% match and {skills}",
                        "Solid fit showing {score}% alignment and {skills}",
                        "Capable candidate with {score}% compatibility and {skills}",
                        "Promising match at {score}% with {skills}",
                        "Well-qualified with {score}% alignment and {skills}"
                    ]
                },
                'moderate': {
                    'senior': [
                        "Moderate candidate with {score}% match and {skills}",
                        "Developing fit showing {score}% alignment and {skills}",
                        "Potential candidate with {score}% compatibility and {skills}",
                        "Emerging skills at {score}% match with {skills}",
                        "Adequate foundation with {score}% alignment and {skills}"
                    ],
                    'junior': [
                        "Moderate candidate with {score}% match and {skills}",
                        "Developing potential showing {score}% alignment and {skills}",
                        "Emerging candidate with {score}% compatibility and {skills}",
                        "Growing skills at {score}% match with {skills}",
                        "Learning foundation with {score}% alignment and {skills}"
                    ],
                    'mid': [
                        "Moderate candidate with {score}% match and {skills}",
                        "Developing fit showing {score}% alignment and {skills}",
                        "Potential candidate with {score}% compatibility and {skills}",
                        "Emerging skills at {score}% match with {skills}",
                        "Adequate foundation with {score}% alignment and {skills}"
                    ]
                },
                'weak': {
                    'senior': [
                        "Limited match with {score}% alignment and {skills}",
                        "Developing candidate showing {score}% potential and {skills}",
                        "Emerging fit with {score}% compatibility and {skills}",
                        "Basic skills at {score}% match with {skills}",
                        "Foundation level with {score}% alignment and {skills}"
                    ],
                    'junior': [
                        "Limited match with {score}% alignment and {skills}",
                        "Developing potential showing {score}% growth and {skills}",
                        "Emerging candidate with {score}% compatibility and {skills}",
                        "Learning skills at {score}% match with {skills}",
                        "Foundation level with {score}% alignment and {skills}"
                    ],
                    'mid': [
                        "Limited match with {score}% alignment and {skills}",
                        "Developing candidate showing {score}% potential and {skills}",
                        "Emerging fit with {score}% compatibility and {skills}",
                        "Basic skills at {score}% match with {skills}",
                        "Foundation level with {score}% alignment and {skills}"
                    ]
                }
            },
            'skills': {
                'matched_required': [
                    "Has required skill: {skill}",
                    "Demonstrates {skill} proficiency",
                    "Shows expertise in {skill}",
                    "Possesses {skill} capabilities"
                ],
                'matched_required_multiple': [
                    "Required skills: {skills}",
                    "Key competencies: {skills}",
                    "Core skills: {skills}",
                    "Essential skills: {skills}"
                ],
                'matched_required_many': [
                    "Key required skills: {skills} (and {count} more)",
                    "Core competencies: {skills} (plus {count} additional)",
                    "Essential skills: {skills} (and {count} others)",
                    "Primary skills: {skills} (with {count} more)"
                ],
                'matched_nice': [
                    "Bonus skill: {skill}",
                    "Additional competency: {skill}",
                    "Extra qualification: {skill}",
                    "Plus skill: {skill}"
                ],
                'matched_nice_multiple': [
                    "Additional skills: {skills}",
                    "Bonus competencies: {skills}",
                    "Extra qualifications: {skills}",
                    "Plus skills: {skills}"
                ],
                'missing_required': [
                    "Missing critical skill: {skill}",
                    "Lacks {skill} proficiency",
                    "No {skill} experience",
                    "Missing {skill} capabilities"
                ],
                'missing_required_multiple': [
                    "Missing skills: {skills}",
                    "Lacks: {skills}",
                    "Missing competencies: {skills}",
                    "Gap in: {skills}"
                ]
            },
            'experience': {
                'single_role': [
                    "Experience: {role}",
                    "Background: {role}",
                    "Work history: {role}",
                    "Professional experience: {role}"
                ],
                'multiple_roles': [
                    "Experience includes: {roles}",
                    "Background spans: {roles}",
                    "Work history: {roles}",
                    "Professional experience: {roles}"
                ],
                'extensive_roles': [
                    "Extensive experience: {roles} (and {count} more roles)",
                    "Rich background: {roles} (plus {count} additional)",
                    "Diverse experience: {roles} (with {count} more)",
                    "Comprehensive history: {roles} (and {count} others)"
                ],
                'no_experience': [
                    "No work experience provided",
                    "Limited professional background",
                    "Entry-level candidate",
                    "New to professional work"
                ]
            },
            'gaps': {
                'excellent': [
                    "Strong overall fit with minor skill gaps",
                    "Excellent match with minimal development needed",
                    "Outstanding candidate with minor gaps",
                    "Exceptional fit requiring minimal training"
                ],
                'strong': [
                    "Good fit but needs: {skills}",
                    "Solid candidate requiring: {skills}",
                    "Strong potential with gaps in: {skills}",
                    "Good match needing: {skills}"
                ],
                'strong_no_gaps': [
                    "Solid candidate with room for growth",
                    "Good fit with development potential",
                    "Strong foundation for advancement",
                    "Capable candidate ready to expand"
                ],
                'moderate': [
                    "Potential candidate requiring: {skills}",
                    "Developing fit needing: {skills}",
                    "Moderate match with gaps in: {skills}",
                    "Emerging candidate requiring: {skills}"
                ],
                'moderate_no_gaps': [
                    "Moderate fit with development potential",
                    "Adequate candidate with growth opportunity",
                    "Developing skills with advancement potential",
                    "Foundation level with expansion possible"
                ],
                'weak': [
                    "Significant skill gaps in: {skills}",
                    "Major development needed in: {skills}",
                    "Substantial gaps requiring: {skills}",
                    "Limited match needing: {skills}"
                ],
                'weak_no_gaps': [
                    "Limited match requiring substantial development",
                    "Foundation level with significant growth needed",
                    "Basic qualifications requiring major development",
                    "Entry-level requiring extensive training"
                ]
            },
            # New evidence-based template categories
            'caveat': {
                'low_parsing': [
                    "Limited parsing data available; assessment based on available sections.",
                    "Incomplete resume parsing; analysis relies on extracted content.",
                    "Parsing constraints noted; evaluation based on accessible information."
                ],
                'low_ce_evidence': [
                    "Assessment based on skill matching and experience analysis.",
                    "Using skill extraction and text analysis for evaluation.",
                    "Analysis based on available resume content and skill matching."
                ],
                'low_confidence': [
                    "Evidence confidence is limited; focusing on high-certainty findings.",
                    "Data quality constraints noted; emphasizing verified information.",
                    "Assessment based on available evidence with noted limitations."
                ]
            },
            'evidence': {
                'strength_with_context': [
                    "Strong {skill} background ({context})",
                    "Demonstrates {skill} expertise ({context})",
                    "Shows {skill} proficiency ({context})",
                    "Exhibits {skill} capabilities ({context})"
                ],
                'strength_with_quote': [
                    "Strong {skill} background ({context}) ('{quote}')",
                    "Demonstrates {skill} expertise ({context}) ('{quote}')",
                    "Shows {skill} proficiency ({context}) ('{quote}')",
                    "Exhibits {skill} capabilities ({context}) ('{quote}')"
                ]
            },
            'gap_context': {
                'critical_missing': [
                    "Missing {skill} experience (required for {role_context})",
                    "No {skill} background (needed for {role_context})",
                    "Lacks {skill} exposure (critical for {role_context})",
                    "Missing {skill} skills (essential for {role_context})"
                ]
            },
            'concrete_example': {
                'with_context': [
                    "Demonstrated {skill} through {context} at {company}",
                    "Applied {skill} in {context} while at {company}",
                    "Used {skill} for {context} at {company}",
                    "Implemented {skill} via {context} at {company}"
                ],
                'with_quote': [
                    "Demonstrated {skill} through {context} at {company} ('{quote}')",
                    "Applied {skill} in {context} while at {company} ('{quote}')",
                    "Used {skill} for {context} at {company} ('{quote}')",
                    "Implemented {skill} via {context} at {company} ('{quote}')"
                ]
            }
        }
    
    def _initialize_synonyms(self) -> Dict[str, List[str]]:
        """Initialize synonym dictionaries."""
        return {
            'strong': ['solid', 'robust', 'capable', 'competent', 'qualified'],
            'good': ['solid', 'decent', 'adequate', 'satisfactory', 'acceptable'],
            'moderate': ['developing', 'emerging', 'potential', 'adequate', 'basic'],
            'limited': ['developing', 'emerging', 'basic', 'foundation', 'entry-level'],
            'experience': ['background', 'history', 'track record', 'work history'],
            'skills': ['competencies', 'capabilities', 'qualifications', 'expertise'],
            'match': ['fit', 'alignment', 'compatibility', 'suitability'],
            'candidate': ['professional', 'individual', 'applicant', 'prospect'],
            'demonstrates': ['shows', 'exhibits', 'displays', 'presents'],
            'requires': ['needs', 'demands', 'necessitates', 'calls for'],
            'development': ['growth', 'advancement', 'improvement', 'enhancement']
        }
    
    def _initialize_transitions(self) -> List[str]:
        """Initialize transition phrases."""
        return [
            "Additionally",
            "However",
            "Furthermore",
            "Moreover",
            "In contrast",
            "Specifically",
            "Notably",
            "Importantly",
            "Significantly",
            "Particularly"
        ]
    
    def get_template(self, section: str, tier: str, seniority: str, variant_index: int) -> str:
        """Get a specific template by variant index."""
        try:
            templates = self.templates[section][tier][seniority]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            # Fallback to first available template
            try:
                templates = self.templates[section][tier][seniority]
                return templates[0]
            except (KeyError, IndexError):
                # Ultimate fallback
                return "{adjective} candidate with {score}% match and {skills}"
    
    def get_phrase_template(self, section: str, phrase_type: str, variant_index: int) -> str:
        """Get a template by phrase type (for skills, experience, etc.)."""
        try:
            templates = self.templates[section][phrase_type]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            # Fallback to first available template
            try:
                templates = self.templates[section][phrase_type]
                return templates[0]
            except (KeyError, IndexError):
                # Ultimate fallback
                return "Template not available"
    
    def get_synonym(self, word: str, variant_index: int) -> str:
        """Get a synonym for a word based on variant index."""
        synonyms = self.synonyms.get(word.lower(), [word])
        return synonyms[variant_index % len(synonyms)]
    
    def get_transition(self, variant_index: int) -> str:
        """Get a transition phrase based on variant index."""
        return self.transitions[variant_index % len(self.transitions)]
    
    def get_caveat_template(self, caveat_type: str, variant_index: int) -> str:
        """Get a caveat template based on type and variant index."""
        try:
            templates = self.templates['caveat'][caveat_type]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            return "Assessment based on available data."
    
    def get_evidence_template(self, evidence_type: str, variant_index: int) -> str:
        """Get an evidence template based on type and variant index."""
        try:
            templates = self.templates['evidence'][evidence_type]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            return "Demonstrates {skill} capabilities"
    
    def get_gap_context_template(self, gap_type: str, variant_index: int) -> str:
        """Get a gap context template based on type and variant index."""
        try:
            templates = self.templates['gap_context'][gap_type]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            return "Missing {skill} experience"
    
    def get_concrete_example_template(self, example_type: str, variant_index: int) -> str:
        """Get a concrete example template based on type and variant index."""
        try:
            templates = self.templates['concrete_example'][example_type]
            return templates[variant_index % len(templates)]
        except (KeyError, IndexError):
            return "Demonstrated {skill} at {company}"


class VariantSelector:
    """Deterministic variant selector for template rotation."""
    
    def __init__(self, candidate_id: str):
        """Initialize with candidate ID for deterministic selection."""
        self.candidate_id = str(candidate_id)
        self._hash_seed = self._generate_hash_seed()
    
    def _generate_hash_seed(self) -> int:
        """Generate deterministic hash seed from candidate ID."""
        hash_obj = hashlib.md5(self.candidate_id.encode('utf-8'))
        return int(hash_obj.hexdigest()[:8], 16)
    
    def select_variant(self, pool_size: int, context: str = '') -> int:
        """Select variant index deterministically."""
        # Use context to create unique selection per context
        context_seed = f"{self.candidate_id}_{context}"
        hash_obj = hashlib.md5(context_seed.encode('utf-8'))
        variant_hash = int(hash_obj.hexdigest()[:8], 16)
        
        return variant_hash % pool_size
    
    def select_opening(self, tier: str, seniority: str, template_lib: TemplateLibrary) -> int:
        """Select opening template variant."""
        # Get the number of variants for this tier/seniority combination
        try:
            variants = template_lib.templates['opening'][tier][seniority]
            pool_size = len(variants)
        except (KeyError, IndexError):
            pool_size = 5  # Default fallback
        
        return self.select_variant(pool_size, f"opening_{tier}_{seniority}")
    
    def select_skills_phrase(self, phrase_type: str, template_lib: TemplateLibrary) -> int:
        """Select skills phrase variant."""
        try:
            variants = template_lib.templates['skills'][phrase_type]
            pool_size = len(variants)
        except (KeyError, IndexError):
            pool_size = 4  # Default fallback
        
        return self.select_variant(pool_size, f"skills_{phrase_type}")
    
    def select_experience_phrase(self, phrase_type: str, template_lib: TemplateLibrary) -> int:
        """Select experience phrase variant."""
        try:
            variants = template_lib.templates['experience'][phrase_type]
            pool_size = len(variants)
        except (KeyError, IndexError):
            pool_size = 3  # Default fallback
        
        return self.select_variant(pool_size, f"experience_{phrase_type}")
    
    def select_gap_analysis(self, tier: str, has_gaps: bool, template_lib: TemplateLibrary) -> int:
        """Select gap analysis variant."""
        gap_type = tier if not has_gaps else f"{tier}_no_gaps"
        try:
            variants = template_lib.templates['gaps'][gap_type]
            pool_size = len(variants)
        except (KeyError, IndexError):
            pool_size = 4  # Default fallback
        
        return self.select_variant(pool_size, f"gaps_{gap_type}")
    
    def select_transition(self, template_lib: TemplateLibrary) -> int:
        """Select transition phrase variant."""
        return self.select_variant(len(template_lib.transitions), "transition")


class TemplateEngine:
    """Main template engine for rendering analysis text."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
        self.template_lib = TemplateLibrary(self.config)
    
    def render_opening(self, score_pct: float, tier: str, seniority: str, 
                      skills_info: Dict[str, Any], candidate_id: str) -> str:
        """Render opening sentence with variation."""
        selector = VariantSelector(candidate_id)
        variant_index = selector.select_opening(tier, seniority, self.template_lib)
        
        # Get template
        template = self.template_lib.get_template('opening', tier, seniority, variant_index)
        
        # Get tier config for adjectives
        tier_config = self.config.score_tiers[tier]
        adjective = tier_config.adjectives[variant_index % len(tier_config.adjectives)]
        
        # Format skills string
        matched_count = len(skills_info.get('matched_required', []))
        total_required = skills_info.get('total_required', 0)
        skills_str = f"{matched_count}/{total_required} required skills"
        
        # Render template
        return template.format(
            adjective=adjective,
            score=int(score_pct),
            skills=skills_str
        )
    
    def render_skills_analysis(self, skills_info: Dict[str, Any], tier: str, 
                              candidate_id: str) -> str:
        """Render skills analysis with variation."""
        selector = VariantSelector(candidate_id)
        parts = []
        
        matched_required = skills_info.get('matched_required', [])
        matched_nice = skills_info.get('matched_nice', [])
        missing_required = skills_info.get('missing_required', [])
        
        # Required skills
        if matched_required:
            if len(matched_required) == 1:
                phrase_type = 'matched_required'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skill=matched_required[0]))
            elif len(matched_required) <= 3:
                phrase_type = 'matched_required_multiple'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skills=', '.join(matched_required)))
            else:
                phrase_type = 'matched_required_many'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                top_skills = matched_required[:3]
                parts.append(template.format(
                    skills=', '.join(top_skills),
                    count=len(matched_required) - 3
                ))
        
        # Nice-to-have skills
        if matched_nice:
            if len(matched_nice) == 1:
                phrase_type = 'matched_nice'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skill=matched_nice[0]))
            else:
                phrase_type = 'matched_nice_multiple'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skills=', '.join(matched_nice[:3])))
        
        # Missing skills (only for weak/moderate tiers)
        if missing_required and tier in ['weak', 'moderate']:
            if len(missing_required) == 1:
                phrase_type = 'missing_required'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skill=missing_required[0]))
            else:
                phrase_type = 'missing_required_multiple'
                variant_index = selector.select_skills_phrase(phrase_type, self.template_lib)
                template = self.template_lib.get_phrase_template('skills', phrase_type, variant_index)
                parts.append(template.format(skills=', '.join(missing_required[:3])))
        
        return '; '.join(parts) if parts else "No matching skills identified in resume"
    
    def render_experience_analysis(self, experience_info: Dict[str, Any], 
                                  candidate_id: str) -> str:
        """Render experience analysis with variation."""
        if not experience_info.get('has_relevant', False):
            return "No work experience provided"
        
        selector = VariantSelector(candidate_id)
        count = experience_info.get('count', 0)
        top_roles = experience_info.get('top_roles', [])
        
        if count == 1:
            phrase_type = 'single_role'
            variant_index = selector.select_experience_phrase(phrase_type, self.template_lib)
            template = self.template_lib.get_phrase_template('experience', phrase_type, variant_index)
            return template.format(role=top_roles[0])
        elif count <= 3:
            phrase_type = 'multiple_roles'
            variant_index = selector.select_experience_phrase(phrase_type, self.template_lib)
            template = self.template_lib.get_phrase_template('experience', phrase_type, variant_index)
            return template.format(roles='; '.join(top_roles))
        else:
            phrase_type = 'extensive_roles'
            variant_index = selector.select_experience_phrase(phrase_type, self.template_lib)
            template = self.template_lib.get_phrase_template('experience', phrase_type, variant_index)
            return template.format(
                roles='; '.join(top_roles[:2]),
                count=count - 2
            )
    
    def render_gap_analysis(self, skills_info: Dict[str, Any], tier: str, 
                           candidate_id: str) -> str:
        """Render gap analysis with variation."""
        selector = VariantSelector(candidate_id)
        missing_required = skills_info.get('missing_required', [])
        has_gaps = len(missing_required) > 0
        
        gap_type = tier if has_gaps else f"{tier}_no_gaps"
        variant_index = selector.select_gap_analysis(tier, not has_gaps, self.template_lib)
        
        try:
            template = self.template_lib.get_phrase_template('gaps', gap_type, variant_index)
        except (KeyError, IndexError):
            # Fallback to first available template
            try:
                template = self.template_lib.templates['gaps'][gap_type][0]
            except (KeyError, IndexError):
                return "Analysis unavailable"
        
        if has_gaps and missing_required:
            return template.format(skills=', '.join(missing_required[:2]))
        else:
            return template
    
    def add_transition(self, candidate_id: str) -> str:
        """Add transition phrase."""
        selector = VariantSelector(candidate_id)
        variant_index = selector.select_transition(self.template_lib)
        return self.template_lib.get_transition(variant_index)
