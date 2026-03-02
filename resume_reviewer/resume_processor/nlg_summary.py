"""
Bullet summary generator for scannable facts.
Generates concise bullet points separate from prose narrative.
"""

import logging
from typing import Dict, List, Any, Optional
from .nlg_config import get_domain_config, DomainConfig

logger = logging.getLogger(__name__)


class BulletSummaryGenerator:
    """Generates scannable bullet point summaries."""
    
    def __init__(self, config: Optional[DomainConfig] = None):
        """Initialize with domain configuration."""
        self.config = config or get_domain_config()
    
    def generate_bullets(self, candidate_data: Dict[str, Any], 
                        facts: Dict[str, Any], batch_stats: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate bullet point summary for a candidate.
        
        Args:
            candidate_data: Candidate resume data
            facts: Extracted facts
            batch_stats: Batch statistics for comparative insights
            
        Returns:
            List of bullet point strings
        """
        try:
            bullets = []
            
            # Score context bullet
            score_bullet = self._generate_score_bullet(facts, batch_stats)
            if score_bullet:
                bullets.append(score_bullet)
            
            # Skills match bullet
            skills_bullet = self._generate_skills_bullet(facts)
            if skills_bullet:
                bullets.append(skills_bullet)
            
            # Missing skills bullet (if applicable)
            missing_bullet = self._generate_missing_skills_bullet(facts)
            if missing_bullet:
                bullets.append(missing_bullet)
            
            # Experience bullet
            experience_bullet = self._generate_experience_bullet(facts)
            if experience_bullet:
                bullets.append(experience_bullet)
            
            # Special achievements bullet
            achievement_bullet = self._generate_achievement_bullet(facts)
            if achievement_bullet:
                bullets.append(achievement_bullet)
            
            # Coverage bullet
            coverage_bullet = self._generate_coverage_bullet(facts)
            if coverage_bullet:
                bullets.append(coverage_bullet)
            
            # Limit to 5 bullets maximum
            return bullets[:5]
            
        except Exception as e:
            logger.warning(f"Error generating bullets: {e}")
            return [f"Summary unavailable: {str(e)}"]
    
    def _generate_score_bullet(self, facts: Dict[str, Any], 
                              batch_stats: Optional[Dict[str, Any]]) -> Optional[str]:
        """Generate score context bullet with adaptive detail for shortlist candidates."""
        try:
            scores = facts.get('scores', {})
            score_pct = scores.get('final_score_percentage', 0)
            
            if not batch_stats:
                return f"{score_pct}% match"
            
            # Add batch context
            batch_position = self._calculate_batch_position(score_pct, batch_stats)
            percentile = batch_position.get('percentile', 50)
            rank = batch_position.get('rank_estimate', 1)
            batch_size = batch_position.get('batch_size', 1)
            gap_from_top = batch_position.get('gap_from_top', 0)
            
            # Determine if this is a shortlist candidate (top 25% or rank <= 3)
            is_shortlist = rank <= 3 or percentile >= 75
            
            if is_shortlist and batch_size > 1:
                # Detailed context for shortlist candidates
                return f"#{rank} of {batch_size} ({score_pct}%, {gap_from_top:.1f} pts behind leader)"
            else:
                # Simple tier description for others
                if percentile >= 90:
                    return f"{score_pct}% match (top 10% of batch)"
                elif percentile >= 75:
                    return f"{score_pct}% match (top 25% of batch)"
                elif percentile >= 50:
                    return f"{score_pct}% match (above median)"
                elif percentile >= 25:
                    return f"{score_pct}% match (below median)"
                else:
                    return f"{score_pct}% match (bottom 25% of batch)"
                
        except Exception as e:
            logger.warning(f"Error generating score bullet: {e}")
            return None
    
    def _generate_skills_bullet(self, facts: Dict[str, Any]) -> Optional[str]:
        """Generate skills match bullet."""
        try:
            skills = facts.get('skills', {})
            matched_required = skills.get('matched_required', [])
            total_required = skills.get('total_required', 0)
            
            if not matched_required:
                return "No required skills matched"
            
            if len(matched_required) == total_required:
                return f"{len(matched_required)}/{total_required} required skills: {', '.join(matched_required)}"
            else:
                return f"{len(matched_required)}/{total_required} required skills: {', '.join(matched_required[:3])}"
                
        except Exception as e:
            logger.warning(f"Error generating skills bullet: {e}")
            return None
    
    def _generate_missing_skills_bullet(self, facts: Dict[str, Any]) -> Optional[str]:
        """Generate missing skills bullet."""
        try:
            skills = facts.get('skills', {})
            missing_required = skills.get('missing_required', [])
            
            if not missing_required:
                return None
            
            if len(missing_required) == 1:
                return f"Missing: {missing_required[0]}"
            else:
                return f"Missing: {', '.join(missing_required[:3])}"
                
        except Exception as e:
            logger.warning(f"Error generating missing skills bullet: {e}")
            return None
    
    def _generate_experience_bullet(self, facts: Dict[str, Any]) -> Optional[str]:
        """Generate experience bullet."""
        try:
            experience = facts.get('experience', {})
            
            if not experience.get('has_relevant', False):
                return "No work experience provided"
            
            count = experience.get('count', 0)
            top_roles = experience.get('top_roles', [])
            
            if count == 1:
                return f"Experience: {top_roles[0]}"
            elif count <= 3:
                return f"{count} roles: {top_roles[0]}"
            else:
                return f"{count} roles: {top_roles[0]} (and {count-1} more)"
                
        except Exception as e:
            logger.warning(f"Error generating experience bullet: {e}")
            return None
    
    def _generate_achievement_bullet(self, facts: Dict[str, Any]) -> Optional[str]:
        """Generate special achievement bullet."""
        try:
            # Check for rare qualifications
            rare_qualifications = facts.get('rare_qualifications', [])
            if rare_qualifications:
                return f"Special qualifications: {', '.join(rare_qualifications[:2])}"
            
            # Check for leadership experience
            leadership = facts.get('leadership', {})
            if leadership.get('has_management', False):
                avg_team_size = leadership.get('avg_team_size', 0)
                if avg_team_size > 0:
                    return f"Leadership experience: managed teams of {avg_team_size:.0f}"
                else:
                    return "Leadership experience: team management"
            
            # Check for career progression
            progression = facts.get('progression', {})
            if progression.get('leadership_growth', False):
                return "Career progression: leadership growth"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating achievement bullet: {e}")
            return None
    
    def _generate_coverage_bullet(self, facts: Dict[str, Any]) -> Optional[str]:
        """Generate coverage bullet."""
        try:
            scores = facts.get('scores', {})
            coverage = scores.get('coverage', 0)
            
            if coverage >= 0.8:
                return f"High coverage: {coverage:.0%}"
            elif coverage >= 0.6:
                return f"Good coverage: {coverage:.0%}"
            elif coverage >= 0.4:
                return f"Moderate coverage: {coverage:.0%}"
            else:
                return f"Low coverage: {coverage:.0%}"
                
        except Exception as e:
            logger.warning(f"Error generating coverage bullet: {e}")
            return None
    
    def _calculate_batch_position(self, score_pct: float, 
                                 batch_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate candidate's position in batch."""
        try:
            # Get batch statistics
            score_distribution = batch_stats.get('score_distribution', {})
            top_score = batch_stats.get('top_score', 100)
            median_score = batch_stats.get('median_score', 50)
            candidate_count = batch_stats.get('candidate_count', 1)
            
            # Calculate percentile (rough estimate)
            if score_pct >= top_score:
                percentile = 100
            elif score_pct >= median_score:
                # Linear interpolation between median and top
                percentile = 50 + ((score_pct - median_score) / (top_score - median_score)) * 50
            else:
                # Linear interpolation between 0 and median
                percentile = (score_pct / median_score) * 50
            
            # Calculate gap from top
            gap_from_top = top_score - score_pct
            
            return {
                'percentile': percentile,
                'gap_from_top': gap_from_top,
                'above_median': score_pct >= median_score,
                'rank_estimate': int((100 - percentile) / 100 * candidate_count) + 1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating batch position: {e}")
            return {
                'percentile': 50,
                'gap_from_top': 0,
                'above_median': True,
                'rank_estimate': 1
            }
    
    def generate_comparative_bullets(self, candidate_data: Dict[str, Any], 
                                   facts: Dict[str, Any], 
                                   batch_stats: Dict[str, Any]) -> List[str]:
        """Generate bullets with comparative batch insights."""
        try:
            bullets = []
            
            # Batch rank bullet
            batch_position = self._calculate_batch_position(
                facts.get('scores', {}).get('final_score_percentage', 0), 
                batch_stats
            )
            
            rank_estimate = batch_position['rank_estimate']
            total_candidates = batch_stats.get('candidate_count', 1)
            
            if rank_estimate <= 3:
                bullets.append(f"Rank: #{rank_estimate} of {total_candidates} (top performer)")
            elif rank_estimate <= total_candidates // 4:
                bullets.append(f"Rank: #{rank_estimate} of {total_candidates} (top 25%)")
            elif rank_estimate <= total_candidates // 2:
                bullets.append(f"Rank: #{rank_estimate} of {total_candidates} (above median)")
            else:
                bullets.append(f"Rank: #{rank_estimate} of {total_candidates} (below median)")
            
            # Gap from top bullet
            gap_from_top = batch_position['gap_from_top']
            if gap_from_top > 0:
                bullets.append(f"Gap from top: {gap_from_top:.1f} points")
            
            # Skills comparison bullet
            skills_comparison = self._generate_skills_comparison_bullet(facts, batch_stats)
            if skills_comparison:
                bullets.append(skills_comparison)
            
            return bullets
            
        except Exception as e:
            logger.warning(f"Error generating comparative bullets: {e}")
            return []
    
    def _generate_skills_comparison_bullet(self, facts: Dict[str, Any], 
                                         batch_stats: Dict[str, Any]) -> Optional[str]:
        """Generate skills comparison bullet."""
        try:
            skills = facts.get('skills', {})
            matched_count = len(skills.get('matched_required', []))
            total_required = skills.get('total_required', 0)
            
            if total_required == 0:
                return None
            
            # Get batch average (if available)
            batch_avg_skills = batch_stats.get('avg_skills_matched', 0)
            
            if batch_avg_skills > 0:
                if matched_count > batch_avg_skills:
                    return f"Skills: {matched_count}/{total_required} (above batch avg: {batch_avg_skills:.1f})"
                elif matched_count < batch_avg_skills:
                    return f"Skills: {matched_count}/{total_required} (below batch avg: {batch_avg_skills:.1f})"
                else:
                    return f"Skills: {matched_count}/{total_required} (at batch avg)"
            else:
                return f"Skills: {matched_count}/{total_required}"
                
        except Exception as e:
            logger.warning(f"Error generating skills comparison bullet: {e}")
            return None
