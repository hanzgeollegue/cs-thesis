# Skill Taxonomy Directory

This directory contains domain-specific skill taxonomies for the resume ranking system.

## Structure

- `config.json` - Configuration file that specifies which domains to load and fuzzy matching settings
- `marketing_sales.json` - Marketing and sales tools/platforms (Salesforce, HubSpot, Google Analytics, etc.)
- `healthcare.json` - Healthcare systems and certifications (Epic, Cerner, EHR systems, etc.)
- `finance_accounting.json` - Finance and accounting software (QuickBooks, SAP, Oracle Financials, etc.)

## Configuration

The `config.json` file controls:
- `enabled_domains`: List of domain JSON files to load (e.g., `["marketing_sales", "healthcare"]`)
- `fuzzy_matching.enabled`: Enable/disable semantic fuzzy matching for unknown skills
- `fuzzy_matching.threshold`: Minimum similarity score (0.0-1.0) for fuzzy matches
- `confidence_weights`: Confidence scores for exact vs fuzzy matches

## Adding New Skills

To add skills to a domain:
1. Edit the appropriate JSON file (e.g., `marketing_sales.json`)
2. Add entries in the format: `"canonical_skill_name": ["variation1", "variation2", ...]`
3. The system will automatically load new skills on next initialization

## Adding New Domains

To add a new domain:
1. Create a new JSON file (e.g., `engineering.json`)
2. Add skills in the same format as existing domain files
3. Add the domain name to `enabled_domains` in `config.json`

## Example

```json
{
  "salesforce": ["salesforce", "sfdc", "salesforce crm", "salesforce.com"],
  "hubspot": ["hubspot", "hubspot crm", "hubspot marketing"]
}
```

## Notes

- Core tech skills (Python, JavaScript, React, etc.) are hard-coded in `text_processor.py` for performance
- Domain skills are loaded from JSON files for flexibility
- Fuzzy matching uses semantic similarity (SBERT) to match unknown skills with lower confidence
- All skills are normalized to canonical forms for consistent matching

