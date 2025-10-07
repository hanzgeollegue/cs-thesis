# Resume Ranking – Process Guide

## What this app does
- Upload a batch of PDF resumes for a role
- Enter the job description and key criteria
- The system parses resumes, scores them against the criteria, and produces a ranked list with friendly rationales
- Export results (JSON/CSV) for tracking and analysis

## Run locally
1. Install
   ```bash
   pip install -r requirements.txt
   ```
2. Start server
   ```bash
   python manage.py runserver
   ```
3. Open
   - Browser: http://127.0.0.1:8000

## End-to-end process
1. Define role
   - Go to JD page
   - Provide position title, must-have skills, nice-to-have skills, and notes
2. Upload resumes
   - Select one or more PDF files
   - Submit to start processing; the page will show progress
3. Processing
   - Text is extracted per resume and parsed into sections (experience/skills/education)
   - Multiple scoring passes run to assess relevance and coverage
   - Scores are combined into a final score per candidate
4. Results
   - See a ranked list of candidates with:
     - Candidate name
     - Final score (0–100%)
     - High-level rationale (strengths and gaps)
     - Explicit and inferred skill matches
   - Click a candidate to view details
   - Compare candidates side-by-side
5. Export
   - On the results page, click Export
   - Default: JSON. Hold Shift for CSV

## Pages and navigation
- Home → overview of the flow and entry point
- JD → enter or edit the role criteria
- Upload → select PDF resumes and start processing
- Processing → shows batch progress
- Results → ranked candidates; drill into details or compare

## Interpreting scores
- Final score reflects a blend of lexical, semantic, and evidence-based matching
- Coverage emphasizes how many key requirements are met
- A strong candidate tends to have both broad coverage and good evidence in experience

## Tips for best results
- Keep the job description clear and specific; list must-have skills explicitly
- Prefer searchable/resident text PDFs (avoid scans); OCR is disabled by default for speed
- Batch sizes of 5–20 resumes are a good balance for iteration speed

## Export format
- JSON export: one entry per candidate with raw and normalized scores, final score, explanation
- CSV export: tabular view of key fields for spreadsheets/dashboards

## Troubleshooting
- If processing seems slow, check `resume_reviewer/debug.log` for timing lines `[TIMING]`
- Large or image-only PDFs take longer; try saving as text PDFs
- If a candidate shows as "Unknown", ensure the name appears near the top of the resume

## Notes
- This app focuses on ranking for a single role at a time
- Results are deterministic given the same inputs and configuration 