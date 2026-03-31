# System Diagrams

This document contains three core system diagrams for the Resume Ranking System:
an Entity-Relationship Diagram (ERD), a Dataflow Diagram (DFD), and a Use Case
Diagram.  All diagrams are written in Mermaid.js and reflect the actual source
code in this repository.

---

## 1. Entity-Relationship Diagram (ERD)

> **Source file:** `resume_reviewer/resume_processor/models.py`
>
> The system stores three first-class entities.  `RankingSession` holds a soft
> reference to `JobPosting` via the `job_id` string field (no database-level
> foreign key because the original schema targets SQLite).  Every resume that is
> uploaded and processed belongs conceptually to a `RankingSession`; the
> relationship is captured inside the `ranking_results_json` column of
> `RankingSession`, which is an array of `candidate_id` references.

```mermaid
erDiagram
    RESUME {
        int     id                  PK
        string  candidate_id        "unique"
        string  filename
        string  original_file_path
        datetime uploaded_at
        datetime processed_at
        string  processing_status   "pending | processing | completed | failed"
        text    error_message
        text    parsed_data_json    "raw sections extracted by PDFParser"
        text    processed_data_json "NLP-normalised text + token lists"
        text    ranking_data_json   "BM25 / SBERT / CE / final scores"
    }

    JOB_POSTING {
        int     id              PK
        string  job_id          "unique"
        string  title
        string  company
        text    description
        text    requirements_json    "must-have skills list (JSON)"
        text    nice_to_have_json    "nice-to-have skills list (JSON)"
        text    job_analysis_json    "parsed criteria dict (JSON)"
        datetime created_at
        string  status          "active | closed"
    }

    RANKING_SESSION {
        int     id                  PK
        string  session_id          "unique"
        string  job_id              "FK → JOB_POSTING.job_id (soft)"
        text    job_description
        int     candidates_count
        datetime created_at
        string  status              "processing | completed | failed"
        text    ranking_results_json "ordered list of candidate_id refs (JSON)"
        text    summary_json         "aggregate statistics (JSON)"
    }

    JOB_POSTING  ||--o{ RANKING_SESSION : "is evaluated in (soft ref via job_id)"
    RANKING_SESSION }o--o{ RESUME : "ranks (logical ref via ranking_results_json)"
```

---

## 2. Dataflow Diagram (DFD)

### Level 0 — Context Diagram

> The system receives job-description criteria and PDF résumés from the HR User
> and returns ranked candidate lists, individual profiles, and exportable reports.

```mermaid
flowchart LR
    HR([HR User\nRecruiter / Hiring Manager])

    subgraph SYS ["Resume Ranking System"]
        PROC[["0\nResume Ranking\nSystem"]]
    end

    HR -- "JD criteria\n(position, skills,\nexperience)" --> PROC
    HR -- "PDF résumé files\n(up to 25)" --> PROC
    PROC -- "Ranked candidate list\nwith scores & NLG summaries" --> HR
    PROC -- "Candidate detail\nprofiles" --> HR
    PROC -- "Exported results\n(JSON / CSV)" --> HR
```

---

### Level 1 — Decomposed DFD

> Each numbered process maps directly to a module in the repository.

```mermaid
flowchart TD
    HR([HR User])

    %% --- External data stores ---
    DS1[(D1 — Session Store\nDjango server-side session)]
    DS2[(D2 — Database\nSQLite via Django ORM\nResume · JobPosting\nRankingSession)]
    DS3[(D3 — File System\nMedia / temp_uploads\nPDF files)]

    %% ---- Process 1 ----
    HR -- "JD title, skills,\nexperience, seniority" --> P1["1\nCapture Job Description\njd_new() → jd_criteria.py\nbuild_jd_text()"]
    P1 -- "jd_criteria dict" --> DS1

    %% ---- Process 2 ----
    HR -- "PDF résumé\nfiles (≤ 25)" --> P2["2\nUpload &\nQueue Résumés\nresume_upload()\nasync_processor.py"]
    DS1 -- "jd_criteria" --> P2
    P2 -- "PDF bytes" --> DS3
    P2 -- "task_id\n+ temp paths" --> DS1
    P2 -- "Resume records\n(status = pending)" --> DS2

    %% ---- Process 3 ----
    DS3 -- "PDF files" --> P3["3\nParse PDFs\nBatchProcessor\n._parse_pdfs_to_json()\nPDFParser (fitz / pytesseract)\n→ ParsedResume dataclass"]
    DS1 -- "task_id" --> P3
    P3 -- "parsed_data\n(raw sections)" --> DS2

    %% ---- Process 4 ----
    P3 -- "ParsedResume list\n+ jd_normalized" --> P4["4\nNLP Ranking Pipeline\nHybridRanker.rank()\n① BM25 lexical score\n② S-BERT semantic score\n③ RRF fusion\n④ Cross-Encoder reranking"]
    P4 -- "ranking_data\n(BM25 · SBERT · CE\n· final_score)" --> DS2

    %% ---- Process 5 ----
    P4 -- "ranked candidates\n+ score breakdown" --> P5["5\nGenerate NLG Summaries\nEnhancedCandidateAnalyzer\nnlg_generator_enhanced.py\n→ nlg_summary per candidate"]
    P5 -- "batch_results dict" --> DS1
    P5 -- "RankingSession record\n(status = completed)" --> DS2

    %% ---- Process 6 ----
    DS1 -- "batch_results" --> P6["6\nPresent Results\nranking_list()\ncandidate_detail()\ncandidate_compare()"]
    P6 -- "Ranked list\n& profiles" --> HR

    %% ---- Process 7 ----
    DS1 -- "batch_results" --> P7["7\nExport Results\nexport_results_json()\nexport_results_csv()"]
    P7 -- "JSON / CSV\ndownload" --> HR

    %% ---- Process 8 (async status polling) ----
    DS1 -- "task_id" --> P8["8\nReport Progress\nbatch_progress()\nprocessing_status()"]
    P8 -- "progress %\n& status" --> HR
```

---

## 3. Use Case Diagram

> **Actor:** HR User (Recruiter / Hiring Manager) — the single human actor who
> interacts with the system.  The **NLP Pipeline** is shown as a secondary
> (system) actor to highlight that automated ranking is triggered internally.

```mermaid
graph LR
    %% Actors
    HR(["👤 HR User\n(Recruiter /\nHiring Manager)"])
    SYS(["⚙️ NLP Pipeline\n(System Actor)"])

    %% Use cases — grouped by subject boundary
    subgraph boundary ["Resume Ranking System"]

        subgraph jd ["Job Description Management"]
            UC1(["Create Job Description\n(title, skills, experience,\nseniority level)"])
        end

        subgraph upload ["Résumé Ingestion"]
            UC2(["Upload PDF Résumés\n(up to 25 files)"])
            UC3(["Monitor Processing\nStatus"])
        end

        subgraph ranking ["Ranking & Review"]
            UC4(["View Ranked\nCandidate List"])
            UC5(["View Candidate\nDetail Profile"])
            UC6(["Compare\nCandidates"])
        end

        subgraph export ["Reporting"]
            UC7(["Export Results\nas JSON"])
            UC8(["Export Results\nas CSV"])
        end

        subgraph auto ["Automated Processing (System)"]
            UC9(["Parse PDF\nRésumés"])
            UC10(["Run NLP Ranking\n(BM25 → SBERT →\nRRF → Cross-Encoder)"])
            UC11(["Generate NLG\nSummaries"])
        end

    end

    %% HR User associations
    HR --- UC1
    HR --- UC2
    HR --- UC3
    HR --- UC4
    HR --- UC5
    HR --- UC6
    HR --- UC7
    HR --- UC8

    %% System actor associations
    SYS --- UC9
    SYS --- UC10
    SYS --- UC11

    %% Include / extend relationships
    UC2 -.->|"«include»"| UC1
    UC3 -.->|"«include»"| UC2
    UC4 -.->|"«include»"| UC3
    UC5 -.->|"«extend»"| UC4
    UC6 -.->|"«extend»"| UC4
    UC7 -.->|"«include»"| UC4
    UC8 -.->|"«include»"| UC4
    UC2 -.->|"«triggers»"| UC9
    UC9 -.->|"«triggers»"| UC10
    UC10 -.->|"«triggers»"| UC11
```

**Use-case narrative summary**

| ID   | Use Case                  | Primary Actor | Pre-condition                        | Post-condition                              |
|------|---------------------------|---------------|--------------------------------------|---------------------------------------------|
| UC1  | Create Job Description    | HR User       | —                                    | JD criteria stored in session               |
| UC2  | Upload PDF Résumés        | HR User       | JD criteria exists in session        | PDFs saved; background task started         |
| UC3  | Monitor Processing Status | HR User       | Background task is running           | User sees real-time progress %              |
| UC4  | View Ranked Candidate List| HR User       | Processing completed                 | Ordered list with scores displayed          |
| UC5  | View Candidate Detail     | HR User       | Candidate exists in batch results    | Full profile + NLG summary shown            |
| UC6  | Compare Candidates        | HR User       | ≥ 2 candidates in results            | Side-by-side score comparison               |
| UC7  | Export Results as JSON    | HR User       | Batch results available              | `export.json` file downloaded               |
| UC8  | Export Results as CSV     | HR User       | Batch results available              | `export.csv` file downloaded                |
| UC9  | Parse PDF Résumés         | NLP Pipeline  | PDFs queued by UC2                   | `ParsedResume` objects created              |
| UC10 | Run NLP Ranking           | NLP Pipeline  | `ParsedResume` list ready            | Scores (BM25 / SBERT / RRF / CE) assigned   |
| UC11 | Generate NLG Summaries    | NLP Pipeline  | Ranking scores computed              | Natural-language rationale per candidate    |
