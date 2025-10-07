from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import json
from typing import List
from resume_processor.batch_processor import BatchProcessor

class Command(BaseCommand):
    help = 'Process a batch of resumes against a job description'

    def add_arguments(self, parser):
        parser.add_argument(
            '--resumes',
            nargs='+',
            type=str,
            required=True,
            help='Paths to resume PDF files (max 25)'
        )
        parser.add_argument(
            '--job-description',
            type=str,
            required=True,
            help='Job description text or path to PDF file'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='batch_results.json',
            help='Output JSON file path'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='OpenAI API key for LLM ranking'
        )

    def handle(self, *args, **options):
        resume_paths = options['resumes']
        job_description = options['job_description']
        output_path = options['output']
        api_key = options['api_key']

        # Validate inputs
        if len(resume_paths) > 25:
            raise CommandError(f"Maximum 25 resumes allowed, got {len(resume_paths)}")

        # Check if resume files exist
        for path in resume_paths:
            if not os.path.exists(path):
                raise CommandError(f"Resume file not found: {path}")

        # Load job description
        if os.path.exists(job_description):
            # It's a file path
            try:
                with open(job_description, 'r', encoding='utf-8') as f:
                    job_description_text = f.read()
            except Exception as e:
                raise CommandError(f"Error reading job description file: {e}")
        else:
            # It's the text itself
            job_description_text = job_description

        # Initialize batch processor
        processor = BatchProcessor(api_key=api_key)

        # Process batch
        self.stdout.write(f"Processing {len(resume_paths)} resumes...")
        results = processor.process_batch(resume_paths, job_description_text)

        # Check for errors
        if 'error' in results:
            raise CommandError(f"Batch processing failed: {results['error']}")

        # Save results
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.stdout.write(
                self.style.SUCCESS(f"Batch processing completed successfully. Results saved to {output_path}")
            )
            
            # Print summary
            self._print_summary(results)
            
        except Exception as e:
            raise CommandError(f"Error saving results: {e}")

    def _print_summary(self, results: dict):
        """Print a human-readable summary of the results."""
        self.stdout.write("\n" + "="*50)
        self.stdout.write("BATCH PROCESSING SUMMARY")
        self.stdout.write("="*50)
        
        # Resume count
        resume_count = len(results.get('resumes', []))
        self.stdout.write(f"Resumes processed: {resume_count}")
        
        # Top candidates
        top_candidates = results.get('batch_summary', {}).get('top_candidates', [])
        if top_candidates:
            self.stdout.write(f"Top candidates: {', '.join(top_candidates[:3])}")
        
        # Common gaps
        common_gaps = results.get('batch_summary', {}).get('common_gaps', [])
        if common_gaps:
            self.stdout.write(f"Common gaps: {', '.join(common_gaps)}")
        
        # Processing notes
        notes = results.get('batch_summary', {}).get('notes', '')
        if notes:
            self.stdout.write(f"Notes: {notes}")
        
        # Final ranking
        final_ranking = results.get('final_ranking', [])
        if final_ranking:
            self.stdout.write("\nTop 3 Rankings:")
            for rank in final_ranking[:3]:
                self.stdout.write(f"  {rank['rank']}. {rank['id']} - {rank['reasoning'][:100]}...")
        
        self.stdout.write("="*50) 