from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.utils import timezone
import os
import logging
from django.conf import settings
import json
from .models import Resume
from .enhanced_pdf_parser import PDFParser
from .batch_processor import BatchProcessor
from .jd_criteria import parse_criteria_from_post, build_jd_text
from .config import get_openai_config, get_llm_config, validate_config
from .async_processor import start_batch_processing_async, get_task

logger = logging.getLogger(__name__)

def home(request):
    """Home page - explains the flow and provides start button."""
    return render(request, 'resume_processor/home.html', {
        'show_nav': False
    })

def jd_new(request):
    """Job Description form page."""
    if request.method == 'POST':
        # Store JD criteria in session and redirect to upload
        jd_criteria = parse_criteria_from_post(request.POST)
        request.session['jd_criteria'] = jd_criteria
        return redirect('resume_upload')
    
    # Load last JD criteria if available (for testing)
    last_jd_criteria = None
    if settings.DEBUG:
        try:
            last_jd_file = os.path.join(settings.BASE_DIR, 'batch_processing_output', 'last_jd_criteria.json')
            if os.path.exists(last_jd_file):
                with open(last_jd_file, 'r', encoding='utf-8') as f:
                    last_jd_criteria = json.load(f)
        except Exception:
            pass
    
    return render(request, 'resume_processor/jd_new.html', {
        'last_jd_criteria': last_jd_criteria,
        'show_nav': True
    })

@csrf_exempt
def debug_upload(request):
    """Debug endpoint to test upload handling."""
    if request.method == 'POST':
        logger.info(f"Debug upload - Method: {request.method}")
        logger.info(f"Debug upload - Files: {list(request.FILES.keys())}")
        logger.info(f"Debug upload - POST data: {dict(request.POST)}")
        return JsonResponse({
            'method': request.method,
            'files': list(request.FILES.keys()),
            'post_data': dict(request.POST),
            'session_keys': list(request.session.keys())
        })
    return JsonResponse({'error': 'POST only'}, status=405)

@csrf_exempt
def resume_upload(request):
    """Upload resumes for the active JD."""
    # Check if JD exists
    if 'jd_criteria' not in request.session:
        messages.warning(request, 'Please add a job description first.')
        return redirect('jd_new')
    
    if request.method == 'POST':
        # Process the batch and return JSON response
        try:
            logger.info("Starting resume_upload POST processing")
            resume_files = request.FILES.getlist('resumes')
            jd_criteria = request.session.get('jd_criteria', {})
            
            logger.info(f"Received {len(resume_files)} resume files")
            logger.info(f"JD criteria: {jd_criteria}")
            
            if not resume_files:
                logger.warning("No resume files uploaded")
                return JsonResponse({'error': 'No resume files uploaded'}, status=400)
            
            if len(resume_files) > 25:
                logger.warning(f"Too many files: {len(resume_files)}")
                return JsonResponse({'error': 'Maximum 25 resumes allowed'}, status=400)
            
            # Synthesize job description from criteria
            job_description = ""
            try:
                synthesized = build_jd_text(jd_criteria)
                job_description = synthesized or ''
                logger.info(f"Synthesized JD: {job_description[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to synthesize JD from criteria: {e}")
            
            if not job_description:
                logger.warning("No job description available")
                return JsonResponse({'error': 'Job description/criteria is required'}, status=400)
            
            disable_ocr_flag = request.POST.get('disable_ocr', '') in ['1', 'true', 'on', 'yes']
            logger.info(f"OCR disabled: {disable_ocr_flag}")
            
            # Get API key from provider-agnostic configuration
            llm_cfg = get_llm_config()
            api_key = llm_cfg['api_key']
            logger.info(f"API key configured: {bool(api_key)}")
            
            # Check configuration status
            config_issues = validate_config()
            if config_issues:
                logger.warning("Configuration issues detected: %s", config_issues)
            
            # Save uploaded files to temporary paths
            temp_paths = []
            for resume_file in resume_files:
                file_path = os.path.join(settings.MEDIA_ROOT, 'temp_uploads', resume_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb+') as destination:
                    for chunk in resume_file.chunks():
                        destination.write(chunk)
                temp_paths.append(file_path)
                logger.info(f"Saved file: {file_path}")
            
            logger.info(f"Starting ASYNC batch processing with {len(temp_paths)} files")
            
            # Start async processing and return task ID immediately
            task_id = start_batch_processing_async(
                resume_paths=temp_paths,
                job_description=job_description,
                jd_criteria=jd_criteria,
                disable_ocr=disable_ocr_flag
            )
            
            # Store task ID in session for later retrieval
            request.session['current_task_id'] = task_id
            request.session['jd_criteria'] = jd_criteria
            request.session.modified = True
            
            # Persist JD criteria for testing
            if settings.DEBUG:
                try:
                    os.makedirs(os.path.join(settings.BASE_DIR, 'batch_processing_output'), exist_ok=True)
                    jd_file = os.path.join(settings.BASE_DIR, 'batch_processing_output', 'last_jd_criteria.json')
                    with open(jd_file, 'w', encoding='utf-8') as f:
                        json.dump(jd_criteria, f, indent=2)
                except Exception:
                    pass
            
            logger.info(f"Returning task ID: {task_id}")
            
            # Return task ID immediately (async processing continues in background)
            return JsonResponse({
                'task_id': task_id,
                'status': 'processing',
                'message': 'Batch processing started. Please wait...'
            })
                
        except Exception as e:
            logger.error(f"Error in resume_upload: {e}", exc_info=True)
            return JsonResponse({'error': f'Processing error: {str(e)}'}, status=500)
    
    return render(request, 'resume_processor/resume_upload.html', {
        'jd_criteria': request.session.get('jd_criteria', {}),
        'show_nav': True
    })

def batch_progress(request, task_id):
    """API endpoint to check batch processing progress."""
    task = get_task(task_id)
    
    if not task:
        return JsonResponse({'error': 'Task not found'}, status=404)
    
    response = task.to_dict()
    
    # If completed, store results in session
    if task.status == 'completed' and task.result:
        request.session['batch_results'] = task.result
        request.session.modified = True
    
    return JsonResponse(response)

def processing_status(request):
    """Shows batch progress with autorefresh."""
    # Check if task exists
    task_id = request.session.get('current_task_id')
    
    if not task_id:
        messages.warning(request, 'No batch processing in progress.')
        return redirect('resume_upload')
    
    task = get_task(task_id)
    
    if not task:
        messages.warning(request, 'Batch processing task not found.')
        return redirect('resume_upload')
    
    return render(request, 'resume_processor/processing_status.html', {
        'task_id': task_id,
        'task_status': task.to_dict(),
        'show_nav': True
    })

def ranking_list(request):
    """List of candidates with scores/rationales."""
    try:
        logger.info("Accessing ranking_list view")
        
        # Check if batch exists
        if 'batch_results' not in request.session:
            logger.warning("No batch_results in session")
            messages.warning(request, 'No batch results available. Please process resumes first.')
            return redirect('resume_upload')
        
        batch_results = request.session.get('batch_results', {})
        logger.info(f"Batch results keys: {list(batch_results.keys())}")
        
        candidates = batch_results.get('resumes', [])
        logger.info(f"Found {len(candidates)} candidates")
        
        # Add candidate names and friendly rationales
        for i, candidate in enumerate(candidates):
            try:
                candidate['display_name'] = get_candidate_display_name(candidate)
                candidate['friendly_rationale'] = candidate.get('scores', {}).get('rationale', 'Overall Match: Unable to assess')
                # Pre-calculate percentage for display
                final_score = candidate.get('scores', {}).get('final_score', 0.0)
                candidate['final_score_percentage'] = round(float(final_score) * 100, 1)
                logger.info(f"Candidate {i}: {candidate.get('display_name', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error processing candidate {i}: {e}")
                candidate['display_name'] = f'Candidate {i+1}'
                candidate['friendly_rationale'] = 'Overall Match: Unable to assess'
                candidate['final_score_percentage'] = 0.0
        
        batch_info = batch_results.get('job_description_digest', {})
        logger.info(f"Batch info keys: {list(batch_info.keys())}")
        
        # Ensure criteria exists for template compatibility
        if 'criteria' not in batch_info:
            batch_info['criteria'] = {}
        
        return render(request, 'resume_processor/ranking_list.html', {
            'candidates': candidates,
            'batch_info': batch_info,
            'show_nav': True
        })
        
    except Exception as e:
        logger.error(f"Error in ranking_list: {e}", exc_info=True)
        messages.error(request, f'Error loading results: {str(e)}')
        return redirect('resume_upload')

def _get_current_batch_for_export(request):
    """Helper to fetch the current batch results from session for export."""
    if 'batch_results' not in request.session:
        return None
    batch_results = request.session.get('batch_results', {})
    # Expecting processor result schema
    if not isinstance(batch_results, dict) or 'resumes' not in batch_results:
        return None
    return batch_results

def export_results_json(request):
    """Download current batch results as a JSON file."""
    batch = _get_current_batch_for_export(request)
    if batch is None:
        messages.warning(request, 'No batch results available to export.')
        return redirect('resume_upload')

    # Produce simplified export shape for easier debugging
    try:
        simplified = {
            'job': batch.get('job_description_digest', {}),
            'candidates': []
        }
        for c in batch.get('resumes', []):
            s = c.get('scores', {}) or {}
            simplified['candidates'].append({
                'id': c.get('id'),
                'name': get_candidate_display_name(c),
                'raw_scores': {
                    'tfidf_combined': float(s.get('combined_tfidf', 0.0) or 0.0),
                    'sbert_raw': float(s.get('sbert_score', s.get('semantic_score', 0.0)) or 0.0),
                    'ce_raw': float(s.get('ce_score', 0.0) or 0.0),
                    'coverage': s.get('coverage', 0.0),
                    'has_match_experience': bool(s.get('has_match_experience', False)),
                    'matched_required_skills': s.get('matched_required_skills', []) or []
                },
                'normalized_scores': {
                    'tfidf_norm': float(s.get('tfidf_norm', 0.0) or 0.0),
                    'semantic_norm': float(s.get('semantic_norm', 0.0) or 0.0),
                    'ce_norm': float(s.get('ce_norm', 0.0) or 0.0)
                },
                'final_score': float(s.get('final_score', 0.0) or 0.0),
                'explanation': s.get('rationale', 'Overall Match: Unable to assess')
            })
        data_bytes = json.dumps(simplified, ensure_ascii=False, indent=2).encode('utf-8')
    except Exception:
        # Fallback to full batch on any error
        data_bytes = json.dumps(batch, ensure_ascii=False, indent=2).encode('utf-8')
    response = HttpResponse(data_bytes, content_type='application/json; charset=utf-8')
    filename = 'results.json'
    try:
        jd = batch.get('job_description_digest', {})
        title = (jd.get('criteria', {}) or {}).get('position_title') or jd.get('title') or 'results'
        filename = f"{title.replace(' ', '_').lower()}_results.json"
    except Exception:
        pass
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

def export_results_csv(request):
    """Download current batch results as a CSV file."""
    import csv
    import io

    batch = _get_current_batch_for_export(request)
    if batch is None:
        messages.warning(request, 'No batch results available to export.')
        return redirect('resume_upload')

    resumes = batch.get('resumes', [])
    jd = batch.get('job_description_digest', {})
    criteria = jd.get('criteria', {}) if isinstance(jd, dict) else {}
    must_have = criteria.get('must_have_skills', []) or []

    output = io.StringIO()
    writer = csv.writer(output)
    # Header
    writer.writerow([
        'id', 'display_name', 'final_score', 'final_score_%',
        'sbert_score', 'ce_score', 'tfidf_section', 'tfidf_skill',
        'coverage', 'gate_threshold', 'has_match_skills', 'has_match_experience',
        'matched_required_count', 'must_have_total', 'missing_required', 'rationale'
    ])

    for c in resumes:
        scores = c.get('scores', {}) or {}
        display_name = get_candidate_display_name(c)
        final_score = float(scores.get('final_score', 0.0) or 0.0)
        final_pct = round(final_score * 100, 2)
        sbert = float(scores.get('sbert_score', scores.get('semantic_score', 0.0)) or 0.0)
        ce = float(scores.get('ce_score', 0.0) or 0.0)
        tfidf_sec = float(scores.get('tfidf_section_score', 0.0) or 0.0)
        tfidf_skill = float(scores.get('tfidf_skill_score', 0.0) or 0.0)
        coverage = scores.get('coverage', None)
        threshold = scores.get('gate_threshold', None)
        has_ms = 1 if (scores.get('has_match_skills') or False) else 0
        has_exp = 1 if (scores.get('has_match_experience') or False) else 0
        matched_required = scores.get('matched_required_skills', []) or []
        missing = [s for s in must_have if s not in matched_required] if must_have else []
        rationale = scores.get('rationale', '')

        writer.writerow([
            c.get('id'), display_name, f"{final_score:.6f}", f"{final_pct:.2f}",
            f"{sbert:.6f}", f"{ce:.6f}", f"{tfidf_sec:.6f}", f"{tfidf_skill:.6f}",
            '' if coverage is None else coverage, '' if threshold is None else threshold,
            has_ms, has_exp, len(matched_required), len(must_have), '|'.join(missing), rationale
        ])

    data = output.getvalue().encode('utf-8')
    response = HttpResponse(data, content_type='text/csv; charset=utf-8')
    filename = 'results.csv'
    try:
        title = (criteria or {}).get('position_title') or jd.get('title') or 'results'
        filename = f"{title.replace(' ', '_').lower()}_results.csv"
    except Exception:
        pass
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

def candidate_detail(request, candidate_id):
    """Detail view for one candidate."""
    # Check if batch exists
    if 'batch_results' not in request.session:
        messages.warning(request, 'No batch results available.')
        return redirect('resume_upload')
    
    batch_results = request.session.get('batch_results', {})
    candidates = batch_results.get('resumes', [])
    
    # Find candidate by ID
    candidate = None
    for c in candidates:
        if c.get('id') == candidate_id:
            candidate = c
            break
    
    if not candidate:
        messages.error(request, 'Candidate not found.')
        return redirect('ranking_list')
    
    candidate['display_name'] = get_candidate_display_name(candidate)
    candidate['friendly_rationale'] = candidate.get('scores_snapshot', {}).get('rationale', 'Overall Match: Unable to assess')
    
    return render(request, 'resume_processor/candidate_detail.html', {
        'candidate': candidate,
        'show_nav': True
    })

def candidate_compare(request):
    """Compare 2-3 candidates side by side."""
    # Check if batch exists
    if 'batch_results' not in request.session:
        messages.warning(request, 'No batch results available.')
        return redirect('resume_upload')
    
    batch_results = request.session.get('batch_results', {})
    candidates = batch_results.get('resumes', [])
    
    # Get candidate IDs from query params
    candidate_ids = request.GET.getlist('ids')
    if not candidate_ids or len(candidate_ids) < 2:
        messages.warning(request, 'Please select at least 2 candidates to compare.')
        return redirect('ranking_list')
    
    # Limit to 3 candidates
    candidate_ids = candidate_ids[:3]
    
    # Find candidates
    compare_candidates = []
    for cid in candidate_ids:
        for c in candidates:
            if c.get('id') == int(cid):
                c['display_name'] = get_candidate_display_name(c)
                c['friendly_rationale'] = c.get('scores_snapshot', {}).get('rationale', 'Overall Match: Unable to assess')
                compare_candidates.append(c)
                break
    
    if len(compare_candidates) < 2:
        messages.error(request, 'Could not find selected candidates.')
        return redirect('ranking_list')
    
    return render(request, 'resume_processor/candidate_compare.html', {
        'candidates': compare_candidates,
        'show_nav': True
    })

def get_candidate_display_name(candidate):
    """Get display name for candidate (parsed name, inferred from filename, or filename)."""
    # Try to get parsed name from candidate data
    parsed_data = candidate.get('parsed', {})
    parsed_name = parsed_data.get('candidate_name') or parsed_data.get('parsed_name') or parsed_data.get('name') or candidate.get('parsed_name') or candidate.get('name')
    if parsed_name and parsed_name.strip():
        return parsed_name.strip()
    
    # Infer name from filename
    filename = candidate.get('meta', {}).get('source_file', '') or candidate.get('filename', '')
    if filename:
        # Remove extension and clean up
        name = os.path.splitext(filename)[0]
        # Replace underscores with spaces and title case
        name = name.replace('_', ' ').replace('-', ' ')
        # Title case
        name = ' '.join(word.capitalize() for word in name.split())
        if name.strip():
            return name.strip()
    
    # Fallback to filename or generic name
    return filename or 'Unknown Candidate'

@csrf_exempt
def landing_page(request):
    """Landing page view - main entry point for the website."""
    if request.method == 'POST':
        try:
            resume_files = request.FILES.getlist('resumes')
            # Collect criteria first and synthesize JD text (criteria becomes primary)
            jd_criteria = {}
            try:
                jd_criteria = parse_criteria_from_post(request.POST)
            except Exception as e:
                logger.warning(f"Failed to parse JD criteria: {e}")
            # If raw JD is provided, keep it as a fallback; otherwise synthesize from criteria
            job_description = request.POST.get('job_description', '')
            if not job_description.strip():
                try:
                    synthesized = build_jd_text(jd_criteria)
                    job_description = synthesized or ''
                except Exception as e:
                    logger.warning(f"Failed to synthesize JD from criteria: {e}")
            disable_ocr_flag = request.POST.get('disable_ocr', '') in ['1', 'true', 'on', 'yes']
            
            if not resume_files:
                return JsonResponse({'error': 'No resume files uploaded'}, status=400)
            
            if not job_description:
                return JsonResponse({'error': 'Job description/criteria is required'}, status=400)
            
            if len(resume_files) > 25:
                return JsonResponse({'error': 'Maximum 25 resumes allowed'}, status=400)
            
            # Get API key from provider-agnostic configuration
            llm_cfg = get_llm_config()
            api_key = llm_cfg['api_key']
            
            # Check configuration status
            config_issues = validate_config()
            if config_issues:
                logger.warning("Configuration issues detected: %s", config_issues)
            
            temp_paths = []
            for resume_file in resume_files:
                file_path = os.path.join(settings.MEDIA_ROOT, 'temp_uploads', resume_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb+') as destination:
                    for chunk in resume_file.chunks():
                        destination.write(chunk)
                temp_paths.append(file_path)
            
            # Initialize batch processor with API key
            processor = BatchProcessor(api_key=api_key, disable_ocr=disable_ocr_flag)
            
            # Process the batch
            results = processor.process_batch(temp_paths, job_description, jd_criteria=jd_criteria)
            
            # Store batch results in session for ranking view
            if 'batch_results' not in request.session:
                request.session['batch_results'] = []
            
            # Add current batch results with timestamp
            batch_info = {
                'timestamp': timezone.now().isoformat(),
                'job_description': job_description,
                'resume_count': len(resume_files),
                'results': results,
                'filenames': [f.name for f in resume_files],
                'disable_ocr': disable_ocr_flag,
                'jd_criteria': jd_criteria
            }
            request.session['batch_results'].append(batch_info)
            # Ensure Django persists in-place session mutations
            request.session.modified = True
            
            # Keep only last 5 batches to avoid session bloat
            if len(request.session['batch_results']) > 5:
                request.session['batch_results'] = request.session['batch_results'][-5:]
                request.session.modified = True
            
            # Persist last JD and criteria for development convenience (DEBUG only)
            try:
                if settings.DEBUG:
                    cache_dir = os.path.join(settings.BASE_DIR, 'batch_processing_output')
                    os.makedirs(cache_dir, exist_ok=True)
                    last_jd_path = os.path.join(cache_dir, 'last_job_description.txt')
                    with open(last_jd_path, 'w', encoding='utf-8') as f:
                        f.write(job_description or '')
                    # Persist criteria as JSON
                    last_criteria_path = os.path.join(cache_dir, 'last_jd_criteria.json')
                    try:
                        with open(last_criteria_path, 'w', encoding='utf-8') as cf:
                            json.dump(jd_criteria or {}, cf, ensure_ascii=False, indent=2)
                    except Exception as e2:
                        logger.warning(f"Failed to persist last criteria: {e2}")
            except Exception as e:
                logger.warning(f"Failed to persist last JD: {e}")

            # Clean up temporary files
            for path in temp_paths:
                try:
                    os.remove(path)
                except:
                    pass
            
            return JsonResponse(results)
            
        except Exception as e:
            logger.error(f"Error in batch upload: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    # GET request - show the landing page
    llm_cfg = get_llm_config()
    config_issues = validate_config()
    # Load last JD/criteria for development convenience (DEBUG only)
    last_job_description = ''
    last_jd_criteria = {}
    try:
        if settings.DEBUG:
            last_jd_path = os.path.join(settings.BASE_DIR, 'batch_processing_output', 'last_job_description.txt')
            if os.path.exists(last_jd_path):
                with open(last_jd_path, 'r', encoding='utf-8') as f:
                    last_job_description = f.read()
            last_criteria_path = os.path.join(settings.BASE_DIR, 'batch_processing_output', 'last_jd_criteria.json')
            if os.path.exists(last_criteria_path):
                try:
                    with open(last_criteria_path, 'r', encoding='utf-8') as cf:
                        last_jd_criteria = json.load(cf) or {}
                except Exception as e2:
                    logger.warning(f"Failed to load last criteria: {e2}")
    except Exception as e:
        logger.warning(f"Failed to load last JD: {e}")
    
    context = {
        'api_key_configured': bool(llm_cfg['api_key']),
        'config_issues': config_issues,
        'llm_provider': llm_cfg.get('provider', 'openai'),
        'llm_model': llm_cfg.get('model', ''),
        'llm_enabled': llm_cfg.get('enabled', False),
        'last_job_description': last_job_description,
        'last_jd_criteria': last_jd_criteria,
    }
    
    return render(request, 'resume_processor/landing_page.html', context)

def upload_resume(request):
    """View for uploading resume PDF files."""
    if request.method == 'POST':
        if 'resume_file' in request.FILES:
            resume_file = request.FILES['resume_file']
            
            # Check if file is a PDF
            if not resume_file.name.lower().endswith('.pdf'):
                messages.error(request, 'Please upload a PDF file.')
                return redirect('resume_processor:upload_resume')
            
            # Save the uploaded file
            media_dir = os.path.join(settings.MEDIA_ROOT, 'resumes')
            os.makedirs(media_dir, exist_ok=True)
            
            file_path = os.path.join(media_dir, resume_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in resume_file.chunks():
                    destination.write(chunk)
            
            # Create Resume instance
            resume = Resume(
                filename=resume_file.name,
                original_file_path=file_path,
                candidate_id=os.path.splitext(resume_file.name)[0],
                processing_status='processing'
            )
            resume.save()
            
            # Process the PDF and extract to JSON
            pdf_parser = PDFParser()
            try:
                json_file_path = pdf_parser.extract_to_json(file_path, resume.filename)
                
                # Load structured data and store in parsed_data field
                import json
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    structured_data = json.load(f)
                
                resume.parsed_data = structured_data
                resume.processing_status = 'completed'
                resume.processed_at = timezone.now()
                resume.save()
                messages.success(request, f'Resume "{resume.filename}" processed successfully!')
            except Exception as e:
                resume.processing_status = 'failed'
                resume.error_message = str(e)
                resume.save()
                messages.error(request, f'Failed to process resume "{resume.filename}": {str(e)}')
            return redirect('resume_processor:resume_list')
    return render(request, 'resume_processor/upload.html', {'show_nav': True})

def resume_list(request):
    """View for listing all processed resumes."""
    resumes = Resume.objects.all()
    return render(request, 'resume_processor/resume_list.html', {'resumes': resumes, 'show_nav': True})

def resume_detail(request, resume_id):
    """View for showing details of a specific resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        return render(request, 'resume_processor/resume_detail.html', {'resume': resume, 'show_nav': True})
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def resume_structured(request, resume_id):
    """View for showing structured analysis of a specific resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        
        # Load structured data if available
        structured_data = resume.parsed_data if resume.parsed_data else None
        
        return render(request, 'resume_processor/resume_structured.html', {
            'resume': resume,
            'structured_data': structured_data,
            'show_nav': True
        })
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def resume_json(request, resume_id):
    """View for showing raw JSON data for ML model input."""
    try:
        resume = Resume.objects.get(id=resume_id)
        
        # Load structured data if available
        structured_data = resume.parsed_data if resume.parsed_data else None
        structured_data_json = None
        if structured_data:
            try:
                import json
                from .enhanced_pdf_parser import PDFParser
                pdf_parser = PDFParser()
                cleaned_data = pdf_parser.clean_json(structured_data)
                structured_data_json = json.dumps(cleaned_data, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error processing structured data: {str(e)}")
        
        return render(request, 'resume_processor/resume_json.html', {
            'resume': resume,
            'structured_data': structured_data,
            'structured_data_json': structured_data_json,
            'show_nav': True
        })
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')

def download_json(request, resume_id):
    """View to download the cleaned JSON file for a resume."""
    try:
        resume = Resume.objects.get(id=resume_id)
        if resume.parsed_data:
            import json
            from .enhanced_pdf_parser import PDFParser
            pdf_parser = PDFParser()
            cleaned_data = pdf_parser.clean_json(resume.parsed_data)
            import io
            cleaned_json_bytes = io.BytesIO(json.dumps(cleaned_data, indent=2, ensure_ascii=False).encode('utf-8'))
            response = FileResponse(cleaned_json_bytes)
            response['Content-Type'] = 'application/json'
            response['Content-Disposition'] = f'attachment; filename="{resume.filename.replace(".pdf", "_structured.json")}"'
            return response
        else:
            messages.error(request, 'Resume data not found.')
            return redirect('resume_processor:resume_detail', resume_id=resume_id)
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
        return redirect('resume_processor:resume_list')
    except Exception as e:
        messages.error(request, f'Error downloading JSON: {str(e)}')
        return redirect('resume_processor:resume_detail', resume_id=resume_id)

def delete_resume(request, resume_id):
    """View to delete a processed resume and its files."""
    try:
        resume = Resume.objects.get(id=resume_id)
        # Delete associated files if they exist
        if resume.original_file_path and os.path.exists(resume.original_file_path):
            os.remove(resume.original_file_path)
        # Clean up JSON file if it exists
        json_file_path = os.path.join(settings.MEDIA_ROOT, 'parsed_resumes', f"{resume.candidate_id}.json")
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        # Optionally, delete extracted text file if you save it separately
        # if resume.text_file_path and os.path.exists(resume.text_file_path):
        #     os.remove(resume.text_file_path)
        resume.delete()
        messages.success(request, 'Resume deleted successfully.')
    except Resume.DoesNotExist:
        messages.error(request, 'Resume not found.')
    except Exception as e:
        messages.error(request, f'Error deleting resume: {str(e)}')
    return redirect('resume_processor:resume_list')

@csrf_exempt
def process_resume_api(request, resume_id):
    """API endpoint to process a resume."""
    if request.method == 'POST':
        try:
            resume = Resume.objects.get(id=resume_id)
            pdf_parser = PDFParser()
            resume.processing_status = 'processing'
            resume.save()
            json_file_path = pdf_parser.extract_to_json(resume.original_file_path, resume.filename)
            # Load and store structured data
            import json
            with open(json_file_path, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
            resume.parsed_data = structured_data
            resume.processing_status = 'completed'
            resume.processed_at = timezone.now()
            resume.save()
            return JsonResponse({
                'success': True,
                'status': resume.processing_status,
                'error_message': resume.error_message
            })
        except Resume.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Resume not found'}, status=404)
        except Exception as e:
            resume.processing_status = 'failed'
            resume.error_message = str(e)
            resume.save()
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def ranking_view(request):
    """View for showing resume rankings and scores."""
    try:
        # Get the most recent batch results from session
        batch_results = request.session.get('batch_results', [])
        
        if not batch_results:
            # No batch results, show empty state
            context = {
                'ranked_resumes': [],
                'show_nav': True,
                'total_resumes': 0,
                'ranked_count': 0,
                'batch_info': None,
                'no_batches': True
            }
            return render(request, 'resume_processor/ranking.html', context)
        
        # Get the most recent batch
        latest_batch = batch_results[-1]
        batch_results_data = latest_batch['results']
        
        
        # Check if there was an error in processing
        if 'error' in batch_results_data:
            context = {
                'ranked_resumes': [],
                'show_nav': True,
                'total_resumes': 0,
                'ranked_count': 0,
                'batch_info': latest_batch,
                'processing_error': batch_results_data['error']
            }
            return render(request, 'resume_processor/ranking.html', context)
        
        # Process the batch results to create ranking display
        ranked_resumes = []
        resumes_data = batch_results_data.get('resumes', [])
        final_ranking = batch_results_data.get('final_ranking', [])
        
        # Create a mapping of resume ID to ranking info
        ranking_map = {r['id']: r for r in final_ranking}
        
        for resume_data in resumes_data:
            resume_id = resume_data['id']
            ranking_info = ranking_map.get(resume_id, {})
            
            # Derive a display name (prefer parsed name; fallback to prettified filename)
            meta = resume_data.get('meta', {})
            parsed = resume_data.get('parsed', {})
            source_file = meta.get('source_file', 'Unknown.pdf')
            base_name = os.path.splitext(source_file)[0]
            pretty_base = base_name.replace('_', ' ').replace('-', ' ').strip()
            pretty_base = pretty_base.title() if pretty_base else base_name

            candidate_name = None
            if isinstance(parsed, dict):
                candidate_name = parsed.get('name') or (
                    isinstance(parsed.get('profile'), dict) and parsed.get('profile', {}).get('name')
                )
            candidate_name = candidate_name or meta.get('candidate_name')
            
            # If no name found in parsed data, try to extract from resume sections
            if not candidate_name:
                sections = resume_data.get('sections', [])
                for section in sections:
                    content = section.get('content', [])
                    if content:
                        # Look for lines that look like names (2-4 capitalized words)
                        for line in content[:3]:  # Check first 3 lines of each section
                            line = line.strip()
                            if line and not line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                                words = line.split()
                                if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word and word[0].isalpha()):
                                    candidate_name = line
                                    break
                    if candidate_name:
                        break
            
            display_name = candidate_name if candidate_name else pretty_base
            
            scores_obj = resume_data.get('scores', {})
            default_score = (
                scores_obj.get('final_pre_llm_display')
                or scores_obj.get('semantic_score')
                or scores_obj.get('tfidf_section_score')
                or 0
            )

            ranking_display = {
                'resume_id': resume_id,
                'display_name': display_name,
                'candidate_name': display_name,  # Add candidate_name field
                'has_ranking': resume_id in ranking_map,
                'ranking_score': float(default_score) if default_score is not None else 0.0,
                'ranking_reason': None,
                'rank': None,
                'scores': resume_data.get('scores', {}),
                'matched_skills': resume_data.get('matched_skills', []),
                'parsed': resume_data.get('parsed', {}),
                'meta': resume_data.get('meta', {})
            }
            
            if ranking_info:
                snap = ranking_info.get('scores_snapshot', {})
                # Prefer final_pre_llm_display if present, else semantic, else keep default
                ranking_display['ranking_score'] = snap.get('final_pre_llm_display') or snap.get('semantic') or ranking_display['ranking_score']
                ranking_display['ranking_reason'] = ranking_info.get('reasoning', '')
                ranking_display['rank'] = ranking_info.get('rank', 0)
            
            ranked_resumes.append(ranking_display)
        
        # Sort by rank if available, otherwise by score
        ranked_resumes.sort(
            key=lambda x: (x['rank'] if x['rank'] is not None else 0, 
                          x['ranking_score'] if x['ranking_score'] is not None else 0),
            reverse=False  # Lower rank numbers first
        )
        
        context = {
            'ranked_resumes': ranked_resumes,
            'show_nav': True,
            'total_resumes': len(ranked_resumes),
            'ranked_count': sum(1 for r in ranked_resumes if r['has_ranking']),
            'batch_info': latest_batch,
            'no_batches': False,
            'processing_error': None
        }
        
        return render(request, 'resume_processor/ranking.html', context)
        
    except Exception as e:
        logger.error(f"Error in ranking view: {str(e)}")
        messages.error(request, f'Error loading rankings: {str(e)}')
        return redirect('resume_processor:resume_list')

def all_batches_view(request):
    """View for showing all batch processing results."""
    try:
        batch_results = request.session.get('batch_results', [])
        
        # Process batch results for display
        processed_batches = []
        for i, batch in enumerate(reversed(batch_results)):  # Show newest first
            batch_info = {
                'index': len(batch_results) - i,
                'timestamp': batch['timestamp'],
                'job_description': batch['job_description'],
                'resume_count': batch['resume_count'],
                'filenames': batch['filenames'],
                'has_error': 'error' in batch['results'],
                'error_message': batch['results'].get('error') if 'error' in batch['results'] else None,
                'top_candidates': batch['results'].get('batch_summary', {}).get('top_candidates', []) if 'error' not in batch['results'] else []
            }
            processed_batches.append(batch_info)
        
        context = {
            'batches': processed_batches,
            'show_nav': True,
            'total_batches': len(processed_batches)
        }
        
        return render(request, 'resume_processor/all_batches.html', context)
        
    except Exception as e:
        logger.error(f"Error in all batches view: {str(e)}")
        messages.error(request, f'Error loading batch history: {str(e)}')
        return redirect('resume_processor:resume_list')

@csrf_exempt
def deterministic_ranking_api(request):
    """API endpoint for deterministic resume ranking."""
    if request.method == 'POST':
        try:
            from .deterministic_ranker import process_ranking_payload
            
            # Get JSON payload from request body
            payload_json = request.body.decode('utf-8')
            
            # Process ranking
            result_json = process_ranking_payload(payload_json)
            
            # Return JSON response
            return HttpResponse(result_json, content_type='application/json')
            
        except Exception as e:
            logger.error(f"Error in deterministic ranking: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
