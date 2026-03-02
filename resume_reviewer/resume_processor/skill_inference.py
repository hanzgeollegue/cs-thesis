import re
from typing import Dict, List, Any, Tuple

# Section weights per spec
SECTION_WEIGHTS: Dict[str, float] = {
    'skills': 1.00,
    'experience': 1.00,
    'projects': 0.95,
    'education': 0.75,
    'misc': 0.60,
}

# Confidence buckets
BUCKET_CONF: Dict[str, float] = {
    'explicit': 1.00,
    'strong': 0.90,
    'weak': 0.70,
    'neg': 0.0,
}

# Comprehensive skill inference rules
RULES: Dict[str, Dict[str, List[re.Pattern]]] = {
    'git': {
        'explicit': [re.compile(r"\b(git|github|gitlab|bitbucket)\b", re.I)],
        'strong': [re.compile(r"\b(pull requests?|merge requests?|branch(?:ing)?|commit|github actions?|gitlab ci)\b", re.I)],
        'weak': [re.compile(r"\b(version control|code review|collaborative project)\b", re.I)],
        'neg': [re.compile(r"\b(svn|tfs)\b", re.I)],
    },
    'java': {
        'explicit': [re.compile(r"\bjava\b", re.I)],
        'strong': [re.compile(r"\b(spring(?: boot)?|maven|gradle|jpa|servlet|tomcat|junit|jvm|\.java|android)\b", re.I)],
        'weak': [re.compile(r"\b(oop|enterprise java|jakarta ee)\b", re.I)],
        'neg': [],
    },
    'python': {
        'explicit': [re.compile(r"\bpython\b", re.I)],
        'strong': [re.compile(r"\b(django|flask|fastapi|pandas|numpy|scikit-learn|tensorflow|pytorch|jupyter|pip|conda|\.py)\b", re.I)],
        'weak': [re.compile(r"\b(scripting|automation|data analysis|machine learning)\b", re.I)],
        'neg': [],
    },
    'javascript': {
        'explicit': [re.compile(r"\b(javascript|js|ecmascript)\b", re.I)],
        'strong': [re.compile(r"\b(react|angular|vue|node\.?js|express|jquery|typescript|es6|es2015)\b", re.I)],
        'weak': [re.compile(r"\b(frontend|front-end|web development|client-side)\b", re.I)],
        'neg': [],
    },
    'react': {
        'explicit': [re.compile(r"\b(react|reactjs|react\.js)\b", re.I)],
        'strong': [re.compile(r"\b(jsx|redux|hooks|component|react native|next\.js)\b", re.I)],
        'weak': [re.compile(r"\b(frontend|ui|user interface)\b", re.I)],
        'neg': [],
    },
    'sql': {
        'explicit': [re.compile(r"\b(sql|mysql|postgresql|postgres|sqlite|oracle|mssql)\b", re.I)],
        'strong': [re.compile(r"\b(database|db|query|table|join|select|insert|update|delete)\b", re.I)],
        'weak': [re.compile(r"\b(data storage|data management|relational)\b", re.I)],
        'neg': [],
    },
    'docker': {
        'explicit': [re.compile(r"\bdocker\b", re.I)],
        'strong': [re.compile(r"\b(container|containerization|dockerfile|docker compose|kubernetes|k8s)\b", re.I)],
        'weak': [re.compile(r"\b(devops|deployment|microservices)\b", re.I)],
        'neg': [],
    },
    'aws': {
        'explicit': [re.compile(r"\b(aws|amazon web services)\b", re.I)],
        'strong': [re.compile(r"\b(ec2|s3|lambda|rds|cloudformation|iam|vpc|route53)\b", re.I)],
        'weak': [re.compile(r"\b(cloud|cloud computing|infrastructure)\b", re.I)],
        'neg': [],
    },
    'kubernetes': {
        'explicit': [re.compile(r"\b(kubernetes|k8s)\b", re.I)],
        'strong': [re.compile(r"\b(pod|deployment|service|ingress|helm|minikube)\b", re.I)],
        'weak': [re.compile(r"\b(container orchestration|microservices)\b", re.I)],
        'neg': [],
    },
    'typescript': {
        'explicit': [re.compile(r"\b(typescript|ts)\b", re.I)],
        'strong': [re.compile(r"\b(interface|type|generic|decorator|\.ts)\b", re.I)],
        'weak': [re.compile(r"\b(typed javascript|type safety)\b", re.I)],
        'neg': [],
    },
    'nodejs': {
        'explicit': [re.compile(r"\b(node\.?js|nodejs)\b", re.I)],
        'strong': [re.compile(r"\b(express|npm|yarn|server-side|backend|api)\b", re.I)],
        'weak': [re.compile(r"\b(javascript runtime|server-side)\b", re.I)],
        'neg': [],
    },
    'mongodb': {
        'explicit': [re.compile(r"\b(mongodb|mongo)\b", re.I)],
        'strong': [re.compile(r"\b(document database|nosql|aggregation|mongoose)\b", re.I)],
        'weak': [re.compile(r"\b(database|data storage)\b", re.I)],
        'neg': [],
    },
    'redis': {
        'explicit': [re.compile(r"\bredis\b", re.I)],
        'strong': [re.compile(r"\b(cache|caching|in-memory|key-value)\b", re.I)],
        'weak': [re.compile(r"\b(database|data storage)\b", re.I)],
        'neg': [],
    },
    'ci/cd': {
        'explicit': [re.compile(r"\b(ci/cd|continuous integration|continuous deployment)\b", re.I)],
        'strong': [re.compile(r"\b(jenkins|github actions|gitlab ci|azure devops|travis|circleci)\b", re.I)],
        'weak': [re.compile(r"\b(automation|pipeline|deployment)\b", re.I)],
        'neg': [],
    },
    'microservices': {
        'explicit': [re.compile(r"\b(microservices|micro service)\b", re.I)],
        'strong': [re.compile(r"\b(api gateway|service mesh|distributed system|soa)\b", re.I)],
        'weak': [re.compile(r"\b(architecture|system design)\b", re.I)],
        'neg': [],
    },
    'api': {
        'explicit': [re.compile(r"\b(api|apis)\b", re.I)],
        'strong': [re.compile(r"\b(rest|graphql|soap|endpoint|http|json|xml)\b", re.I)],
        'weak': [re.compile(r"\b(integration|web service)\b", re.I)],
        'neg': [],
    },
    'linux': {
        'explicit': [re.compile(r"\b(linux|ubuntu|centos|debian|red hat)\b", re.I)],
        'strong': [re.compile(r"\b(bash|shell|terminal|command line|unix)\b", re.I)],
        'weak': [re.compile(r"\b(operating system|server)\b", re.I)],
        'neg': [],
    },
    'html': {
        'explicit': [re.compile(r"\b(html|html5)\b", re.I)],
        'strong': [re.compile(r"\b(semantic|accessibility|seo|markup)\b", re.I)],
        'weak': [re.compile(r"\b(web development|frontend)\b", re.I)],
        'neg': [],
    },
    'css': {
        'explicit': [re.compile(r"\b(css|css3)\b", re.I)],
        'strong': [re.compile(r"\b(flexbox|grid|sass|scss|bootstrap|tailwind)\b", re.I)],
        'weak': [re.compile(r"\b(styling|frontend|design)\b", re.I)],
        'neg': [],
    },
}

def _max_bucket_conf(text: str, rules: Dict[str, List[re.Pattern]]) -> Tuple[str, float, str]:
    """Return (bucket, confidence, evidence) for the highest-confidence match in text."""
    best: Tuple[str, float, str] = ('', 0.0, '')
    for bucket in ('explicit', 'strong', 'weak'):
        for pat in rules.get(bucket, []):
            m = pat.search(text)
            if m:
                conf = BUCKET_CONF[bucket]
                if conf > best[1]:
                    best = (bucket, conf, m.group(0))
    # Negative evidence cancels
    for pat in rules.get('neg', []):
        if pat.search(text):
            return ('neg', 0.0, '')
    return best

def infer_required_skills(required_skills: List[str], parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Infer required skills from parsed resume sections.
    Returns list of {skill, confidence, section, evidence}.
    """
    details: List[Dict[str, Any]] = []
    # Compose per-section text
    sections: Dict[str, str] = {
        'skills': ' '.join([s.get('skill', s) if isinstance(s, dict) else str(s) for s in parsed.get('skills', [])]),
        'experience': ' '.join([' '.join(e.get('bullets', []) or []) for e in parsed.get('experience', [])]),
        'projects': ' '.join([
            ' '.join(p.get('bullets', []) or [] ) + ' ' + str(p.get('summary', '')) for p in parsed.get('projects', [])
        ]),
        'education': ' '.join([f"{ed.get('degree','')} {ed.get('school','')}" for ed in parsed.get('education', [])]),
        'misc': str(parsed.get('misc', '')),
    }
    for skill in required_skills or []:
        rules = RULES.get(skill, None)
        if not rules:
            continue
        best_section = None
        best_conf = 0.0
        best_ev = ''
        for sec, text in sections.items():
            if not text:
                continue
            bucket, bucket_conf, ev = _max_bucket_conf(text, rules)
            if bucket_conf <= 0.0:
                continue
            conf = min(1.0, bucket_conf * SECTION_WEIGHTS.get(sec, 1.0))
            if conf > best_conf:
                best_conf = conf
                best_section = sec
                best_ev = ev
        if best_section and best_conf > 0.0:
            details.append({'skill': skill, 'confidence': best_conf, 'section': best_section, 'evidence': [best_ev] if best_ev else []})
    return details


