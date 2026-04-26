

import re
from urllib.parse import urlparse

_IP_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
    r"|"
    r"^0x[0-9a-fA-F]+"
    r"|"
    r"^\d{8,10}$"
)
_SHORT_RE = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|cutt\.ly|rb\.gy|shorturl\.at|yfrog\.com|migre\.me|ff\.im|"
    r"tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|"
    r"BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|"
    r"loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|"
    r"bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|"
    r"ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|"
    r"cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|"
    r"vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net"
)
_PHISHING_TLDS = {
    'xyz', 'tk', 'ml', 'ga', 'cf', 'pw', 'top', 'click',
    'loan', 'work', 'date', 'review', 'stream', 'download',
    'ru', 'info', 'biz', 'online', 'site', 'fun', 'icu', 'vip',
    'gq', 'buzz', 'live', 'cc', 'ws', 'us',
}
_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'github.com', 'wikipedia.org',
    'stackoverflow.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'netflix.com', 'reddit.com', 'twitch.tv', 'discord.com',
    'drive.google.com', 'docs.google.com', 'sheets.google.com',
    'slides.google.com', 'mail.google.com', 'accounts.google.com',
    'dropbox.com', 'onedrive.live.com', 'sharepoint.com',
    'office.com', 'live.com', 'outlook.com', 'hotmail.com',
    'gitlab.com', 'bitbucket.org', 'npmjs.com', 'pypi.org',
    'cloudflare.com', 'amazonaws.com', 'azure.com', 'storage.googleapis.com',
    'scholar.google.com', 'arxiv.org', 'medium.com', 'quora.com',
}

def extract_features(url: str) -> dict:
    
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    parsed     = urlparse(url)
    domain     = (parsed.hostname or '').lower()
    bare_domain = domain[4:] if domain.startswith('www.') else domain
    is_known_safe = any(
        bare_domain == s or bare_domain.endswith('.' + s)
        for s in _SAFE_DOMAINS
    )
    having_ip_address = -1 if _IP_RE.match(domain) else 1
    n = len(url)
    url_length = 1 if n < 54 else (0 if n <= 75 else -1)
    shortining_service = -1 if _SHORT_RE.search(domain) else 1
    having_at_symbol = -1 if '@' in url else 1
    after_proto = url.split('//', 1)[1] if '//' in url else url
    double_slash_redirecting = -1 if '//' in after_proto else 1
    prefix_suffix = -1 if '-' in bare_domain else 1
    dots = bare_domain.count('.')
    having_sub_domain = 1 if dots == 0 else (0 if dots == 1 else -1)
    sslfinal_state = -1 if url.startswith('http://') else 1
    https_token = -1 if 'https' in bare_domain else 1
    dnsrecord = -1 if _IP_RE.match(domain) else 1
    tld = bare_domain.rsplit('.', 1)[-1] if '.' in bare_domain else ''
    age_of_domain = -1 if tld in _PHISHING_TLDS else 1
    path_segments = [s for s in parsed.path.split('/') if s]
    url_depth = len(path_segments)
    domain_registration_length = 1 if url_depth <= 2 else (0 if url_depth <= 4 else -1)
    UV = 0

    features = {
        'having_ip_address':          having_ip_address,
        'url_length':                 url_length,
        'shortining_service':         shortining_service,
        'having_at_symbol':           having_at_symbol,
        'double_slash_redirecting':   double_slash_redirecting,
        'prefix_suffix':              prefix_suffix,
        'having_sub_domain':          having_sub_domain,
        'sslfinal_state':             sslfinal_state,
        'domain_registration_length': domain_registration_length,
        'favicon':                    UV,
        'port':                       UV,
        'https_token':                https_token,
        'request_url':                UV,
        'url_of_anchor':              UV,
        'links_in_tags':              UV,
        'sfh':                        UV,
        'submitting_to_email':        UV,
        'abnormal_url':               UV,
        'redirect':                   UV,
        'on_mouseover':               UV,
        'rightclick':                 UV,
        'popupwindow':                UV,
        'iframe':                     UV,
        'age_of_domain':              age_of_domain,
        'dnsrecord':                  dnsrecord,
        'web_traffic':                UV,
        'page_rank':                  UV,
        'google_index':               UV,
        'links_pointing_to_page':     UV,
        'statistical_report':         UV,
    }
    if is_known_safe:
        features.update({
            'sslfinal_state':             1,
            'age_of_domain':              1,
            'dnsrecord':                  1,
            'url_length':                 1,
            'prefix_suffix':              1,
            'having_sub_domain':          1,
            'shortining_service':         1,
            'having_at_symbol':           1,
            'double_slash_redirecting':   1,
            'https_token':                1,
            'web_traffic':                1,
            'google_index':               1,
            'page_rank':                  1,
            'domain_registration_length': 1,
        })

    return features
