import re
from urllib.parse import urlparse

_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_SHORT_RE = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|"
    r"tinyurl|tr\.im|is\.gd|cli\.gs|cutt\.ly|rb\.gy|shorturl\.at"
)

_PHISHING_TLDS = {
    'xyz', 'tk', 'ml', 'ga', 'cf', 'pw', 'top', 'click',
    'loan', 'work', 'date', 'review', 'stream', 'download',
    'ru', 'info', 'biz', 'online', 'site', 'fun', 'icu', 'vip'
}

_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'github.com', 'wikipedia.org',
    'stackoverflow.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'netflix.com', 'reddit.com', 'twitch.tv', 'discord.com',
}

def extract_features(url: str) -> dict:
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    parsed = urlparse(url)
    domain = (parsed.hostname or '').lower()

    bare_domain = domain[4:] if domain.startswith('www.') else domain
    is_known_safe = bare_domain in _SAFE_DOMAINS

    having_ip_address = -1 if _IP_RE.match(domain) else 1

    n = len(url)
    url_length = 1 if n < 54 else (0 if n <= 75 else -1)

    shortining_service = -1 if _SHORT_RE.search(domain) else 1

    having_at_symbol = -1 if '@' in url else 1

    after_proto = url.split('//', 1)[1] if '//' in url else url
    double_slash_redirecting = -1 if '//' in after_proto else 1

    prefix_suffix = -1 if '-' in domain else 1

    check_domain = domain[4:] if domain.startswith('www.') else domain
    dots = check_domain.count('.')
    having_sub_domain = 1 if dots == 1 else (0 if dots == 2 else -1)

    sslfinal_state = -1 if url.startswith('http://') else 1

    https_token = -1 if 'https' in domain else 1

    dnsrecord = -1 if _IP_RE.match(domain) else 1

    tld = domain.rsplit('.', 1)[-1] if '.' in domain else ''
    bare_tld = bare_domain.rsplit('.', 1)[-1] if '.' in bare_domain else tld
    age_of_domain = -1 if bare_tld in _PHISHING_TLDS else 1

    UV = -1

    features = {
        'having_ip_address':          having_ip_address,
        'url_length':                 url_length,
        'shortining_service':         shortining_service,
        'having_at_symbol':           having_at_symbol,
        'double_slash_redirecting':   double_slash_redirecting,
        'prefix_suffix':              prefix_suffix,
        'having_sub_domain':          having_sub_domain,
        'sslfinal_state':             sslfinal_state,
        'domain_registration_length': UV,
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
        features['sslfinal_state'] = 1
        features['age_of_domain'] = 1
        features['dnsrecord'] = 1

    return features
