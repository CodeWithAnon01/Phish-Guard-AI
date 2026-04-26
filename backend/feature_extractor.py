"""
feature_extractor.py
====================
Extracts the 30-feature UCI Phishing Websites vector from a raw URL string.
All features are computed from the URL text alone — no network calls —
so extraction is O(1) and completes in < 1 ms.

Feature scale: -1 = phishing indicator, 0 = suspicious/unknown, 1 = legitimate
(matches the UCI Phishing Websites dataset encoding)

Incorporates address-bar features from:
  Shreya Gopal (2021) "Phishing Website Detection by Machine Learning Techniques"
  https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
"""

import re
from urllib.parse import urlparse

# ── Compiled Patterns ─────────────────────────────────────────────────────────

_IP_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
    r"|"
    r"^0x[0-9a-fA-F]+"          # hex-encoded IP
    r"|"
    r"^\d{8,10}$"               # decimal-encoded IP
)

# Expanded shortener list — merged from UCI dataset + Shreya Gopal reference
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

# TLDs strongly correlated with phishing / free-domain abuse
_PHISHING_TLDS = {
    'xyz', 'tk', 'ml', 'ga', 'cf', 'pw', 'top', 'click',
    'loan', 'work', 'date', 'review', 'stream', 'download',
    'ru', 'info', 'biz', 'online', 'site', 'fun', 'icu', 'vip',
    'gq', 'buzz', 'live', 'cc', 'ws', 'us',
}

# Domains we unconditionally trust — subdomains are matched via suffix check
_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'github.com', 'wikipedia.org',
    'stackoverflow.com', 'amazon.com', 'microsoft.com', 'apple.com',
    'netflix.com', 'reddit.com', 'twitch.tv', 'discord.com',
    # Cloud storage & productivity
    'drive.google.com', 'docs.google.com', 'sheets.google.com',
    'slides.google.com', 'mail.google.com', 'accounts.google.com',
    'dropbox.com', 'onedrive.live.com', 'sharepoint.com',
    'office.com', 'live.com', 'outlook.com', 'hotmail.com',
    # Dev & CDN
    'gitlab.com', 'bitbucket.org', 'npmjs.com', 'pypi.org',
    'cloudflare.com', 'amazonaws.com', 'azure.com', 'storage.googleapis.com',
    # Education & reference
    'scholar.google.com', 'arxiv.org', 'medium.com', 'quora.com',
}


# ── Main Extraction Function ──────────────────────────────────────────────────

def extract_features(url: str) -> dict:
    """Return a dict of 30 UCI-schema features for the given URL.

    All values are in {-1, 0, 1}:
        -1  → phishing signal
         0  → suspicious / cannot determine (used for content-level features
               we cannot compute without fetching the page)
         1  → legitimate signal
    """
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    parsed     = urlparse(url)
    domain     = (parsed.hostname or '').lower()
    bare_domain = domain[4:] if domain.startswith('www.') else domain

    # ── Known-safe shortcut ──────────────────────────────────────────────────
    is_known_safe = any(
        bare_domain == s or bare_domain.endswith('.' + s)
        for s in _SAFE_DOMAINS
    )

    # ── Address-bar features ─────────────────────────────────────────────────

    # 1. IP address in domain
    having_ip_address = -1 if _IP_RE.match(domain) else 1

    # 2. URL length  (< 54 → safe | 54–75 → suspicious | > 75 → phishing)
    n = len(url)
    url_length = 1 if n < 54 else (0 if n <= 75 else -1)

    # 3. URL shortening service
    shortining_service = -1 if _SHORT_RE.search(domain) else 1

    # 4. @ symbol in URL
    having_at_symbol = -1 if '@' in url else 1

    # 5. Double-slash redirection: '//' appearing after position 7 in the URL
    #    (beyond 'https://') indicates a redirect trick — reference feature #6
    after_proto = url.split('//', 1)[1] if '//' in url else url
    double_slash_redirecting = -1 if '//' in after_proto else 1

    # 6. Prefix/suffix '-' in domain
    prefix_suffix = -1 if '-' in bare_domain else 1

    # 7. Sub-domain depth  (1 dot = no sub-domain → legit; 2 = one sub-domain
    #    → suspicious; 3+ = multiple sub-domains → phishing)
    dots = bare_domain.count('.')
    having_sub_domain = 1 if dots == 0 else (0 if dots == 1 else -1)

    # 8. HTTPS in scheme
    sslfinal_state = -1 if url.startswith('http://') else 1

    # 9. 'https' token appearing inside the domain string (deceptive trick)
    https_token = -1 if 'https' in bare_domain else 1

    # ── Domain-level heuristics (no network) ────────────────────────────────

    # 10. DNS record proxy: IP-in-domain = missing DNS
    dnsrecord = -1 if _IP_RE.match(domain) else 1

    # 11. Age-of-domain proxy via high-risk TLD
    tld = bare_domain.rsplit('.', 1)[-1] if '.' in bare_domain else ''
    age_of_domain = -1 if tld in _PHISHING_TLDS else 1

    # 12. URL depth — number of non-empty path segments (reference feature #5)
    #     Maps to domain_registration_length slot; higher depth raises suspicion
    path_segments = [s for s in parsed.path.split('/') if s]
    url_depth = len(path_segments)
    # Normalise to UCI scale: 0–2 segments = legit, 3–4 = suspicious, 5+ = phishing
    domain_registration_length = 1 if url_depth <= 2 else (0 if url_depth <= 4 else -1)

    # ── Content-level features (unknown at URL-scan time) ────────────────────
    # We use 0 (suspicious/unknown) so these do not artificially push the model
    # toward phishing for legitimate sites we haven't fetched.
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

    # ── Known-safe override ──────────────────────────────────────────────────
    # Override every heuristic flag for verified-safe domains so that long
    # Google Drive links, hyphenated subdomains, etc. never trigger phishing.
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
