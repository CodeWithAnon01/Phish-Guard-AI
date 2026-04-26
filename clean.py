import os, re
for root, _, files in os.walk(os.getcwd()):
    if 'venv' in root or '.git' in root: continue
    for file in files:
        if not file.endswith(('.py', '.js', '.css', '.html')): continue
        p = os.path.join(root, file)
        with open(p, 'r') as f:
            c = f.read()
        ext = file.split('.')[-1]
        if ext == 'py':
            c = re.sub(r'^\s*#.*?\n', '', c, flags=re.M)
            c = re.sub(r'\s+#.*$', '', c, flags=re.M)
            c = re.sub(r'"""[\s\S]*?"""', '', c)
            c = re.sub(r"'''[\s\S]*?'''", '', c)
        elif ext in ['js', 'css']:
            c = re.sub(r'^\s*//.*?\n', '', c, flags=re.M)
            c = re.sub(r'/\*[\s\S]*?\*/', '', c)
        elif ext == 'html':
            c = re.sub(r'<!--[\s\S]*?-->', '', c)
        with open(p, 'w') as f:
            f.write(c)
