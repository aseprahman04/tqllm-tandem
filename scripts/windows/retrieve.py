import sys, re
task = open(sys.argv[1], encoding='utf-8').read().lower()
words = set(re.findall(r'[a-z0-9_]+', task))
scored = []
for ln in open(sys.argv[2], encoding='utf-8'):
    if '|' not in ln: continue
    keys, rule = ln.split('|', 1)
    ks = [k.strip() for k in keys.split(',') if k.strip()]
    s = sum(1 for k in ks if (k in task if ' ' in k else k in words))
    if s >= 2: scored.append((s, rule.strip()))
scored.sort(key=lambda x: -x[0])
for s, r in scored[:5]: print('- ' + r)
