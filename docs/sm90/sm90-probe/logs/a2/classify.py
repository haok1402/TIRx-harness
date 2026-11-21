import sys, re, collections, xml.etree.ElementTree as ET
t = ET.parse(sys.argv[1]); groups = collections.defaultdict(list); n = collections.Counter()
for tc in t.iter('testcase'):
    f = tc.find('failure')
    if f is None: f = tc.find('error')
    if tc.find('skipped') is not None: n['skip'] += 1; continue
    if f is None: n['pass'] += 1; continue
    n['fail'] += 1
    txt = (f.get('message') or '') + '\n' + (f.text or '')
    m = re.search(r"(error\s*:\s*(Feature|Instruction|Modifier)[^\n]{0,90})", txt) or \
        re.search(r"(CUDA_ERROR_\w+|Back optional access|KeyError: '\w+'|cuTensorMapEncodeTiled: invalid argument|AssertionError[^\n]{0,100}|\w+Error: [^\n]{0,100})", txt)
    key = re.sub(r'\d+', 'N', m.group(1) if m else (f.get('message') or '')[:100])
    groups[key].append(tc.get('classname').split('.')[-1] + '::' + tc.get('name'))
print(dict(n))
for k, v in sorted(groups.items(), key=lambda x: -len(x[1])):
    tests = sorted(set(re.sub(r'\[.*', '', x) for x in v))
    print(f"{len(v):4d} {k[:120]}\n       {', '.join(tests[:6])}{' ...' if len(tests) > 6 else ''}")
