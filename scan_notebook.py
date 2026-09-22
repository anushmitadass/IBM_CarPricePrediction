import json, sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

nb = json.load(open('AnushmitaDas_CarPricePrediction.ipynb', encoding='utf-8'))
cells = nb['cells']

issues = []
for i, c in enumerate(cells):
    for o in c.get('outputs', []):
        if o.get('output_type') == 'error':
            issues.append('Cell {}: has error output'.format(i))
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if 'squared=False' in src:
        issues.append('Cell {}: squared=False (deprecated)'.format(i))
    if 'vert=True' in src:
        issues.append('Cell {}: vert=True (deprecated)'.format(i))
    if 'figure.titlecolor' in src:
        issues.append('Cell {}: figure.titlecolor (invalid key)'.format(i))
    if 'get_xticklabels' in src:
        issues.append('Cell {}: get_xticklabels warning pattern'.format(i))
    if c.get('execution_count') not in (None,):
        issues.append('Cell {}: non-null execution_count = {}'.format(i, c['execution_count']))

code_count = sum(1 for c in cells if c['cell_type'] == 'code')
print('Notebook: {} cells total, {} code cells.'.format(len(cells), code_count))

if issues:
    print('REMAINING ISSUES:')
    for iss in issues:
        print(' -', iss)
else:
    print('All clear - zero issues found!')
