import json, re

with open('AnushmitaDas_CarPricePrediction.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
fixes = []

# Fix 1: remove stale execution_count on every code cell
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code' and cell.get('execution_count') not in (None,):
        cell['execution_count'] = None
        fixes.append(f'Cell {i}: reset execution_count to None')

# Fix 2: replace set_xticklabels(get_xticklabels()...) with tick_params in cell 14
TARGET = "axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha='right')"
REPLACE = "axes[0].tick_params(axis='x', rotation=25)"

for i, cell in enumerate(cells):
    if cell['cell_type'] != 'code':
        continue
    new_src = []
    changed = False
    for line in cell['source']:
        stripped = line.strip()
        if stripped == TARGET:
            # preserve indentation
            indent = line[: len(line) - len(line.lstrip())]
            new_src.append(indent + REPLACE + '\n')
            changed = True
        else:
            new_src.append(line)
    if changed:
        cell['source'] = new_src
        fixes.append(f'Cell {i}: replaced set_xticklabels(get_xticklabels()) with tick_params')

with open('AnushmitaDas_CarPricePrediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Fixes applied:')
for fix in fixes:
    print(' -', fix)
print('Done.')
