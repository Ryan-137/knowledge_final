"""预览输出数据质量（Step 8 前的内容确认）。"""
import sys, json
sys.path.insert(0,'.')
from pathlib import Path
from optimize.config import cfg

clusters = json.loads((cfg.paths.canonical_root/'new_entity_clusters.json').read_text(encoding='utf-8'))
print('新实体簇总数:', clusters['total_clusters'])
print()
print('可自动新建节点的候选簇（size>=3）：')
valid_types = {'tool','skill','knowledge','project','soft_skill','language','interest','constraint'}
for c in clusters['clusters']:
    if c['size'] >= 3 and c['suggested_type'] in valid_types:
        cid   = c['cluster_id']
        ctype = c['suggested_type']
        size  = c['size']
        surfs = ', '.join(c['surfaces'][:5])
        print(f'  cid={cid:02d} size={size:3d} type={ctype:<12}: {surfs}')
print()

alignment = json.loads((cfg.paths.canonical_root/'external_alignment.json').read_text(encoding='utf-8'))
print('O*NET 命中节点：')
for nid, refs in alignment['alignment'].items():
    if refs:
        best = refs[0]
        print(f'  {nid:<40} -> {best["tool_name"]}  sim={best["similarity"]}')
