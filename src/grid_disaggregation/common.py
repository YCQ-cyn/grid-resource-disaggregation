from pathlib import Path
import hashlib
import json

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding='utf-8'))
    date_roles = {}
    for batch in plan['batches']:
        seen = set()
        for role, dates in batch['dates'].items():
            for date in dates:
                from datetime import date as calendar_date
                calendar_date.fromisoformat(date)
                if date in seen:
                    raise ValueError('A batch date cannot appear in multiple roles')
                seen.add(date)
                if date in date_roles and date_roles[date] != role:
                    raise ValueError('A calendar date cannot span training and evaluation')
                date_roles[date] = role
        for role in ['main', 'pv', 'energy', 'charger']:
            if batch.get(role):
                batch[role] = str((path.parent / batch[role]).resolve())
    for source in plan.get('auxiliary_chargers', []):
        source['path'] = str((path.parent / source['path']).resolve())
    return plan
