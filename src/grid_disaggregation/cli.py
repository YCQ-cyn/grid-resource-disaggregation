"""Command-line workflows for private user-provided electrical measurements."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(prog='grid-disaggregation')
    commands = p.add_subparsers(dest='command', required=True)
    for name in ['resource-train', 'pv-prepare']:
        q = commands.add_parser(name)
        q.add_argument('--plan', required=True)
        q.add_argument('--fields', default='config/fields.json')
        q.add_argument('--output', required=True)
    q = commands.add_parser('pv-train')
    q.add_argument('--table', required=True)
    q.add_argument('--normalizer', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--method', choices=['small_s4d', 'tcn_lstm'], default='small_s4d')
    for name in ['resource-predict', 'pv-predict']:
        q = commands.add_parser(name)
        q.add_argument('--input', required=True)
        q.add_argument('--output', required=True)
        q.add_argument('--model-dir', required=True)
    args = p.parse_args()
    if args.command=='resource-train':
        from .resource.training import run
        run(args.plan, args.fields, args.output)
    elif args.command=='pv-prepare':
        from .pv.data import prepare
        prepare(args.plan, args.fields, args.output)
    elif args.command=='pv-train':
        from .pv.training import run
        run(args.table, args.normalizer, args.output, args.method)
    else:
        import pandas as pd
        from .pv.inference import canonical_csv
        if args.command=='pv-predict':
            from .pv.inference import OnlinePV
            runner = OnlinePV(args.model_dir)
        else:
            from .resource.inference import OnlineResource
            model_dir = Path(args.model_dir)
            selected = json.loads((model_dir/'selection.json').read_text(encoding='utf-8'))
            runner = OnlineResource(model_dir/'selected.pt', model_dir/'data/normalizer.json', selected['thresholds'])
        d = canonical_csv(args.input)
        rows = []
        for at in pd.date_range(d.index.min(), d.index.max(), freq='min'):
            record = d.loc[at].to_dict() if at in d.index else None
            valid = bool(record.get('observed', True)) if record else False
            if args.command=='pv-predict':
                rows.append(runner.update(at, record, valid))
            else:
                rows.append(runner.update(at, record, valid))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)

if __name__=='__main__':
    main()
