"""Select by generated validation WER, allowing the unchanged adapter to win."""
import argparse
import hashlib
import json
import logging
from pathlib import Path


def select(reports):
    if not reports:raise ValueError('no candidates')
    baseline=reports[0]
    for report in reports:
        for key in ('source_sha256','records','reference_words','model','revision','score_weight','scoring_token_modes','selection'):
            if report[key]!=baseline[key]:raise ValueError(f'unmatched validation {key}')
        if report.get('limit') is not None:raise ValueError('partial validation')
        if report['scoring_token_modes']!=['exact_generated_tokens']:raise ValueError('exact scoring required')
        if len(report['details'])!=report['records']:raise ValueError('incomplete details')
        if [(r['clip_id'],r['reference_words']) for r in report['details']] != [(r['clip_id'],r['reference_words']) for r in baseline['details']]:raise ValueError('unmatched validation IDs/references')
    return min(range(len(reports)),key=lambda i:(reports[i]['metrics']['probability_ranked']['word_errors'],i))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,nargs='+',required=True,help='baseline first, checkpoints in increasing update order')
    p.add_argument('--expected-source',type=Path,required=True,help='predeclared validation phone export')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s:%(lineno)d %(message)s',handlers=[logging.FileHandler(a.output/'run.log'),logging.StreamHandler()])
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    reports=[json.loads(path.read_text()) for path in a.reports]
    if any(r['source_sha256']!=sha(a.expected_source) for r in reports):raise ValueError('not the predeclared validation source')
    winner=select(reports)
    result=dict(rule='minimum validation ranked word errors; ties prefer baseline then earlier checkpoint',winner_index=winner,winner_report=str(a.reports[winner]),winner_adapter_sha256=reports[winner]['adapter_sha256'],candidates=[dict(report=str(p),sha256=sha(p),errors=r['metrics']['probability_ranked']['word_errors']) for p,r in zip(a.reports,reports)])
    (a.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n');logging.info('Selected %s',result)
if __name__=='__main__':main()
