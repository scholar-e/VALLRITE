"""Download and verify the fixed 12-speaker GRID pilot (research use)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1] / 'datasets' / 'grid-pilot'
BASE = 'https://spandh.dcs.shef.ac.uk/gridcorpus/'


def fetch(job):
    speaker, kind, relative = job
    destination = ROOT / 'archives' / f's{speaker}' / kind / Path(relative).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    if not destination.exists():
        subprocess.run(['curl', '-L', '--fail', '--retry', '5', '--retry-delay', '3',
                        '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                        '--silent', '--show-error', '--continue-at', '-',
                        '--output', str(partial), BASE + relative], check=True)
        partial.rename(destination)
    out = ROOT / 'extracted' / f's{speaker}' / kind
    out.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(destination):
        with zipfile.ZipFile(destination) as archive:
            for member in archive.infolist():
                target = (out / member.filename).resolve()
                if not target.is_relative_to(out.resolve()):
                    raise ValueError('unsafe archive path')
            bad = archive.testzip()
            if bad:
                raise ValueError(f'CRC failed: {bad}')
            archive.extractall(out)
    else:
        with tarfile.open(destination) as archive:
            archive.extractall(out, filter='data')
    with destination.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    files = list(out.rglob('*'))
    count = sum(p.is_file() for p in files)
    print(f's{speaker} {kind}: verified and extracted {count} files ({destination.stat().st_size / 1e6:.1f} MB)', flush=True)
    return {'speaker': str(speaker), 'kind': kind, 'url': BASE + relative,
            'archive': str(destination.relative_to(ROOT)), 'sha256': sha,
            'bytes': destination.stat().st_size, 'extracted_files': count}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, kind, f's{s}/{folder}/{name}') for s in range(1, 13)
            for kind, folder, name in [('video', 'video', f's{s}.mpg_vcd.zip'),
                                       ('audio', 'audio', f's{s}_50kHz.tar'),
                                       ('alignment', 'align', f's{s}.tar')]]
    report = {'source': BASE, 'terms': 'Freely available for research use; see source page.',
              'video_quality': '360x288 normal quality', 'audio': 'original 50 kHz; not endpointed 25 kHz',
              'labels': 'word alignments; phoneme alignment still required',
              'splits': {'train': [str(s) for s in range(1, 9)], 'validation': ['9', '10'], 'test': ['11', '12']},
              'files': [], 'errors': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                report['files'].append(future.result())
            except Exception as error:
                report['errors'].append({'job': futures[future], 'error': str(error)})
                print(f'FAILED {futures[future]}: {error}', flush=True)
            (ROOT / 'download-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['errors']:
        raise SystemExit('Some downloads failed; rerun to resume.')
    print('Complete: all 36 archives verified and extracted.', flush=True)


if __name__ == '__main__':
    main()
