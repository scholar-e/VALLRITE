"""Decode exported VisualPhoneme probabilities with a local JSON lexicon."""
import argparse
import json
import logging
from pathlib import Path
from .decoder import Lexicon, StreamingDecoder, VOCABULARY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('emissions', type=Path)
    parser.add_argument('--lexicon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beam-width', type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output.with_suffix('.log'))])
    data = json.loads(args.emissions.read_text())
    if tuple(data['vocabulary']) != VOCABULARY:
        raise ValueError('emission vocabulary mismatch')
    decoder = StreamingDecoder(Lexicon(json.loads(args.lexicon.read_text())), beam_width=args.beam_width)
    decoder.accept(data['probabilities'])
    args.output.write_text(json.dumps(decoder.result(), indent=2, allow_nan=False)+'\n')
    logging.info('Decoded %d steps; result=%s', decoder.steps, args.output)


if __name__ == '__main__':
    main()
