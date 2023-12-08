from utils import highlight_finder, box_finder
import argparse

parser = argparse.ArgumentParser(description='Mode Selection')
parser.add_argument('mode', type=str, default='highlight', help='highlight or box')
args = parser.parse_args()

if args.mode == 'highlight':
    highlight_finder()
elif args.mode == 'box':
    box_finder()