#!/usr/bin/env python3
import os, argparse, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='Path to CSV with a text column')
parser.add_argument('--text-col', default='transcription', help='Name of text column')
parser.add_argument('--out-dir', default='/mnt/data', help='Output dir for artifacts')
args = parser.parse_args()

df = pd.read_csv(args.csv)
text_col = args.text_col if args.text_col in df.columns else df.columns[0]
df[text_col] = df[text_col].astype(str)

vec = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_features=100000)
X = vec.fit_transform(df[text_col].tolist())

os.makedirs(args.out_dir, exist_ok=True)
joblib.dump(vec, os.path.join(args.out_dir, 'retrieval_tfidf.joblib'))
sparse.save_npz(os.path.join(args.out_dir, 'recovery_matrix.npz').replace('recovery', 'retrieval'), X)
df[[text_col]].to_csv(os.path.join(args.out_dir, 'retrieval_corpus.csv'), index=False)

print('Wrote artifacts to', args.out_dir)
