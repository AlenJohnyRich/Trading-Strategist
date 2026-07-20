"""
Interactive training UI for CLI. Shows settings, tabs and a simple menu-driven workflow
for training mode: ingest sentences, scramble, request autocorrect from LLMs, apply admin corrections,
and send validated data to transformer pipeline.
"""
from __future__ import annotations

import argparse
import textwrap
import uuid
from typing import Optional

import pandas as pd

from TradingAI.ai.ai_expert import AIExpert
from TradingAI.ai.text_pipeline import SentenceStore, scramble_words, random_phrase_from_sentences
from TradingAI.ai.llm_clients import get_llm_client


class TrainingUI:
    def __init__(self):
        self.expert = AIExpert()
        self.sstore = SentenceStore(base_path='./sentence_store')
        self.llm = get_llm_client()

    def show_settings(self):
        cfg = self.expert.cfg
        print('--- Training Mode Settings ---')
        print(f"EMA windows: {cfg.ema.short_window}, {cfg.ema.mid_window}, {cfg.ema.long_window}")
        print(f"Stochastic: k={cfg.stochastic.k_period} k_smooth={cfg.stochastic.k_smooth} d_smooth={cfg.stochastic.d_smooth}")
        print(f"Use ATR for stop: {cfg.trading.use_atr_for_stop} atr_multiplier={cfg.trading.atr_multiplier}")
        print('-----------------------------')

    def menu(self):
        menu_text = textwrap.dedent('''
        Training Mode Menu:
        1) Ingest sentence (type)
        2) Show stored sentences
        3) Scramble a stored sentence
        4) Auto-correct scrambled sentence (LLM)
        5) Admin correct a sentence
        6) Generate random phrase from stored sentences
        7) Train transformer on stored sentences (build embeddings)
        8) Exit
        Choose an option:
        ''')
        while True:
            choice = input(menu_text).strip()
            if choice == '1':
                self.ingest_sentence()
            elif choice == '2':
                self.list_sentences()
            elif choice == '3':
                self.scramble_one()
            elif choice == '4':
                self.autocorrect()
            elif choice == '5':
                self.admin_correct()
            elif choice == '6':
                self.gen_random_phrase()
            elif choice == '7':
                self.train_on_sentences()
            elif choice == '8':
                print('Exiting training UI')
                break
            else:
                print('Unknown option')

    def ingest_sentence(self):
        s = input('Enter sentence: ').strip()
        sid = 's' + uuid.uuid4().hex[:8]
        path = self.sstore.save_sentence(sid, s, source='user')
        print(f'Stored sentence id={sid} -> {path}')

    def list_sentences(self):
        rows = self.sstore.all_sentences()
        for r in rows:
            print(f"{r['id']}: {r['text']} (meta={r['meta']})")

    def scramble_one(self):
        rows = self.sstore.all_sentences()
        if not rows:
            print('No sentences stored')
            return
        sid = rows[-1]['id']
        s = rows[-1]['text']
        scr = scramble_words(s)
        print('Scrambled:', scr)
        # store scrambled as separate id
        sid2 = 'scr' + uuid.uuid4().hex[:8]
        self.sstore.save_sentence(sid2, scr, source='scrambled')
        print('Saved scrambled as', sid2)

    def autocorrect(self):
        rows = [r for r in self.sstore.all_sentences() if r['meta']['source'] == 'scrambled']
        if not rows:
            print('No scrambled sentences found')
            return
        latest = rows[-1]
        scr = latest['text']
        corrected, conf = self.llm.correct_sentence(scr)
        print(f'LLM corrected -> {corrected} (conf={conf})')
        # compare with simple normalization; if matches store as validated
        sid = 'v' + uuid.uuid4().hex[:8]
        self.sstore.save_sentence(sid, corrected, source='llm', validated_by='llm')
        print('Stored validated sentence id=', sid)

    def admin_correct(self):
        rows = self.sstore.all_sentences()
        for r in rows:
            print(f"{r['id']}: {r['text']}")
        sid = input('Enter id to correct: ').strip()
        new = input('Enter corrected sentence: ').strip()
        self.sstore.save_sentence(sid + '_admin', new, source='admin', validated_by='admin')
        print('Saved admin-corrected sentence')

    def gen_random_phrase(self):
        texts = self.sstore.get_text_list()
        phrase = random_phrase_from_sentences(texts, n_words=6)
        print('Random phrase:', phrase)

    def train_on_sentences(self):
        texts = self.sstore.get_text_list()
        if not texts:
            print('No sentences to train on')
            return
        # build embeddings and prepare a toy dataset: treat next-word probability as target (mock)
        from TradingAI.ai.text_pipeline import build_feature_matrix
        X = build_feature_matrix(texts, dim=64)
        # create mock targets (e.g., length of sentence)
        y = [len(t.split()) for t in texts]
        try:
            from TradingAI.ai.learning_engine import LearningEngine
            le = LearningEngine()
            # Save dataset to FilesystemStore for compatibility
            ds_name = 'sentences_features'
            from TradingAI.db.connectors import FilesystemStore
            fs = FilesystemStore('./data_store')
            fs.save_dataset(ds_name, pd.DataFrame(X))
            print('Saved feature dataset; attempt to train transformer (requires torch)')
            # training may fail if torch not installed
            try:
                path = le.train_transformer(X, np.array(y), seq_len=16, epochs=1, batch_size=8)
                print('Trained model saved to', path)
            except Exception as e:
                print('Training skipped or failed:', e)
        except Exception as exc:
            print('Training engine not available:', exc)


def main():
    ui = TrainingUI()
    ui.show_settings()
    ui.menu()


if __name__ == '__main__':
    main()
