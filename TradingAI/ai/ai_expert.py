"""
AI Expert orchestration tool. This module ties together data ingestion, database storage,
transformer training/inference, visualization, memory and a simple language interface
so the AI responds to user requests like 'train model', 'predict', 'show graph', 'explain strategy'.

The language intelligence is a compact rule-based parser that maps natural language
requests to actions; it's extendable with real LLMs or DeepSeek integration.
"""
from __future__ import annotations

import logging
import textwrap
from typing import Any, Optional

import numpy as np
import pandas as pd

from TradingAI.db.connectors import FilesystemStore
from TradingAI.datasets.generator import generate_dataset
from TradingAI.ai.learning_engine import LearningEngine
from TradingAI.ai.inference import TransformerInferencer
from TradingAI.memory.memory_bank import MemoryBank, MemoryItem
from TradingAI.visualization.graphs import plot_price_and_indicators
from TradingAI.strategies.reversal import generate_signals
from TradingAI.config import get_config

logger = logging.getLogger('TradingAI.ai_expert')


class AIExpert:
    """High-level AI tool to manage pipeline and respond to language requests.

    Key capabilities:
    - ingest_data(df): store raw market data
    - prepare_and_save_dataset(name): create features and save to DB
    - train_transformer(name, dataset_name): train transformer model and save checkpoint
    - infer(path_to_model, df): run inference and return predictions
    - respond_to_text(query): simple NLU dispatcher that executes tasks and returns textual responses

    This is intentionally implemented without heavy external LLM dependencies; however,
    it includes hooks to integrate DeepSeek or other LLMs for improved language understanding.
    """

    def __init__(self, store_path: str = './data_store', model_dir: str = './models'):
        self.cfg = get_config()
        self.store = FilesystemStore(store_path)
        self.memory = MemoryBank(path='./memory_store.json', autosave=True)
        self.learning = LearningEngine(model_dir)
        self.last_ingested_name: Optional[str] = None

    def ingest_data(self, name: str, df: pd.DataFrame) -> str:
        """Ingest and persist raw market data (DataFrame expected)."""
        # Normalize index
        if 'timestamp' in df.columns:
            df = df.set_index(pd.to_datetime(df['timestamp']))
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # persist
        path = self.store.save_dataset(name, df)
        self.last_ingested_name = name
        # record in memory
        self.memory.add(MemoryItem(id=f'data:{name}', type='dataset', payload={'path': path}))
        logger.info('Ingested dataset %s -> %s', name, path)
        return path

    def prepare_and_save_dataset(self, name: str, window: int = 100) -> str:
        """Load last ingested dataset or named dataset and export features for ML training."""
        if self.last_ingested_name is None and not name:
            raise ValueError('No dataset available to prepare')
        ds_name = name or self.last_ingested_name
        df = self.store.load_dataset(ds_name)
        out = generate_dataset(df, window=window)
        path = self.store.save_dataset(f'{ds_name}_features', out)
        self.memory.add(MemoryItem(id=f'features:{ds_name}', type='features', payload={'path': path}))
        return path

    def train_transformer(self, dataset_name: str, seq_len: int = 64, epochs: int = 5, batch_size: int = 64) -> str:
        """Train transformer on a named dataset stored in FilesystemStore.
        dataset_name should reference the features dataset produced by prepare_and_save_dataset.
        """
        obj = self.store.load_dataset(dataset_name)
        # expected dataframe with 'target' column
        if isinstance(obj, pd.DataFrame):
            X = obj.drop(columns=['target']).values
            y = obj['target'].values
        else:
            raise ValueError('Unsupported dataset object type for transformer training')
        save_path = self.learning.train_transformer(X, y, seq_len=seq_len, epochs=epochs, batch_size=batch_size)
        # persist metadata in memory
        self.memory.add(MemoryItem(id=f'model:{save_path}', type='model', payload={'path': save_path}))
        return save_path

    def infer(self, model_path: str, df: pd.DataFrame):
        inferencer = TransformerInferencer(model_path)
        # build feature matrix similar to training: use last N rows with raw OHLCV -> minimal conversion here
        X = df[['open', 'high', 'low', 'close']].values
        preds = inferencer.predict(X)
        return preds

    def graph(self, df: pd.DataFrame, signals: Optional[list] = None, preds: Optional[np.ndarray] = None) -> str:
        # compute indicators if missing
        if 'ema21' not in df.columns:
            from TradingAI.strategies.reversal import ema
            df = df.copy()
            df['ema21'] = ema(df['close'], self.cfg.ema.short_window)
            df['ema34'] = ema(df['close'], self.cfg.ema.mid_window)
            df['ema144'] = ema(df['close'], self.cfg.ema.long_window)
            k, d = __import__('TradingAI').indicators.stochastic_expert.stochastic(df, self.cfg.stochastic.k_period, self.cfg.stochastic.k_smooth, self.cfg.stochastic.d_smooth)
            df['stoch_k'] = k
            df['stoch_d'] = d
        out = plot_price_and_indicators(df, signals=signals, preds=preds)
        self.memory.add(MemoryItem(id=f'plot:{out}', type='plot', payload={'path': out}))
        return out

    def explain_strategy(self) -> str:
        # Return the detailed explanation of the EMA+Stochastic reversal strategy
        text = textwrap.dedent('''
        Strategy: Reversal using EMA(21,34,144) and Stochastic(7,3,3).

        1) Trend confirmation: price crosses EMA144 and EMA21/EMA34 align with EMA144.
        2) Wait for first pullback after the trend confirmation.
        3) For long: stochastic K drops below 20, then crosses above D and closes above 20 -> enter long.
        4) For short: stochastic K rises above 80, then crosses below D and closes below 80 -> enter short.
        5) Stop: pullback low/high or ATR-based stop if configured; target: 2x risk.
        6) Exit on EMA21 crossing EMA34 against trade direction.

        The engine enforces the first-pullback-only rule to reduce false signals.
        ''')
        return text

    def respond_to_text(self, text: str) -> str:
        """Simple natural language dispatcher for common requests.
        This is intentionally compact: it maps keywords/phrases to actions. For richer
        language understanding, integrate a dedicated LLM or DeepSeek API.
        """
        t = text.strip().lower()
        try:
            if 'train' in t and 'transformer' in t:
                # expect dataset name like 'features:<name>' in memory
                mem = [m for m in self.memory.query('features')]
                if not mem:
                    return 'No feature dataset available. Please ingest data and prepare features first.'
                ds_path = mem[-1].payload['path']
                # load dataset name from path
                ds_name = pathlib.Path(ds_path).stem.replace('dataset_', '').replace('_features', '')
                model_path = self.train_transformer(f'{ds_name}_features')
                return f'Training started and model saved to {model_path}'
            if 'predict' in t or 'infer' in t:
                # use last ingested dataset for inference
                if self.last_ingested_name is None:
                    return 'No dataset available. Please ingest data first.'
                df = self.store.load_dataset(self.last_ingested_name)
                mem_models = [m for m in self.memory.query('model')]
                if not mem_models:
                    return 'No trained model found. Train a model first.'
                model_path = mem_models[-1].payload['path']
                preds = self.infer(model_path, df)
                # persist preds
                self.memory.add(MemoryItem(id=f'infer:{model_path}', type='inference', payload={'length': len(preds)}))
                return f'Inference complete, produced {len(preds)} predictions. Use graph to visualize.'
            if 'show' in t and 'graph' in t:
                if self.last_ingested_name is None:
                    return 'No dataset ingested yet.'
                df = self.store.load_dataset(self.last_ingested_name)
                from TradingAI.strategies.reversal import generate_signals
                sigs = generate_signals(df)
                mem_models = [m for m in self.memory.query('model')]
                preds = None
                if mem_models:
                    model_path = mem_models[-1].payload['path']
                    try:
                        preds = TransformerInferencer(model_path).predict(df[['open', 'high', 'low', 'close']].values)
                    except Exception:
                        preds = None
                plot_path = self.graph(df, signals=sigs, preds=preds)
                return f'Graph generated at {plot_path}'
            if 'explain' in t and 'strategy' in t:
                return self.explain_strategy()
            if 'help' in t:
                return 'Supported commands: train transformer, predict, show graph, explain strategy, ingest data <name> (CSV path).'
            if t.startswith('ingest'):
                # expected format: ingest name path
                parts = text.split()
                if len(parts) >= 3:
                    name = parts[1]
                    path = parts[2]
                    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
                    p = self.ingest_data(name, df)
                    return f'Ingested dataset {name} -> {p}'
                return 'Ingest command expects: ingest <name> <csv_path>'
            return 'Sorry, I did not understand. Supported commands: train transformer, predict, show graph, explain strategy, ingest data <name> (CSV path).'
        except Exception as exc:
            logger.exception('Error handling user request')
            return f'An error occurred: {exc}'
