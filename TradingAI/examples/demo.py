"""
Example usage notebook launcher script (non-notebook) to demonstrate AIExpert features.
"""
from __future__ import annotations

import pandas as pd
from TradingAI.ai.ai_expert import AIExpert


def demo(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    expert = AIExpert()
    expert.ingest_data('demo', df)
    features_path = expert.prepare_and_save_dataset('demo')
    print('Features saved to', features_path)
    # Note: transformer training requires torch; skip if not installed
    try:
        model_path = expert.train_transformer(features_path, epochs=1)
        print('Model saved to', model_path)
    except Exception as e:
        print('Training skipped (torch missing?):', e)
    print(expert.respond_to_text('explain strategy'))
