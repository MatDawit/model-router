"""LightGBM router: learns "if A and B then likely to fail" style patterns.

Unlike the baselines in benchmarks/baselines/simple.py (which take raw text),
this takes the ready-made feature matrix (n_prompts x 396) from HybridFeatures.
    model.fit(x, y)           # y: 0/1 "tier failed"
    model.predict_proba(x)    # P(tier fails) per prompt
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LGBMRouter:
    def __init__(self, seed=42, **overrides):
        self.seed = seed
        self.params = dict(
            n_estimators=1000,      # upper limit; early stopping usually ends far sooner
            learning_rate=0.05,     # small steps, so each tree corrects a little
            num_leaves=15,          # keeps each tree small and simple
            min_child_samples=50,   # a rule must be backed by at least 50 prompts
            subsample=0.8,          # each tree sees a random 80% of prompts...
            subsample_freq=1,
            colsample_bytree=0.5,   # ...and a random half of the features
            reg_lambda=1.0,         # mild penalty against over-complicated trees
            random_state=seed,
            verbose=-1,
        )
        self.params.update(overrides)

    def fit(self, x, y):
        # Hold back 10% of the TRAINING data to know when to stop adding trees.
        x_fit, x_val, y_fit, y_val = train_test_split(
            x, y, test_size=0.1, stratify=y, random_state=self.seed
        )
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(
            x_fit,
            y_fit,
            eval_set=[(x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],  # stop after 50 trees with no gain
        )
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(x)[:, 1]
