from datetime import datetime

from tensorflow.compat.v1.summary import FileWriter, Summary
from tensorflow.keras.callbacks import Callback

from utils import calculate_metrics, make_predictions


class Metrics(Callback):
    def __init__(self, X_val, y_val, interval=10_000):
        self.interval = interval
        self.step = 0
        self.X_val = X_val
        self.y_val = y_val
        self.writer = FileWriter("./logs/" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    def on_step_end(self, episode_step, logs):
        """Calculate metrics every `interval`-steps. Save target_model if conditions are met."""
        self.step += 1

        if not self.step % self.interval:
            y_pred = make_predictions(self.model.target_model, self.X_val)
            stats = calculate_metrics(self.y_val, y_pred)

            for k, v in stats.items():
                summary = Summary(value=[Summary.Value(tag=k, simple_value=v)])
                self.writer.add_summary(summary, global_step=self.step)

            if stats.get("FN") <= 20 and stats.get("FP") <= 578:
                pass
