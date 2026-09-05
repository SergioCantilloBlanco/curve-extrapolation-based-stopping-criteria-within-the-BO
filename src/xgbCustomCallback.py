import xgboost as xgb

class EarlyStopWithPrediction(Exception):
    def __init__(self, predicted_value, val_mae_history, best_so_far):
        self.predicted_value = predicted_value
        self.val_mae_history = val_mae_history
        self.best_so_far = best_so_far
        super().__init__(f"Early stopped, predicted value: {predicted_value}")

class customCallback(xgb.callback.TrainingCallback):
    def __init__(self, trial, study):
        self.trial = trial
        self.study = study

    def after_iteration(self, model, epoch, evals_log):
        current_mae = evals_log["val"]["mae"][-1]
        self.trial.report(current_mae, step=epoch)

        should_stop, predicted_value, val_mae_history, best_so_far = self.study.pruner.prune(self.study, self.trial, evals_log["val"]["mae"])
        if should_stop:
            raise EarlyStopWithPrediction(predicted_value, val_mae_history, best_so_far)
        return False