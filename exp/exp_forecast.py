"""Optional MAE and tuple-output backbone extensions to the original TSLib loop.

The unmodified MSE baseline remains in exp_long_term_forecasting.py. This class
inherits that training/validation/test loop and never constructs MSAR modules.
"""
import torch.nn as nn

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Exp_Forecast(Exp_Long_Term_Forecast):
    def _build_model(self):
        model = super()._build_model()
        self._last_auxiliary = None
        if self.args.model in {"DUET", "SimpleTM"}:
            model.register_forward_hook(self._capture_auxiliary)
        return model

    def _capture_auxiliary(self, module, inputs, output):
        prediction, self._last_auxiliary = output
        return prediction

    def _select_criterion(self):
        criterion = nn.L1Loss() if self.args.loss == "MAE" else nn.MSELoss()
        if self.args.model not in {"DUET", "SimpleTM"}:
            return criterion

        def loss(prediction, target):
            value = criterion(prediction, target)
            # Match the backbone-specific training regularizers used by MSAR.
            # Validation and test selection use the prediction loss alone.
            if self.model.training and self._last_auxiliary is not None:
                if self.args.model == "DUET":
                    value = value + 0.001 * self._last_auxiliary
                elif self.args.l1_weight > 0:
                    value = value + self.args.l1_weight * self._last_auxiliary[0]
            return value

        return loss
