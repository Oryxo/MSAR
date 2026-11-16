# Code provenance

This focused adaptation is derived from the MSAR implementation at:

- Repository: https://github.com/fanfanda/MSAR
- Source branch: `icml-submit-new`
- Source commit: `2777a97`

The selected PatchTST, DLinear, TimeMixer, iTransformer, TimesNet, SimpleTM, and
DUET models and the MSAR training path originate from that source snapshot.
Unselected model families and dataset loaders were removed. The CLI defaults
to two-scale MSAR with compact imputation (`d_model=16`, `d_ff=32`). The
`--full_imputation` option matches imputation width to forecasting width and is
recommended for DUET. The initial example scripts cover PatchTST on ETTh1.
Selected checkpoints are retained after evaluation.

The intended public repository is https://github.com/Oryxo/MSAR.

## Time-Series-Library forecasting path

The original long-term forecasting experiment, base experiment, and supporting
utilities are retained from `thuml/Time-Series-Library` at commit
`4e938a1767106324dd753b2a44832bf870a0252e`. The original files and Git blob hashes
are listed in `licenses/tslib_sources.json`; their MIT notice is preserved in
`licenses/Time-Series-Library.LICENSE`.

`exp/exp_long_term_forecasting.py` is unchanged from that snapshot. Baselines
use its independent training loop and default MSE criterion. The separate
`exp/exp_forecast.py` adapter supports explicit MAE and the tuple outputs and
auxiliary training losses of DUET and SimpleTM. These extensions are not part
of the original TSLib experiment class.

MSAR retains its existing base class, training helpers, and metrics under
`exp/exp_msar_basic.py`, `utils/msar_tools.py`, and `utils/msar_metrics.py`.
The local ETT/custom CSV loaders remain shared; this repository does not bundle
TSLib's unrelated classification, anomaly detection, or short-term tasks.
Single-scale CLI requests now select the independent baseline experiment.

The MSAR source snapshot did not include a project-level `LICENSE` file. The
third-party MIT notice applies to the copied TSLib code, not to a new license
grant for the entire repository.
