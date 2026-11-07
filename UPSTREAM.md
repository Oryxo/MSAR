# Code provenance

This focused adaptation is derived from the MSAR implementation at:

- Repository: https://github.com/fanfanda/MSAR
- Source branch: `icml-submit-new`
- Source commit: `2777a97`

The selected PatchTST, DLinear, TimeMixer, iTransformer, TimesNet, SimpleTM, and
DUET models and the MSAR training path originate from that source snapshot.
Unselected model families and dataset loaders were removed, and imputation
width reduction is now controlled by an explicit option.

The source snapshot did not include a `LICENSE` file. No license is added here;
confirm the upstream licensing terms before public distribution.
