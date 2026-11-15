# Validation rules and known limits

1. `experiment_id`, `variable_id`, `variable_name`, `unit`, and `source_file` are required.
2. Values must be finite numeric values; NaN and infinity are rejected.
3. Every `variable_id` must exist in the registry.
4. `measurement_class` must be one of `reported_experimental_variable`, `experimental_setting`, or `derived_result`.
5. Duplicate `observation_id` values are rejected.
6. A timestamp is parsed as UTC only when a source timestamp is actually supplied. Missing time stays null.
7. Missing numeric cells are not imputed or converted to zero.
8. Units are not guessed. Unclear units are marked as not specified or source-defined.
9. The supplied processed tables do not establish instrument model, channel ID, calibration, sampling frequency, or raw acquisition timestamp for most variables. Such claims are deliberately absent.
10. Source `quality_flags` and `value_type` are carried forward; they require domain review before downstream use.

## Known limitation
The generic importer emits numeric non-identifier columns. For new datasets, review the emitted registry and exclude numeric codes, indices or categorical encodings if they appear. Add explicit column profiles and unit overrides before production ingestion. This is a conservative first integration, not a replacement for source-by-source scientific review.
