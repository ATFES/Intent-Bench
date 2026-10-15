"""Optional diagnostics for migrated native CPU adapters."""


def report_selection(observed):
    trials = tuple(((dict(candidate.configuration), candidate.elapsed_ms) for candidate in observed.candidates if candidate.elapsed_ms is not None))
    reuse = tuple((f'{entry.layer}/{entry.scope}/{entry.stage}={entry.hit}' for entry in observed.caches if entry.layer != 'native_compilation'))
    print(f'mojo: selected {dict(observed.configuration)}; tuning_trials_ms={trials}; history_unavailable={observed.candidate_history_unavailable}; reuse={reuse}', flush=True)
