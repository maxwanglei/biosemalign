# BioSemAlign PK Pack (Example External Plugin)

A minimal third-party BioSemAlign plugin distribution advertising the
`biosemalign.plugins` entry-point group (`pk = "biosemalign_pk_pack:register"`).

All `ToyPK` parameter concepts (`PK1` `Cmax`, `PK2` `AUC0-inf`, `PK3` `Clearance`)
are synthetic and contain no real terminology content.

## Contents

- `specs/toypk.yaml`: identity-level target spec (`Parameter`) with `specimen` context loss.
- `fields.yaml`: declares `specimen` (`attribute`, `use: [key, clean, resolve, check]`) and blinded `study_arm`.
- `cleaners.py`: `pk.normalize_parameter` (`NormalizePKParameterCleaner`), extracting trailing specimen qualifiers such as `"Cmax (plasma)"` into `sau.attributes.specimen`.
- `checks.py`: `pk.specimen_check` (`PKSpecimenCheck`), warning when a mapped PK parameter lacks `specimen`.
- `profiles/toypk_parameter_coding.yaml`: two-tier profile (`exact` -> `cleaned`) using the registered pack.

## Verify with pytest

```bash
pip install -c ../../constraints.txt --no-deps .
pytest tests
```
