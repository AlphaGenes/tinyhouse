# Changelog

## [X.Y.Z] - 2023-XX-XX

**New features**

- TODO

**Bug fixes**:

- Fix pedigree founder error - sire genotype status was used twice instead of sire and dam genotype status.
  ({issue}`12`, {user}`augustusgrant`)

- Fix individual order in output with the `-onlykeyed` option, reported in AlphaPeel
  (https://github.com/AlphaGenes/AlphaPeel/issues/43). ({pr}`10`, {user}`AudreyAAMartin`)

**Maintenance**:

- Added option to call help

- We have revised and polished the documentation.

**Breaking changes**:

- TODO - mention the change/consolidation of argument names.
  
- There has been an unfortunate split in the codebase between AlphaImpute2 and AlphaPlantImpute2
  in `writePhase()` as reported at AlphaImpute2 repo (https://github.com/AlphaGenes/AlphaImpute2/issues/16).
  We decided to revert back to the initial form used before the code split, which will break
  AlphaPlantImpute2 for now, but that will be fixed later. ({issue}`8`, {issue}`1`, {pr}`4, {user}`XingerTang`)
