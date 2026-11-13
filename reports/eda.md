# Exploratory data analysis

Aggregate counts only. Per-record rows are in the details CSV (gitignored).

## Files

| file | rows | status |
|---|---|---|
| Africa2024-_accepted.csv | 6 | read |
| Africa2025+_accepted.csv | 183 | read |
| Asia2023-_accepted.csv | 118 | read |
| Asia2024+_accepted.csv | 777 | read |
| CBHC2026_accepted.csv | 212 | read |
| GLBIO2023-_accepted.csv | 223 | read |
| GLBIO2024+_accepted.csv | 322 | read |
| ISMB2020_accepted.csv | 1575 | read |
| ISMB2022_accepted.csv | 1133 | read |
| ISMB2024_accepted.csv | 1269 | read |
| ISMB2026_accepted.csv | 1598 | read |
| ISMBECCB2019_accepted.csv | 1424 | read |
| ISMBECCB2021_accepted.csv | 905 | read |
| ISMBECCB2023_accepted.csv | 1634 | read |
| ISMBECCB2025_accepted.csv | 1985 | read |
| LATAM2023-_accepted.csv | 109 | read |
| LATAM2024+_accepted.csv | 319 | read |
| RSGDREAM2023-_accepted.csv | 226 | read |
| RSGDREAM2024+_accepted.csv | 78 | read |
| SisterConferences_accepted.csv | 367 | read |
| UK_accepted.csv | 215 | read |

## Records per conference

14,678 rows loaded from 21 files.

| conference | rows |
|---|---|
| APBJC2024 | 529 |
| Africa2021 | 6 |
| Africa2025 | 183 |
| Asia2025 | 248 |
| BiGEvo2025 | 92 |
| CBHC2026 | 212 |
| GIWAsia2023 | 118 |
| GLBIO2021 | 59 |
| GLBIO2023 | 164 |
| GLBIO2024 | 163 |
| GLBIO2025 | 159 |
| GreatPlains2026 | 116 |
| ISMB/ECCB 2019 | 1424 |
| ISMB2020 | 305 |
| ISMB2022 | 1133 |
| ISMB2024 | 1269 |
| ISMB2026 | 1598 |
| ISMBECCB2021 | 905 |
| ISMBECCB2023 | 1634 |
| ISMBECCB2025 | 1985 |
| InCoB2026 | 159 |
| LA2022 | 109 |
| LATAM2024 | 319 |
| RSGDREAM2022 | 91 |
| RSGDREAM2023 | 135 |
| RSGDREAM2024 | 78 |
| UK2026 | 215 |
| ismb2020 | 1270 |

## Filtering steps

Same steps and order as `train.load_training_data` (`common.filter_training_data`).

| step | remaining | removed |
|---|---|---|
| loaded | 14678 | 0 |
| after reject removal | 14405 | 273 |
| after unmapped removal | 9336 | 5069 |
| after duplicate removal | 9103 | 233 |

## COSIs by conference

Training rows per COSI in `cosis.txt` and conference label (labels as they appear in the data).

| COSI | ISMB/ECCB 2019 | ISMB2020 | ismb2020 | ISMBECCB2021 | ISMB2022 | ISMBECCB2023 | ISMB2024 | ISMBECCB2025 | ISMB2026 | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| 3DSIG | 121 | 1 | 65 | 58 | 70 | 132 | 80 | 138 | 105 | 770 |
| BioInfo-Core | 0 | 7 | 4 | 1 | 10 | 15 | 9 | 17 | 7 | 70 |
| BioVis | 33 | 8 | 18 | 27 | 23 | 28 | 15 | 35 | 19 | 206 |
| BOKR | 16 | 5 | 21 | 11 | 24 | 12 | 12 | 29 | 33 | 163 |
| BOSC | 70 | 0 | 0 | 66 | 59 | 62 | 55 | 72 | 65 | 449 |
| CAMDA | 20 | 7 | 15 | 18 | 27 | 15 | 16 | 21 | 29 | 168 |
| CompMS | 17 | 13 | 49 | 27 | 22 | 28 | 34 | 40 | 19 | 249 |
| CSI | 0 | 0 | 0 | 0 | 0 | 0 | 47 | 55 | 51 | 153 |
| Education | 20 | 8 | 21 | 18 | 16 | 19 | 15 | 28 | 19 | 164 |
| EvolCompGen | 82 | 0 | 52 | 40 | 45 | 84 | 68 | 82 | 61 | 514 |
| Function | 50 | 16 | 30 | 21 | 32 | 40 | 47 | 59 | 41 | 336 |
| HiTSeq | 114 | 3 | 54 | 67 | 63 | 119 | 81 | 87 | 85 | 673 |
| iRNA | 78 | 12 | 39 | 44 | 58 | 67 | 61 | 72 | 51 | 482 |
| MICROBIOME | 54 | 6 | 35 | 28 | 42 | 91 | 63 | 88 | 68 | 475 |
| MLCSB | 118 | 4 | 86 | 98 | 156 | 252 | 194 | 312 | 263 | 1483 |
| NetBio | 74 | 3 | 57 | 55 | 56 | 98 | 53 | 77 | 55 | 528 |
| RegSys | 90 | 8 | 70 | 60 | 63 | 94 | 77 | 90 | 77 | 629 |
| SysMod | 53 | 7 | 30 | 40 | 39 | 80 | 35 | 73 | 60 | 417 |
| TextMining | 0 | 9 | 28 | 13 | 17 | 14 | 18 | 34 | 24 | 157 |
| TransMed | 73 | 4 | 35 | 47 | 66 | 128 | 70 | 124 | 101 | 648 |
| VarI | 53 | 7 | 40 | 27 | 38 | 61 | 40 | 56 | 47 | 369 |
| Total | 1136 | 128 | 749 | 766 | 926 | 1439 | 1090 | 1589 | 1280 | 9103 |

Most recent ISMB conference: ISMB2026 (default)

### No training rows at all

None

### No training rows in ISMB2026, but some earlier

None

## Unmapped tracks

Tracks that map to no COSI, after reject removal. Track names are shown as they appear in the data, so names that differ only in spacing or case are listed separately.

| conference | track | rows |
|---|---|---|
| ISMB/ECCB 2019 | General Comp Bio | 192 |
| ISMB/ECCB 2019 | Text Mining (Special Session) | 31 |
| ISMB/ECCB 2019 | Comparative and Functional Genomics | 13 |
| ISMB/ECCB 2019 | Macromolecular Sequence, Structure, and Function | 13 |
| ISMB/ECCB 2019 | Studies of Phenotypes and Clinical Applications | 13 |
| ISMB/ECCB 2019 | Systems Biology and Networks | 9 |
| ISMB/ECCB 2019 | Population Genomics and Molecular Evolution | 6 |
| ISMB/ECCB 2019 | Bioinformatics of Microbes and Microbiomes | 3 |
| ISMB/ECCB 2019 | Genomic Variation Analysis | 3 |
| ISMB/ECCB 2019 | Bioinformatics Education | 1 |
| ISMB/ECCB 2019 | Genome Privacy and Security | 1 |
| ISMB2020 | SST01 | 27 |
| ISMB2020 | ODSS | 18 |
| ISMB2020 | WEB | 18 |
| ISMB2020 | SST02 | 12 |
| ISMB2020 | SST06 | 10 |
| ISMB2020 | SST07 | 9 |
| ISMB2020 | SST04 | 8 |
| ISMB2020 | SST03 | 7 |
| ISMB2020 | SST05 | 7 |
| ISMB2020 | Testing | 5 |
| ISMB2020 | COVID-19 | 4 |
| ISMB2020 | Green | 4 |
| ISMB2020 | Tech Track | 3 |
| ISMB2020 | (blank) | 1 |
| ismb2020 | General Comp Bio | 84 |
| ismb2020 | COVID-19 | 71 |
| ismb2020 | Final Accept Proceedings | 65 |
| ismb2020 | Fixed Time | 8 |
| ismb2020 | Systems Immunology (Special Session) | 8 |
| ismb2020 | Keynotes | 5 |
| ismb2020 | SCANGEN (Special Session) | 4 |
| ismb2020 | (blank) | 1 |
| ismb2020 | SST02 | 1 |
| Africa2021 | Late Breaking Posters | 6 |
| GLBIO2021 | Posters | 57 |
| GLBIO2021 | General | 1 |
| GLBIO2021 | Genotype to Phenotype in Model and Non-Model Organisms | 1 |
| ISMBECCB2021 | GenCompBio | 63 |
| ISMBECCB2021 | Special Session 05 | 31 |
| ISMBECCB2021 | ODSS | 28 |
| ISMBECCB2021 | Tech Track | 10 |
| ISMB2022 | GenCompBio | 75 |
| ISMB2022 | SSCI | 34 |
| ISMB2022 | SSDA | 26 |
| ISMB2022 | TechTrack | 14 |
| ISMB2022 | WEB | 10 |
| ISMB2022 | NIH-NSFSH | 8 |
| ISMB2022 | SSGPAS | 8 |
| ISMB2022 | Equity | 6 |
| ISMB2022 | BoF | 5 |
| ISMB2022 | Keynotes | 5 |
| ISMB2022 | JPI | 4 |
| ISMB2022 | SSCOVID | 3 |
| ISMB2022 | (blank) | 1 |
| LA2022 | la | 109 |
| RSGDREAM2022 | rsg | 84 |
| RSGDREAM2022 | dream | 7 |
| GIWAsia2023 | Posters | 72 |
| GIWAsia2023 | general | 46 |
| GLBIO2023 | general | 164 |
| ISMBECCB2023 | GenCompBio | 162 |
| ISMBECCB2023 | SSBiF | 23 |
| ISMBECCB2023 | Equity | 3 |
| ISMBECCB2023 | SSHFSP | 2 |
| RSGDREAM2023 | RSG | 125 |
| RSGDREAM2023 | DREAM | 10 |
| APBJC2024 | APBJC | 472 |
| APBJC2024 | ASCS | 57 |
| GLBIO2024 | gencompbio | 149 |
| GLBIO2024 | sabbd | 10 |
| GLBIO2024 | cosa | 2 |
| GLBIO2024 | raimbda | 2 |
| ISMB2024 | GenCompBio | 121 |
| ISMB2024 | SCS | 17 |
| ISMB2024 | BiC | 16 |
| ISMB2024 | DigAg | 14 |
| ISMB2024 | EDCBR | 5 |
| LATAM2024 | LATAM | 288 |
| LATAM2024 | LASCS | 31 |
| RSGDREAM2024 | RSG | 78 |
| Africa2025 | GenCompBio | 170 |
| Africa2025 | AfSCS | 13 |
| Asia2025 | GenCompBio | 211 |
| Asia2025 | ASCS | 37 |
| BiGEvo2025 | BiGEvo | 92 |
| GLBIO2025 | GenCompBio | 157 |
| GLBIO2025 | MLIGR | 2 |
| ISMBECCB2025 | GenCompBio | 207 |
| ISMBECCB2025 | SCS | 52 |
| ISMBECCB2025 | UK | 32 |
| ISMBECCB2025 | TechTrack | 16 |
| ISMBECCB2025 | Equity | 8 |
| CBHC2026 | GenCompBio | 212 |
| GreatPlains2026 | GreatPlains | 116 |
| ISMB2026 | GenCompBio | 121 |
| ISMB2026 | USA | 34 |
| ISMB2026 | SCS | 29 |
| ISMB2026 | Tutorials | 17 |
| ISMB2026 | DREAM | 16 |
| ISMB2026 | TechTrack | 14 |
| ISMB2026 | Equity | 9 |
| InCoB2026 | InCoB | 148 |
| InCoB2026 | APBSCS | 11 |
| UK2026 | GenCompBio | 215 |

Total: 5,069 rows (the "after unmapped removal" step removes 5,069).

## Duplicate titles

Rows that reach the dedupe step. The key is the title in lowercase without punctuation or spaces. Same-COSI classes span all years, as train.py's dedupe does; the multi-COSI class is counted within a conference year. Titles with an empty key are left out of the two key classes and counted as placeholders; the exact-title class counts them as equal, as train.py does.

| class | groups | rows | extra rows |
|---|---|---|---|
| exact title, same COSI | 212 | 445 | 233 |
| same key, same COSI | 227 | 477 | 250 |
| same key, different COSIs (same year) | 14 | 41 | - |

Multi-COSI groups with a placeholder title: 5.

## Placeholder titles

Titles that are agenda placeholders (D11 on #14): an exact key (tba, tbd, tbc, qa, closingremarks, openingremarks, welcome), paper + number (e.g. paper12), an empty key, or TBA/TBD/TBC as a separate word. Rows after reject removal; titles are not shown. Detection only: removing them from training is #8.

Placeholder rows: 80 after reject removal; 31 reach training.

| rule | rows |
|---|---|
| list | 39 |
| paperN | 6 |
| word | 28 |
| empty | 7 |

| conference | rows | in_training |
|---|---|---|
| ISMB2020 | 65 | 24 |
| ismb2020 | 1 | 0 |
| ISMB2022 | 13 | 7 |
| APBJC2024 | 1 | 0 |

| COSI | rows | in_training |
|---|---|---|
| BOSC | 2 | 1 |
| CompMS | 3 | 2 |
| Education | 9 | 5 |
| Function | 11 | 8 |
| iRNA | 9 | 3 |
| MLCSB | 1 | 1 |
| NetBio | 2 | 1 |
| RegSys | 6 | 1 |
| SysMod | 2 | 1 |
| TextMining | 3 | 2 |
| TransMed | 4 | 2 |
| VarI | 8 | 4 |
| (unmapped) | 20 | 0 |

## Missing or empty fields

Share of rows whose field is empty, blank or absent, measured before blanks are filled.

### All loaded rows

| conference | rows | title | keywords | abstract |
|---|---|---|---|---|
| APBJC2024 | 529 | 0.0% | 0.0% | 0.0% |
| Africa2021 | 6 | 0.0% | 0.0% | 0.0% |
| Africa2025 | 183 | 0.0% | 0.0% | 0.0% |
| Asia2025 | 248 | 0.0% | 0.4% | 0.0% |
| BiGEvo2025 | 92 | 0.0% | 1.1% | 0.0% |
| CBHC2026 | 212 | 0.0% | 0.0% | 0.0% |
| GIWAsia2023 | 118 | 0.0% | 0.0% | 0.0% |
| GLBIO2021 | 59 | 0.0% | 0.0% | 0.0% |
| GLBIO2023 | 164 | 0.0% | 0.0% | 0.0% |
| GLBIO2024 | 163 | 0.0% | 0.0% | 0.0% |
| GLBIO2025 | 159 | 0.0% | 0.0% | 0.0% |
| GreatPlains2026 | 116 | 0.0% | 0.0% | 0.0% |
| ISMB/ECCB 2019 | 1424 | 0.0% | 82.5% | 0.0% |
| ISMB2020 | 305 | 0.0% | 100.0% | 46.2% |
| ISMB2022 | 1133 | 0.5% | 16.3% | 4.9% |
| ISMB2024 | 1269 | 0.0% | 0.0% | 0.0% |
| ISMB2026 | 1598 | 0.0% | 0.0% | 0.0% |
| ISMBECCB2021 | 905 | 0.0% | 4.2% | 0.8% |
| ISMBECCB2023 | 1634 | 0.0% | 0.0% | 0.1% |
| ISMBECCB2025 | 1985 | 0.0% | 0.0% | 0.0% |
| InCoB2026 | 159 | 0.0% | 0.0% | 0.0% |
| LA2022 | 109 | 0.0% | 0.0% | 0.0% |
| LATAM2024 | 319 | 0.0% | 0.0% | 0.0% |
| RSGDREAM2022 | 91 | 0.0% | 0.0% | 0.0% |
| RSGDREAM2023 | 135 | 0.0% | 0.0% | 0.0% |
| RSGDREAM2024 | 78 | 0.0% | 0.0% | 0.0% |
| UK2026 | 215 | 0.0% | 0.0% | 0.0% |
| ismb2020 | 1270 | 0.0% | 1.7% | 0.9% |
| All | 14678 | 0.0% | 11.8% | 1.5% |

### Rows that reach training

| conference | rows | title | keywords | abstract |
|---|---|---|---|---|
| ISMB/ECCB 2019 | 1136 | 0.0% | 84.2% | 0.0% |
| ISMB2020 | 128 | 0.0% | 100.0% | 49.2% |
| ISMB2022 | 926 | 0.4% | 9.5% | 3.2% |
| ISMB2024 | 1090 | 0.0% | 0.0% | 0.0% |
| ISMB2026 | 1280 | 0.0% | 0.0% | 0.0% |
| ISMBECCB2021 | 766 | 0.0% | 0.0% | 0.0% |
| ISMBECCB2023 | 1439 | 0.0% | 0.0% | 0.0% |
| ISMBECCB2025 | 1589 | 0.0% | 0.0% | 0.0% |
| ismb2020 | 749 | 0.0% | 0.0% | 0.0% |
| All | 9103 | 0.0% | 12.9% | 1.0% |
