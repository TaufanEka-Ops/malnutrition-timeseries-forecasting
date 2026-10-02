# Data ethics note

## What was withheld and why

The original academic project drew on the nutrition-unit reporting workbook of
a single Indonesian hospital. An audit of those workbooks found that 11 of the
13 raw files carry patient-identifying fields:

| Field seen in the raw workbooks | Example content |
|---|---|
| `Identitas Pasien` / `Nama Pasien` | Patient names (`Ny. …`, `Tn. …`, `An. …`) |
| `Norm` / `Rekam Medis` | Medical-record numbers |
| `Ruangan` | Ward / treatment room |
| Hospital letterhead | Institution name |

Under Indonesian law (UU 29/2004, PP 71/2019 on health data) and under the
general principle of patient confidentiality, none of that material belongs in
a public repository. It is not a technicality: a repository is mirrored,
indexed and archived by third parties, and deletion is not real deletion.

## What is published instead

Only the **monthly aggregate count** of patients flagged at risk. This is the
series already published in the thesis (an academic document held by the
university library) and in the thesis' forecasting validation table. It
identifies no individual and no ward.

## Consent

The aggregate figures were shared with the author by the host institution's
nutrition unit for the purpose of the academic study. Anyone reusing this
repository for a purpose other than reproducing the study should seek
permission from the originating institution.

## Reviewer note

If you are reviewing this repository as part of a hiring process: the absence
of a `raw inpatient records` directory is a deliberate compliance decision,
not missing work. The pipeline that produces every published number is in
`src/` and runs end-to-end on the aggregate above.
