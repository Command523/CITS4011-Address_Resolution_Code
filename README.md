# Neural Network Approaches for Australian Address Resolution

Code for the Honours dissertation of the same name (Ethan Yong, 2026).

The notebooks build a test set of real mutated Australian addresses, generate
training data, and train a two-stage matcher against G-NAF Core. Stage 1 is a
retriever (a bi-encoder) that finds candidate records. Stage 2 is a re-ranker
(a cross-encoder) that picks one candidate or deems the address unmatchable.
The matcher is compared with five rule-based baselines and with a second
re-ranker built on CANINE.

## Notebooks, in run order

| Notebook | What it does |
|---|---|
| `1_harvest_test_set.ipynb` | Finds addresses in the court decisions of the Open Australian Legal Corpus, matches them to G-NAF Core, labels the kinds of mistake each one contains, and writes the test set. It also loads G-NAF Core into the DuckDB database that every later notebook reads. |
| `2_synthesise_training_data.ipynb` | Generates mutated addresses from G-NAF Core records, with mistake rates measured on the harvested addresses. No test-set answer is used as a training record. |
| `3_train_retriever.ipynb` | Stage 1. Trains the retriever, embeds every G-NAF Core record, and measures how often the correct record is among the top K candidates. |
| `4_train_reranker.ipynb` | Stage 2. Trains the MiniLM re-ranker on the retriever's candidates, fits the threshold below which an address is deemed unmatchable, and evaluates the two stages together. |
| `5_baselines.ipynb` | Five rule-based string-matching methods, over every record and again inside a street-number block. |
| `6_train_canine_reranker.ipynb` | A second Stage 2 re-ranker built on CANINE, a character-level model, compared with the MiniLM re-ranker on the same candidates. |

`tools/baseline_metrics.py` holds the metric functions that notebooks 5 and 6
import.

The labels of the test set were also checked by hand after the harvest. That
review is not part of these notebooks.

## Data (not included)

All three sources are public.

- G-NAF Core, May 2026 release, Geoscape Australia:
  https://docs.geoscape.com.au/projects/gnafcore_info/en/stable/
- G-NAF, May 2026 release, used only for its authority code tables (street
  types, street suffixes, flat and level types):
  https://data.gov.au/data/dataset/geocoded-national-address-file-g-naf
- Open Australian Legal Corpus, version 7.1.0, Umar Butler:
  https://huggingface.co/datasets/isaacus/open-australian-legal-corpus

Put them in a folder named `data/` beside the notebooks, or set the
`HARVEST_BASE` environment variable to the folder that holds them. Notebook 1
expects these paths inside that folder:

    GNAF/GNAF/g-naf_may26_allstates_gda2020_psv_1023/G-NAF/G-NAF MAY 2026/Authority Code/
    GNAF/GNAF Core/G-NAF Core/G-NAF Core MAY 2026/Standard/
    Open Australian Legal Corpus/corpus.jsonl

## Running

Run the notebooks in order, from this folder. Notebook 1 creates
`gnaf.duckdb`, and each notebook writes its results to a folder beside it
(`output/`, `output_synthesis/`, `output_stage1/`, `output_stage2/`,
`output_baseline/`, `output_canine/`). Only one notebook can hold
`gnaf.duckdb` open at a time.

The longest steps are the corpus harvest and the G-NAF Core load in notebook 1
(hours on the first run), training the retriever (about two hours), embedding
every G-NAF Core record (30 to 60 minutes) and training the CANINE re-ranker
(about an hour). Each configuration cell lists environment switches, such as
`S1_ALLOW_CACHE=1`, that reuse results already on disk.

Notebooks 3, 5 and 6 also read saved copies of result files (the folders whose
names start with `results_labels_`), which are not included. Their paths are
set in each notebook's configuration cell.

## Environment

Python 3.13.2, with the package versions in `requirements.txt`. All work ran
on an Apple M3 with 16 GB of memory, and training used PyTorch's MPS backend
with 32-bit numbers. Training on MPS is not repeatable bit for bit, so a
re-run gives close but not identical models.

Every random draw uses seed 4640. The one exception is the pools from which
the retriever's hard and very hard wrong records were drawn. The retriever in
the dissertation was trained on pools drawn without a seed, so those pools
cannot be drawn again exactly.

## Saved outputs

The saved cell outputs are from the runs behind the dissertation's results.
Warnings, progress bars and install messages were removed. Cells 3, 4, 9b and
21 of notebook 1 have no saved output.

## Data licences

This research was developed using G-NAF Core © Geoscape Australia 2023
Copyright and Disclaimer Notice. Licensed by Geoscape Australia under the Open
G-NAF Core End User Licence Agreement.

This research was developed using G-NAF © Geoscape Australia licensed by the
Commonwealth of Australia under the Open Geo-coded National Address File
(G-NAF) End User Licence Agreement.

The Open Australian Legal Corpus is licensed under the Creative Commons
Attribution 4.0 International Licence.
