To convert the dataset to the version we used for our experiments open the csaws_creation folder and run the script generate_dataset.py

```
data 
│ 
├── CsawS 
│     │
│     ├ anonymized_dataset -> patient folders with all the screenings and annotations (tumors - expert 1)
│     │
│     ├ test_data
│     │     │
│     │     ├ anonymized_dataset -> patient folders with all the screenings
│     │     │
│     │     ├ annotator_1 -> annotations from expert 1
│     │     │
│     │     ├ annotator_2 -> annotations from expert 2
│     │     │
│     │     └ annotator_3 -> annotations from expert 3
│     │
│     └ training_random_splits.json -> the splits we used for our experiments (can be ommited)
│ 
├── csaws_creation -> scripts to convert the dataset to the version we used for our experiments
│ 
└── README.md -> current file
```
