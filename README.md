# AMPlify

AMPlify is an attentive deep learning model for antimicrobial peptide prediction.

### Dependencies

* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12
* Numpy <1.17
* Pandas
* Scikit-learn
* Biopython
* h5py <3

### Installation

1. Create a new `conda` environment:
```
conda create -n amplify python=3.6
```

2. Activate the environment:
```
conda activate amplify
```

3. Install AMPlify in the environment:
```
conda install -c bioconda amplify
```
`train_amplify` and `AMPlify` can now be run. See usage information below.

4. To deactivate an active environment, use:
```
conda deactivate
```

### Datasets

Datasets for training and testing are stored in the `data` folder. Please specify the directory if you would like to use those datasets for training or testing the model.
* Balanced training set: `AMPlify_AMP_train_common.fa` + `AMPlify_non_AMP_train_balanced.fa`
* Balanced test set: `AMPlify_AMP_test_common.fa` + `AMPlify_non_AMP_test_balanced.fa`
* Imbalanced training set: `AMPlify_AMP_train_common.fa` + `AMPlify_non_AMP_train_imbalanced.fa`
* Imbalanced test set: `AMPlify_AMP_test_common.fa` + `AMPlify_non_AMP_test_imbalanced.fa`


### Pre-trained sub-models

Weights for the pre-trained balanced/imbalanced sub-models are stored in the `models` folder.
* Balanced model: The one proposed in the AMPlify publication; Trained on the balanced training set; Suitable to use for predicting putative AMPs from a relatively curated candidate set (e.g. sequences already filtered by length/charge and through sequence homology search).
* Imbalanced model: Trained on the imbalanced training set; Suitable for situations where the number of non-AMPs in the input sequence set is far greater than that of AMPs (e.g. a large transcriptomics sequence database).

### Train

Usage: `train_amplify [-h] -amp_tr AMP_TR -non_amp_tr NON_AMP_TR [-amp_te AMP_TE] [-non_amp_te NON_AMP_TE] [-sample_ratio {balanced,imbalanced}] -out_dir OUT_DIR -model_name MODEL_NAME
`
```
optional arguments:
  -h, --help            Show this help message and exit
  -amp_tr AMP_TR        Training AMP set, fasta file
  -non_amp_tr NON_AMP_TR
                        Training non-AMP set, fasta file
  -amp_te AMP_TE        Test AMP set, fasta file (optional)
  -non_amp_te NON_AMP_TE
                        Test non-AMP set, fasta file (optional)
  -sample_ratio {balanced,imbalanced}
                        Whether the training set is balanced
                        or not (balanced by default, optional)
  -out_dir OUT_DIR      Output directory
  -model_name MODEL_NAME
                        File name of trained model weights
```
Example: `train_amplify -amp_tr ../data/AMPlify_AMP_train_common.fa -non_amp_tr ../data/AMPlify_non_AMP_train_balanced.fa -amp_te ../data/AMPlify_AMP_test_common.fa -non_amp_te ../data/AMPlify_non_AMP_test_balanced.fa -out_dir ../models/ -model_name model`

Expected output: 1) The model weights trained using the specified data; 2) Test set performance, if test sequences have been specified.

### Predict

Usage: `AMPlify [-h] [-m {balanced,imbalanced}] -s SEQS [-od OUT_DIR] [-of {txt,tsv}] [-sub {on,off}] [-att {on,off}]`
```
optional arguments:
  -h, --help            Show this help message and exit
  -m {balanced,imbalanced}, --model {balanced,imbalanced}
                        Balanced or imbalanced model (balanced by default, optional)
  -s SEQS, --seqs SEQS  Sequences for prediction, fasta file
  -od OUT_DIR, --out_dir OUT_DIR
                        Output directory (optional)
  -of {txt,tsv}, --out_format {txt,tsv}
                        Output format, txt or tsv (tsv by default, optional)
  -sub {on,off}, --sub_model {on,off}
                        Whether to output sub-model results, on or off (off by
                        default, optional)
  -att {on,off}, --attention {on,off}
                        Whether to output attention scores, on or off (off by
                        default, optional)
```
Example: `AMPlify -s ../data/AMPlify_AMP_test_common.fa`

Expected output: Predicted probability scores, AMPlify log scaled scores, and classes of the input sequences. The AMPlify log scaled score is calculated as `-10*log10(1-Probability_score)`. Results for invalid sequences will be filled with `NA`.

### AMP discovery

Additional scripts and data for our AMP discovery pipeline are provided in the `auxiliary` folder. Parameters for GMAP and MAKER2 are described in the Methods section of the manuscript.

### Author

Chenkai Li (cli@bcgsc.ca)

### Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact us.

### Citation

If you use AMPlify in your work, please cite [our publication](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-022-08310-4):
> Li, C., Sutherland, D., Hammond, S.A. et al. AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens. BMC Genomics 23, 77 (2022). https://doi.org/10.1186/s12864-022-08310-4
--------------------------------------------------------------------------------
