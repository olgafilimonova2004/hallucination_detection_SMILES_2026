# Solution

This submission uses the fixed `solution.py` pipeline and implements the model logic in `aggregation.py`, `probe.py`, and `splitting.py`. The final classifier is a stacked ensemble of three separate hallucination detectors: SAPLMA-style hidden-state probing, an ICR-style attention/representation consistency probe, and an LLM-check feature model. The three branches are trained independently first. Their predicted hallucination probabilities are then fused with a small logistic regression meta-model.

The code is designed to run from the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

Running `solution.py` produces two files:

```text
results.json
predictions.csv
```

`results.json` contains cross-validation metrics on the labelled `data/dataset.csv`. `predictions.csv` contains the final labels for the unlabelled competition file `data/test.csv`.

## What runs where

`solution.py` is the main entry point. It loads `data/dataset.csv`, concatenates each row as `prompt + response`, loads Qwen2.5-0.5B, extracts model internals, trains and evaluates the probe, then repeats feature extraction for `data/test.csv`. The batch size is set to 1 because Colab is memory constrained when hidden states, logits, and attentions are all requested.

`aggregation.py` controls feature extraction from the language model. It wraps the Qwen model loader so that each forward pass returns hidden states, logits, and attentions. From those internals it builds one feature vector per example. The vector has three parts: SAPLMA features, ICR features, and LLM-check features.

`probe.py` contains the actual classifier. It slices the combined feature vector back into the three branch-specific feature groups, trains one model per branch, and then trains a logistic regression stacker on out-of-fold branch probabilities.

`splitting.py` defines the evaluation split. It uses stratified 5-fold splits and also carves out a validation subset inside each training fold. This keeps class balance stable while allowing the probe to tune its probability threshold on validation data.

`evaluate.py` is fixed infrastructure. It calls `HallucinationProbe.fit`, evaluates train/validation/test folds, saves the metrics to `results.json`, and later writes the final competition predictions through `save_predictions`.

## Feature extraction

For every input, the text fed to Qwen is the original ChatML prompt followed by the model response. This is important because the hallucination signal depends on both the question/context and the answer being judged.

The extractor identifies the assistant response span by searching for the final ChatML assistant marker:

```text
<|im_start|>assistant
```

It also estimates the user span before the assistant response. These spans are used by the ICR and LLM-check features so they focus on the generated answer and its relation to the prompt, rather than treating the whole sequence as one undifferentiated block.

The final feature vector has 947 dimensions:

```text
896 SAPLMA hidden-state features
24 ICR layer features
27 LLM-check features
```

## Method 1: SAPLMA branch

The SAPLMA branch uses the hidden representation of the last real token from Qwen layer 15. Qwen2.5-0.5B has hidden size 896, so this branch contributes an 896-dimensional vector.

The classifier for this branch is a small neural network:

```text
896 -> 256 -> 128 -> 64 -> 1
```

It uses ReLU activations, dropout 0.3, binary cross-entropy with logits, Adam, L2 weight decay, and a small L1 penalty. This branch is meant to capture whether the final internal state of the model looks more like truthful or hallucinated examples.

## Method 2: ICR probe branch

The ICR branch measures consistency between attention and representation changes. For each transformer layer, it looks at response tokens, averages attention over heads, keeps attention directed to the user context and response region, and compares the top attended positions with the residual change between the previous and current hidden states.

For each layer, the code computes a Jensen-Shannon divergence between two distributions:

```text
attention scores over selected source tokens
projection scores of those source token representations onto the response-token residual update
```

This produces one score per transformer layer, giving 24 ICR features. Intuitively, the branch asks whether the tokens the model attends to are aligned with the directions that actually change the response representation. Hallucinated answers can show weaker or stranger alignment between attention and representational update.

The ICR classifier is an MLP:

```text
24 -> 128 -> 64 -> 32 -> 1
```

It uses batch normalization, LeakyReLU, dropout 0.3, Adam, and L2 regularization.

## Method 3: LLM-check branch

The LLM-check branch combines token-level confidence features with hidden-state geometry features.

The logit features are:

```text
response perplexity
maximum response-token entropy
top-k logit entropy
```

The hidden features are computed for each transformer layer. For response-token hidden states, the code centers the vectors, forms a token-token covariance-like matrix, stabilizes it with a small diagonal term, and takes the mean log singular value. This gives one hidden geometry score for each of the 24 transformer layers.

Together this branch has 27 features:

```text
3 logit features
24 hidden geometry features
```

The branch classifier is logistic regression with standardized inputs, L1 penalty, saga solver, `C=10`, and no class weighting. This keeps the LLM-check branch simple and sparse.

## Fusion strategy

The three branches are frozen before fusion. Each branch outputs a probability:

```text
p_ICR
p_LLM_check
p_SAPLMA
```

The meta-learner is a standard stacking logistic regression:

```text
P(hallucinated) = sigmoid(w0 + w1 * p_ICR + w2 * p_LLM_check + w3 * p_SAPLMA)
```

This model has four learned parameters: one intercept and one weight for each branch probability. It uses L2 regularization.

The stacker is trained only on out-of-fold predictions. Inside `HallucinationProbe.fit`, the training data is split with stratified folds. For each fold, the three branch models are trained on the fold's training portion and predict probabilities for the held-out portion. Those held-out probabilities form the meta-training matrix. This avoids training the stacker on branch predictions from models that already saw the same examples.

After the meta-model is fitted, the three branch models are retrained on the full available training data. At prediction time, those final branch models produce the three probabilities, and the meta-model converts them into the final hallucination probability.

## How results.json is produced

After feature extraction on `data/dataset.csv`, `solution.py` calls:

```python
splits = split_data(y, df)
fold_results = run_evaluation(splits, X, y, HallucinationProbe)
save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)
```

`run_evaluation` trains a fresh `HallucinationProbe` for each split, reports baseline metrics and probe metrics, and returns a list of per-fold dictionaries. `save_results` writes those fold metrics plus averaged metrics to `results.json`.

The reported test metrics in `results.json` refer to held-out folds created from the labelled training dataset. They are not computed from `data/test.csv`, because `data/test.csv` has no labels.

## How predictions.csv is produced

After saving `results.json`, `solution.py` loads `data/test.csv`, extracts the same 947-dimensional feature vector for each test example, and fits one final `HallucinationProbe` on the labelled dataset.

The final line that writes the submission file is:

```python
save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)
```

`save_predictions` calls `final_probe.predict(X_test)` and writes:

```text
id,label
```

to `predictions.csv`. The labels use the competition convention:

```text
0 = truthful
1 = hallucinated
```

## Notes on Colab

This solution asks Qwen for hidden states, logits, and attentions. That is more memory intensive than ordinary text generation. For this reason `solution.py` uses `BATCH_SIZE = 1`. It is slower, but it avoids the common Colab failure mode where the runtime crashes during attention extraction.

The notebook `Run_Colab.ipynb` clones or updates the repository, installs dependencies, runs `solution.py`, and displays the first rows of `predictions.csv`.
