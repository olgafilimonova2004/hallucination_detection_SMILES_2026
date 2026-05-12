# Solution

This submission uses the fixed `solution.py` pipeline and implements the model logic in `aggregation.py`, `probe.py`, and `splitting.py`. The final classifier is a stacked ensemble of three branches: SAPLMA-style hidden-state probing, an ICR-style attention/representation consistency probe, and an LLM-Check feature model. I built this three-branch pipeline because hallucination is not expressed through one single hidden-state signal. The branches are meant to capture complementary information from the same Qwen2.5-0.5B forward pass.

The code is designed to run from the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

Running `solution.py` produces:

```text
results.json
predictions.csv
```

`results.json` contains cross-validation metrics on the labelled `data/dataset.csv`. `predictions.csv` contains the final labels for the unlabelled competition file `data/test.csv`.

## Components Modified

The modeling changes are concentrated in the expected files:

```text
aggregation.py
probe.py
splitting.py
```

`aggregation.py` extracts a 947-dimensional feature vector from Qwen internals. `probe.py` trains the three branch classifiers and the final stacking meta-learner. `splitting.py` defines the stratified evaluation folds used by the official evaluation code.

I also added `Run_Colab.ipynb` for reproducibility and `SOLUTION.md` for this report. `solution.py` keeps the official pipeline structure and uses `BATCH_SIZE = 1` so Colab does not crash while extracting hidden states, logits, and attentions.

## What Runs Where

`solution.py` is the main entry point. It loads `data/dataset.csv`, concatenates each row as `prompt + response`, loads Qwen2.5-0.5B, extracts model internals, trains and evaluates the probe, then repeats feature extraction for `data/test.csv`.

`aggregation.py` wraps the Qwen model loader so that each forward pass returns hidden states, logits, and attentions. It also infers the response span and user span from ChatML markers. From this single model pass it builds one feature vector per example:

```text
896 SAPLMA hidden-state features
24 ICR features
27 LLM-Check features
947 total features
```

`probe.py` slices this combined vector into branch-specific views, trains each branch, creates out-of-fold probabilities for stacking, and finally trains a logistic regression meta-learner.

`splitting.py` uses stratified 5-fold splitting. Inside each fold it also creates a validation subset, which is used for threshold tuning during evaluation.

`evaluate.py` is fixed infrastructure. It calls `HallucinationProbe.fit`, reports train/validation/test metrics for labelled folds, saves `results.json`, and later writes `predictions.csv` through `save_predictions`.

## Publications and Concepts Used

The final pipeline combines ideas from the following methods:

| Method | Publication | Main concept | What it captures |
|---|---|---|---|
| Geometry of Truth | Marks and Tegmark, Oct 2023, arXiv:2310.06824 | PCA diagnostics on hidden states | Which layer and token position contain the strongest truthfulness signal |
| SAPLMA | Azaria and Mitchell, Apr 2023, arXiv:2304.13734 | MLP on a single hidden-state vector | Static truthfulness signal in one selected layer/token representation |
| ICR Probe | Zhang, Hu, Zhang, Zhang, Wan, Jul 2025, arXiv:2507.16488 | Cross-layer residual stream dynamics | Whether attention and hidden-state updates align while the response is processed |
| LLM-Check | Sriramanan, Bharti, Sadasivan, Saha, Kattakinda, Feizi, NeurIPS 2024 | Eigenvalue/geometry analysis of response hidden states | Response-level geometric structure across generated tokens |

The final submission uses SAPLMA, ICR Probe, and LLM-Check as the three active branches. Geometry of Truth was used as a diagnostic method rather than as a final classifier.

## Method 0: Geometry of Truth Diagnostics

Before selecting the SAPLMA layer and token position, I ran PCA-style diagnostics inspired by Marks and Tegmark, "The Geometry of Truth" (Oct 2023, arXiv:2310.06824). The idea is to test whether true and false examples separate in hidden-state space:

```text
X_l = [h_l(x_1), h_l(x_2), ..., h_l(x_n)] in R^(n x 896)
X_centered = X_l - mean(X_l)
PCA_2D(X_centered) -> visual separation
silhouette_score(PCA_2D, labels)
```

The diagnostics did not become a final detector. The separation was weak and noisy, probably because the dataset is small, answers vary in length and style, and Qwen2.5-0.5B is much smaller than the models studied in the original paper. However, the diagnostic was still useful. It showed that the last-token representation was more informative than mean pooling in this setup, and `layer_rankings.csv` ranked layer 15 best by max silhouette:

```text
layer 15: max_silhouette = 0.0456015989
layer 12: max_silhouette = 0.0386708304
layer 14: max_silhouette = 0.0346653871
```

That is why the SAPLMA branch uses the last real token from layer 15.

## Method 1: SAPLMA Branch

The SAPLMA branch follows Azaria and Mitchell's Statement Accuracy Prediction from Language Model Activations idea: a language model's internal activations can contain information about whether an answer is true or false.

In the implementation, for each example I extract:

```text
h = hidden_state[layer=15, token=last_real_token] in R^896
```

The branch classifier is:

```text
z = MLP_SAPLMA(h)
p_SAPLMA = sigmoid(z)
```

with architecture:

```text
896 -> 256 -> 128 -> 64 -> 1
```

Training uses binary cross-entropy with logits and regularization:

```text
L = BCEWithLogits(z, y) + lambda_1 * ||theta||_1
```

The optimizer also applies L2 weight decay:

```text
weight_decay = 1e-4
lambda_1 = 1e-5
dropout = 0.3
epochs = 5
```

This branch captures a static single-vector signal: whether the final representation of the answer looks more like the hidden states of truthful or hallucinated answers.

## Method 2: ICR Probe Branch

The ICR branch is based on the idea that hallucination can appear in how representations change across transformer layers, not only in the final hidden state. The implementation uses a response-token, layer-wise consistency score between attention and residual-stream updates.

For layer `l` and response token `t`:

```text
Delta_l,t = h_l,t - h_(l-1),t
A_l,t = mean_heads(attention_l,t)
```

Only attention to the user span and response span is kept. From that masked attention row, the top `k = 10` source tokens are selected:

```text
K_t = top_k(A_l,t)
```

For every selected source token `j`, the previous-layer representation is projected onto the residual update:

```text
s_j = <h_(l-1),j, Delta_l,t> / ||h_(l-1),j||
```

The code standardizes both the projection scores and attention scores, converts them into distributions, and computes Jensen-Shannon divergence:

```text
p = softmax(standardize(s_K))
q = softmax(standardize(A_K))
m = 0.5 * (p + q)
JS(p, q) = 0.5 * sum_i p_i log(p_i / m_i) + 0.5 * sum_i q_i log(q_i / m_i)
```

The final layer feature is the mean over response tokens:

```text
ICR_l = mean_t JS(p_l,t, q_l,t)
ICR = [ICR_1, ..., ICR_24] in R^24
```

The classifier is an MLP:

```text
24 -> 128 -> 64 -> 32 -> 1
```

with batch normalization, LeakyReLU, dropout 0.3, Adam, 25 epochs, and L2 weight decay `1e-4`.

This branch captures whether the tokens attended to by the model are consistent with the representational direction in which the response token is updated.

## Method 3: LLM-Check Branch

The LLM-Check branch follows Sriramanan et al.'s response-level geometry idea. Instead of using one hidden vector, it analyzes the matrix of response-token hidden states and combines it with logit uncertainty features.

The logit part has three features:

```text
log p(x_t | x_<t) = log_softmax(logits_(t-1))[x_t]
perplexity = exp(-mean_t log p(x_t | x_<t))
H_token,t = -sum_v p_t(v) log p_t(v)
window_entropy = max_t H_token,t
H_topk = -sum_i softmax(topk(logits_t))_i log softmax(topk(logits_t))_i
```

The hidden-state geometry part is computed for every transformer layer. For response-token hidden states:

```text
H_l = [h_l,response_start, ..., h_l,response_end] in R^(m x 896)
H_centered = H_l - mean_hidden_dim(H_l)
Sigma_l = H_centered H_centered^T + alpha I
hidden_score_l = mean_i log(svdvals(Sigma_l)_i)
```

The implementation uses:

```text
alpha = 1e-3
24 hidden scores
3 logit features
27 total LLM-Check features
```

The classifier is logistic regression:

```text
StandardScaler -> LogisticRegression(penalty="l1", solver="saga", C=10)
```

This branch contributed most to the final metric. My interpretation is that it captures response-level geometry very well: hallucinated answers often have a different hidden-state spectrum and uncertainty profile than supported answers.

## Fusion Strategy

The final approach is standard stacking. The three branch models are trained as separate heads first:

```text
p_ICR = P_ICR(y = 1 | x)
p_LLM_check = P_LLM_check(y = 1 | x)
p_SAPLMA = P_SAPLMA(y = 1 | x)
```

The meta-model is a four-parameter L2-regularized logistic regression:

```text
P(y = 1 | x) = sigmoid(w0 + w1*p_ICR + w2*p_LLM_check + w3*p_SAPLMA)
```

The important detail is that the meta-learner is trained only on out-of-fold predictions. Each base head predicts samples that were not used to train that head. This avoids the common stacking failure mode where the meta-model learns overconfident in-sample branch outputs.

```text
FOLD 1           FOLD 2     ...    FOLD 5
                 ┌──────────┐    ┌──────────┐       ┌──────────┐
Train set:       │ 2,3,4,5  │    │ 1,3,4,5  │       │ 1,2,3,4  │
                 └──────────┘    └──────────┘       └──────────┘
                      │               │                   │
                   fit heads       fit heads           fit heads
                      │               │                   │
Val set (unseen): │  fold 1  │    │  fold 2  │       │  fold 5  │
                      │               │                   │
                   predict OOF    predict OOF         predict OOF
                      │               │                   │
                      └───────────────┴──── ... ──────────┘
                                      │
                            oof_ICR              (689,)  all out-of-sample
                            oof_Orgad/LLM_check  (689,)  all out-of-sample
                            oof_SAPLMA           (689,)  all out-of-sample
                                      │
                            X_meta = stack -> (689, 3)
                                      │
                            LogisticRegression(C=0.01)
                                      │
                               meta_lr (4 params)
                                      │
                    ┌─────────────────┘
                    │
             Retrain heads on ALL 689 samples
                    │
             At test time: heads -> (n_test, 3) -> meta_lr -> predictions
```

The base heads are then retrained on all 689 labelled samples. At test time, the final heads produce a `(n_test, 3)` probability matrix, and the meta-model produces the final labels.

## Final Solution Description

What components did I modify? I modified the feature extraction logic in `aggregation.py`, the classifier and stacking logic in `probe.py`, and the split strategy in `splitting.py`. I also added `Run_Colab.ipynb` and this report for reproducibility.

What is the final approach? The final approach is a three-branch ensemble over Qwen hidden-state-derived features. SAPLMA captures a single-layer last-token signal. ICR captures cross-layer attention/update consistency. LLM-Check captures response-level hidden-state geometry and logit uncertainty. The three branch probabilities are fused by an L2-regularized logistic regression stacker trained on out-of-fold predictions.

Why these choices? I used these methods because they enrich the feature space with different views of the same model internals. A single hidden vector is useful but limited. Cross-layer dynamics add information about how the model processes the response. Response-level geometry adds information about the structure of all generated tokens. Combining them reduces dependence on one fragile signal.

What contributed most to the metric? LLM-Check contributed the most as an individual branch, reaching about `0.76` test accuracy. It likely helped because its hidden-state geometry features capture information that is not available from a single token. Regularization also mattered: dropout, L1/L2 penalties, standardization, and the strongly regularized meta-learner all reduced overfitting on the small 689-sample dataset.

## Experiments and Failed Attempts

I tried Geometry of Truth as a direct route to a classifier, but it did not work well enough as a final method. The PCA plots did not show clean separation between truthful and hallucinated examples. The likely reasons are:

```text
the dataset is small and noisy
answers have very different lengths and writing styles
the model is Qwen2.5-0.5B, much smaller than the models in the original paper
the labels judge answer correctness relative to context, not only factual truth in isolation
many samples contain instruction-following failures mixed with factual errors
```

I discarded Geometry of Truth as a final detector, but it was still useful diagnostically. It helped identify the last-token representation and layer 15 as a good SAPLMA configuration.

I also evaluated the branches independently before using stacking:

```text
ICR Probe:   about 0.72 test accuracy
SAPLMA:      about 0.74 test accuracy
LLM-Check:   about 0.76 test accuracy
Fusion:      about 0.87 test accuracy
```

The improvement from fusion suggests that the branches are not redundant. Each one captures a different aspect of the hidden states.

## How results.json Is Produced

After feature extraction on `data/dataset.csv`, `solution.py` calls:

```python
splits = split_data(y, df)
fold_results = run_evaluation(splits, X, y, HallucinationProbe)
save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)
```

`run_evaluation` trains a fresh `HallucinationProbe` for each split, reports baseline metrics and probe metrics, and returns a list of per-fold dictionaries. `save_results` writes those fold metrics plus averaged metrics to `results.json`.

The reported test metrics in `results.json` refer to held-out folds created from the labelled training dataset. They are not computed from `data/test.csv`, because `data/test.csv` has no labels.

## How predictions.csv Is Produced

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

## Final Accuracy Diagram

![Final accuracy comparison](final_accuracy_comparison.png)

## Diagnostic Plots

The following plots came from the Geometry of Truth diagnostic experiments. They were not used directly as the final detector, but they informed the layer and token choices.

![Failed PCA separation](failed_separation.png)

![Token and hidden-state choice](token_hidden_states_choise.png)

## Notes on Colab

This solution asks Qwen for hidden states, logits, and attentions. That is more memory intensive than ordinary text generation. For this reason `solution.py` uses `BATCH_SIZE = 1`. It is slower, but it avoids the common Colab failure mode where the runtime crashes during attention extraction.

The notebook `Run_Colab.ipynb` clones or updates the repository, installs dependencies, runs `solution.py`, and displays the first rows of `predictions.csv`.
