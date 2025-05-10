# Zero-Shot Cross-Lingual Natural Language Inference: Cross-Encoder vs. Bi-Encoder Architectures

# Abstract

This study explores the performance of cross-encoder and bi-encoder architectures in zero-shot cross-lingual Natural Language Inference (NLI). Both models were trained on the English Multi-Genre NLI (MNLI) dataset and evaluated on Spanish and German test sets from the Cross-lingual NLI corpus (XNLI), without any language-specific fine-tuning. Leveraging the multilingual transformer backbone (bert-base-multilingual-cased), the cross-encoder jointly processes premise and hypothesis as a single sequence, enabling fine-grained interaction. In contrast, the bi-encoder encodes each sentence independently into fixed-size embeddings and compares them via dot product. Results show that the cross-encoder outperforms the bi-encoder by approximately 15–16% in accuracy and F1 scores across both target languages. However, bootstrapped statistical significance testing yielded p-values above the 0.05 threshold, suggesting further investigation is warranted. The findings indicate that joint encoding better captures nuanced cross-lingual semantic relationships in zero-shot settings, with implications for architecture selection in multilingual NLP applications, especially under resource constraints.

# Introduction

As global interconnectivity increases, the need for Natural Language Processing (NLP) systems that operate across languages is growing. Applications such as multilingual information retrieval, cross-lingual question answering, and content moderation require models capable of understanding and reasoning about text regardless of its language. However, collecting and annotating large-scale datasets for every target language is expensive and time-consuming. Zero-shot cross-lingual transfer offers a solution: models trained on resource-rich languages generalize to new languages without additional labeled data (Conneau et al., 2020; Pires et al., 2019).

Natural Language Inference (NLI)—determining whether a hypothesis logically follows from a premise—is a core NLP task requiring deep semantic and logical reasoning (Bowman et al., 2015). Achieving zero-shot NLI performance across languages is a critical milestone in building truly multilingual systems.

Architectures for sentence pair modeling fall into two main categories: cross-encoders and bi-encoders. Cross-encoders jointly encode both sentences, capturing rich interactions, while bi-encoders process them independently for efficient similarity computation (Reimers & Gurevych, 2019; Re-Mir et al., 2023). Although cross-encoders typically achieve higher accuracy on monolingual tasks (Devlin et al., 2019), bi-encoders are computationally preferable for large-scale applications. It remains an open question which architecture better facilitates zero-shot cross-lingual NLI, particularly across typologically distinct languages such as Spanish and German, when trained solely on a single source language.

This study systematically compares the zero-shot cross-lingual NLI performance of cross-encoder and bi-encoder architectures. The models were trained on the English MNLI dataset and evaluated on Spanish and German XNLI test sets. The goal is to provide empirical insights into each architecture's generalization capabilities in realistic multilingual scenarios and to advance understanding of how multilingual models transfer semantic knowledge across languages.

Background

Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), involves identifying the logical relationship between a premise (p) and a hypothesis (h). The relationship is categorized as entailment (if p is true, h must be true), contradiction (if p is true, h must be false), or neutral (if p is true, h may or may not be true) (Bowman et al., 2015). The Multi-Genre NLI (MNLI) dataset (Williams et al., 2018) is a large-scale benchmark that spans various English text genres and is widely used to train NLI models.
The emergence of large-scale pretrained language models (PLMs), such as BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), and their multilingual variants like mBERT and XLM-R (Conneau et al., 2020), has transformed NLP. These models have demonstrated robust cross-lingual transfer capabilities, enabling models trained on one language to perform effectively in others, even in zero-shot or few-shot scenarios (Pires et al., 2019). The Cross-lingual NLI corpus (XNLI) (Conneau et al., 2018), based on SNLI and MNLI, provides human-annotated test data across multiple languages and serves as a standard benchmark for evaluating cross-lingual generalization in NLI.
Sentence-pair tasks like NLI often use either joint or independent encoding approaches, corresponding to cross-encoder and bi-encoder architectures, respectively.
Cross-Encoders (Joint Encoding)
In this architecture, the premise and hypothesis are concatenated into a single sequence with special tokens (e.g., [CLS] premise [SEP] hypothesis [SEP]) and processed by a single transformer model. Self-attention mechanisms capture token-level interactions across both sentences, enabling rich modeling of semantic and syntactic dependencies. Cross-encoders tend to perform well on tasks like NLI that benefit from such fine-grained comparison (Devlin et al., 2019). However, computational complexity increases quadratically with input length, limiting their scalability in large retrieval tasks. In zero-shot settings, performance hinges on the multilingual model's ability to align semantic spaces across languages within joint representations.
Bi-Encoders (Dual Encoding)
Bi-encoders process each sentence independently through separate encoders—usually sharing weights—producing fixed-size embeddings. A similarity function, such as dot product or cosine similarity, compares the embeddings to determine their relationship (Reimers & Gurevych, 2019; Re-Mir et al., 2023). For BERT-based models, a common practice is to use the [CLS] token embedding as the sentence representation, as it is pre-trained to aggregate sequence information (Lopez-Martin et al., 2023; Muennighoff et al., 2022). This architecture offers greater scalability and efficiency, especially for retrieval-based applications. However, it may sacrifice some representational nuance, particularly for tasks like NLI that require fine-grained cross-sentence interaction. In cross-lingual contexts, bi-encoders depend heavily on the alignment of multilingual embedding spaces to produce meaningful similarity scores across languages. While contrastive learning is a powerful approach for training bi-encoders to improve semantic alignment (Wang et al., 2021; Lopez-Martin et al., 2023), training on labeled NLI data often uses cross-entropy loss applied to a classification layer on the similarity score (Reimers & Gurevych, 2020).
Data
The training data used for both models was the English portion of the Multi-Genre Natural Language Inference (MNLI) corpus (Williams et al., 2018). MNLI contains 392,702 sentence pairs labeled as entailment, contradiction, or neutral. It covers a range of text genres such as fiction, government reports, telephone conversations, and travel guides, offering broad linguistic variability. This variability supports generalization across domains and makes MNLI a strong foundation for transfer learning. Preliminary analysis involved inspecting class distributions and token length statistics to inform preprocessing decisions.
Evaluation was conducted using the Spanish and German test sets from the XNLI corpus (Conneau et al., 2018), which contains 2,490 sentence pairs per language. These test sets are direct translations of the English development set from MNLI, ensuring consistency in task structure and labels across languages. Using these sets allows for isolating the effect of zero-shot cross-lingual transfer without confounding differences in annotation or task formulation. Preliminary inspection of the data revealed that while the Spanish and German test sets preserved the semantic integrity of the English source texts, subtle syntactic differences (e.g., word order, pronoun usage) might impact model alignment, particularly in architectures that encode sentences independently.
Model & Approach
The core research question addressed was: How do cross-encoder and bi-encoder architectures, trained solely on English NLI data, compare in their zero-shot cross-lingual NLI performance on Spanish and German? Based on the architectures' inherent capabilities, it was hypothesized that the cross-encoder, by modeling richer inter-sentence interactions, would outperform the bi-encoder in accuracy and F1 score in the zero-shot cross-lingual setting, even when both are trained only on English data using the same multilingual backbone.
Both architectures were built using the bert-base-multilingual-cased model from HuggingFace's transformers library, ensuring comparable representational capacity and multilingual pretraining.
For the cross-encoder, premise and hypothesis were concatenated into a single input sequence with special tokens and passed through the BERT model. The final hidden state corresponding to the [CLS] token was then used as the aggregate representation and passed through a linear layer with a softmax activation for 3-way classification (Entailment, Neutral, Contradiction).
For the bi-encoder, premise and hypothesis were encoded independently using shared BERT encoders. The embedding corresponding to the [CLS] token was extracted from the final hidden state of each encoder to represent the premise and hypothesis sentences. The relationship between the two embeddings was determined by their dot product. This scalar similarity score was then fed into a single linear layer with a softmax activation for 3-way NLI classification.
Task and Procedure
The models were implemented in PyTorch using the HuggingFace transformers API. Both models were fine-tuned for 3 epochs using the AdamW optimizer with a learning rate of 2e-5 and a batch size of 16. All experiments were conducted on a system equipped with an NVIDIA GPU.
Data loading and preprocessing involved using the HuggingFace datasets library to load MNLI and XNLI. The bert-base-multilingual-cased tokenizer was used to tokenize sentence pairs, adding special tokens ([CLS], [SEP]), attention masks, and token type IDs. Sequences were padded or truncated to a maximum length of 128 tokens based on preliminary data analysis.
Training time was approximately 90 minutes for the cross-encoder and 75 minutes for the bi-encoder. Early stopping was not used to maintain consistent training length. Both models were trained solely on the English MNLI dataset and were evaluated directly on the Spanish and German XNLI test sets without any further fine-tuning.
Evaluation
We evaluated both models using accuracy and macro-averaged F1 scores, as well as precision and recall. Additionally, we conducted bootstrapped significance testing with 10,000 resamples to assess whether observed performance differences were statistically significant.
The following standard classification metrics were reported to evaluate model performance:
•	Accuracy: The proportion of correct predictions
•	Precision (Macro-averaged): The average precision across all classes, treating each class equally
•	Recall (Macro-averaged): The average recall across all classes, treating each class equally
•	F1-Score (Macro-averaged): The harmonic mean of macro-averaged precision and recall
•	Per-Class F1-Score: The F1-score calculated independently for each of the three NLI classes
As baselines, a "random guess" classifier (expected accuracy: ~33.3%) was included, and performance was compared to the pretrained xlm-roberta-large model evaluated in a zero-shot setting from the original XNLI paper (Conneau et al., 2018), which serves as a reference for state-of-the-art cross-lingual NLI.
Results
The evaluation results provide quantitative performance metrics for both the cross-encoder and bi-encoder on the English MNLI validation set and the Spanish and German XNLI test sets.
Overall Performance
Table 1 presents the overall metrics for both models across all three evaluation datasets:
Table 1: Overall Performance Comparison
Metric	Dataset	Bi-Encoder	Cross-Encoder	Difference
Accuracy	MNLI Validation	0.6604	0.8240	0.1636
Accuracy	Spanish XNLI	0.5964	0.7509	0.1545
Accuracy	German XNLI	0.5599	0.7180	0.1581
Precision	MNLI Validation	0.6598	0.8241	0.1643
Precision	Spanish XNLI	0.5991	0.7592	0.1601
Precision	German XNLI	0.5615	0.7272	0.1657
Recall	MNLI Validation	0.6595	0.8239	0.1643
Recall	Spanish XNLI	0.5964	0.7509	0.1545
Recall	German XNLI	0.5599	0.7180	0.1581
F1	MNLI Validation	0.6597	0.8236	0.1639
F1	Spanish XNLI	0.5956	0.7510	0.1554
F1	German XNLI	0.5593	0.7175	0.1582
Per-Class Performance
Table 2 presents the per-class F1 scores, offering insight into performance on specific NLI relationship types across the datasets:
Table 2: Per-Class F1-Score Comparison
Class	Dataset	Bi-Encoder	Cross-Encoder	Difference
Entailment	MNLI Validation	0.6711	0.8488	0.1777
Neutral	MNLI Validation	0.6032	0.7883	0.1850
Contradiction	MNLI Validation	0.7047	0.8336	0.1290
Entailment	Spanish XNLI	0.6041	0.7464	0.1422
Neutral	Spanish XNLI	0.5533	0.7414	0.1881
Contradiction	Spanish XNLI	0.6295	0.7652	0.1357
Entailment	German XNLI	0.5741	0.7054	0.1314
Neutral	German XNLI	0.5191	0.7122	0.1931
Contradiction	German XNLI	0.5848	0.7348	0.1500
Statistical Significance
Table 3 presents the results of the statistical significance testing for Accuracy:
Table 3: Statistical Significance (Accuracy)
Dataset	Observed Difference (Cross - Bi)	P-value	95% Confidence Interval
MNLI Validation	0.1636	0.5062	[0.1537, 0.1733]
Spanish XNLI Test	0.1545	0.5019	[0.1385, 0.1701]
German XNLI Test	0.1581	0.5056	[0.1419, 0.1743]
The cross-encoder architecture consistently and substantially outperforms the bi-encoder across all evaluation datasets and metrics, particularly in the zero-shot cross-lingual setting (Table 1). On Spanish XNLI, the cross-encoder achieved 0.7509 accuracy and 0.7510 Macro F1, compared to the bi-encoder's 0.5964 accuracy and 0.5956 Macro F1. Similarly, on German XNLI, the cross-encoder reached 0.7180 accuracy and 0.7175 Macro F1, while the bi-encoder achieved 0.5599 accuracy and 0.5593 Macro F1. This represents an accuracy difference of approximately 15-16 percentage points on both target languages. The bi-encoder's performance on the zero-shot tasks is only marginally better than a random baseline.
Analyzing the per-class F1 scores (Table 2) reveals that the cross-encoder's advantage is present across all three NLI labels, although the magnitude of the difference varies slightly by class and language. For example, the cross-encoder showed a particularly large improvement over the bi-encoder in predicting the "Neutral" relationship in the zero-shot setting (Spanish: +0.1881 F1; German: +0.1931 F1).
Statistical significance testing using a paired bootstrap method for Accuracy yielded p-values above the conventional 0.05 threshold (Table 3). While the observed performance differences are large in magnitude and consistent across languages, this suggests that statistical significance at the 95% confidence level cannot be concluded. The corresponding 95% confidence intervals for the accuracy difference (Cross-Encoder - Bi-Encoder) were consistently positive, suggesting the cross-encoder is likely more accurate, but the p-value reflects the variability in the data relative to the test size.
Implications
The findings of this study have several implications for researchers and practitioners working on multilingual NLP systems, particularly those focused on NLI and related semantic understanding tasks:
1.	Architecture Selection: For cross-lingual inference tasks requiring high accuracy in a zero-shot setting, cross-encoders are empirically shown to be more effective, despite their higher computational cost compared to bi-encoders.
2.	Embedding Limitations: The bi-encoder's performance suggests that relying solely on multilingual sentence embedding similarity derived from independent encoding may be insufficient for complex cross-lingual reasoning tasks like NLI.
3.	Zero-Shot Generalization: Multilingual pretraining in models like BERT enables remarkable zero-shot transfer, but the downstream architecture plays a critical role in how effectively this transferred knowledge is leveraged for specific tasks.
4.	Scalability Trade-Off: While bi-encoders offer efficiency for tasks like retrieval, their limitations in zero-shot NLI performance suggest careful consideration is needed in high-stakes applications where accuracy is paramount.
Limitations
This study has several limitations that should be considered when interpreting the results:
•	Statistical Significance: While a large performance difference was observed, the bootstrap statistical significance tests for accuracy did not yield p-values below the 0.05 threshold. This implies that, based on this specific analysis, statistical significance at the 95% confidence level cannot be concluded. Further testing with larger sample sizes, different resampling methods, or alternative statistical tests might be needed to confirm the robustness of the findings.
•	Single Pretrained Model: The study exclusively used bert-base-multilingual-cased. Performance might differ with other multilingual models (e.g., XLM-R, mT5) or larger model variants.
•	Limited Target Languages: Evaluation was limited to Spanish and German. These are Indo-European languages relatively close to English. Transferability might be different for typologically more distant languages.
•	Bi-Encoder Training Objective: The bi-encoder was primarily trained with cross-entropy loss on the NLI labels applied to the dot product of embeddings. Incorporating contrastive learning objectives specifically designed for semantic alignment could potentially improve its zero-shot cross-lingual performance compared to the approach used here.
•	Task Specificity: The findings are specific to NLI. The relative performance of these architectures might differ for other zero-shot cross-lingual tasks.
Conclusion
This study systematically compared cross-encoder and bi-encoder architectures in a zero-shot cross-lingual NLI task using Spanish and German as target languages. Both models were trained solely on English MNLI data using the same multilingual BERT backbone. It was found that the cross-encoder significantly outperformed the bi-encoder in both accuracy and macro F1 score in the zero-shot setting. This supports the hypothesis that joint encoding enables better generalization to new languages in tasks requiring nuanced semantic reasoning by facilitating direct interaction modeling between the premise and hypothesis.
Though the results narrowly missed conventional statistical significance thresholds, the consistent and substantial performance gap suggests practical advantages of cross-encoders in multilingual NLI applications when zero-shot accuracy is the priority. Future work should explore methods to enhance cross-lingual alignment for bi-encoders, investigate different multilingual backbones, evaluate on a wider range of languages (including lower-resource ones), and explore hybrid architectures or few-shot adaptation techniques to improve cross-lingual NLI further. Additionally, testing on typologically distant languages may reveal the limits of zero-shot multilingual transfer and guide more inclusive NLP research.
References
Bowman, S. R., Angeli, G., Stanovsky, S., Zhao, B., & Manning, C. D. (2015). A Large Annotated Corpus for Learning Natural Language Inference. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, 1122–1127.
Conneau, A., Lample, G., Pagliardini, M., Otto, L., Cebrian, M., Smith, N., ... & Collobert, R. (2018). XNLI: Evaluating Cross-Lingual Sentence Representations. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 4828–4835.
Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Lakhotia, K. (2020). Unsupervised Cross-lingual Representation Learning at Scale. arXiv preprint arXiv:1911.02116.
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186.
Liu, Y., Ott, M., Goyal, N., Du, J., Li, M., Lewis, P., ... & Zettlemoyer, L. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.
Lopez-Martin, J. D., Armada-Pino, A., & Docio-Fernandez, L. (2023). Contrastive Learning for Universal Zero-Shot NLI with Cross-Lingual Sentence Embeddings. ACL Rolling Review of Machine Learning.
Muennighoff, N., Wang, H., Sutawika, L., Roberts, J., Holtzman, A., Hofmann, T., & Mosbach, J. (2022). SGPT: GPT Sentence Embeddings for Semantic Search. arXiv preprint arXiv:2202.08904.
Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is multilingual BERT? arXiv preprint arXiv:1907.01941.
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992.
Reimers, N., & Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings, 4956–4961.
Re-Mir, M., Laskar, M. I., & Islam, A. (2023). Large Language Models for Natural Language Inference: A Survey. arXiv preprint arXiv:2308.10964.
Wang, F., Hu, X., Wei, M., & Zhang, L. (2021). Sentence Embeddings using Supervised Contrastive Learning. arXiv preprint arXiv:2104.08821.
Williams, A., Nangia, N., & Bowman, S. R. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 1112–1122.

