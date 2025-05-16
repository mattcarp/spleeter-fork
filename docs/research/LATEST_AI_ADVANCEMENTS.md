# Latest AI Advancements in Audio Source Separation

This document summarizes recent advancements in AI-based audio source separation techniques that could be integrated into our Spleeter fork to improve separation quality.

## Self-Supervised Learning Models

### HuBERT and Its Variants

[HuBERT (Hidden-Unit BERT)](https://arxiv.org/abs/2106.07447) has emerged as a powerful self-supervised learning approach for speech representation. It uses the masked prediction of hidden units, similar to BERT in NLP, and has been shown to learn useful representations for various audio tasks, including source separation.

Recent variants that could be beneficial:

- **DistilHuBERT**: A smaller, distilled version of HuBERT that maintains performance with reduced computational requirements.
- **WavLM**: An extension of HuBERT that incorporates additional denoising objectives during pre-training, potentially useful for separating sources in noisy environments.
- **MR-HuBERT (Multi-Resolution HuBERT)**: Processes audio at multiple time scales, which could better capture both short-term and long-term dependencies in music.

### Other Self-Supervised Approaches

- **Data2vec**: A self-supervised framework that can be applied to speech, vision, and language, offering potential for multi-modal enhancements.
- **PASE+**: A Problem-Agnostic Speech Encoder that learns speech representations through a combination of self-supervised tasks.
- **TERA**: Self-Supervised learning of Transformer Encoder Representation for speech, which could be adapted for music source separation.

## Transformer-Based Architectures

### Zipformer

[Zipformer](https://arxiv.org/abs/2411.17100) is a faster and more memory-efficient Transformer variant specifically designed for speech applications. Key advantages:

- Uses a U-Net-like structure for efficient learning of temporal representations at various resolutions
- Processes sequences at different frame rates (50Hz, 25Hz, 12.5Hz, 6.25Hz)
- Includes BiasNorm for better retention of sequence length
- Incorporates activation functions like SwooshR and SwooshL that outperform Swish
- Can be paired with ScaledAdam optimizer for improved convergence

### Masked Audio Transformer (MAT)

[MAT models](https://arxiv.org/abs/2408.08673) use masked-reconstruction pre-training to enhance performance. The approach:

- Pre-trains a Transformer with relative positional encoding via masked-reconstruction tasks
- Utilizes a global-local feature fusion strategy to enhance localization capability
- Can be fine-tuned for specific audio tasks including source separation

## Source Separation Specific Approaches

### Semantic Grouping Network (SGN)

[SGN](https://arxiv.org/abs/2407.03736) is a novel approach that directly disentangles sound representations and extracts high-level semantic information for each source from the input audio mixture:

- Aggregates category-wise source features through learnable class tokens of sounds
- Uses aggregated semantic features as guidance to separate corresponding audio sources
- Outperforms previous audio-only methods without requiring visual cues

### Audio Masked Autoencoder (A-MAE)

[A-MAE](https://arxiv.org/abs/2407.11745) adapts the masked autoencoder paradigm to audio:

- Pre-trains on unlabeled data to obtain task-agnostic representations
- Can be frozen or fine-tuned during adaptation to source separation tasks
- Representations can be concatenated with STFT features as input to separation models

## Evaluation Approaches

### Objective Metrics

Modern separation evaluation typically uses:

- **SDR (Source-to-Distortion Ratio)**: Overall measurement of separation quality
- **SIR (Source-to-Interference Ratio)**: Measurement of how well sources are isolated from each other
- **SAR (Source-to-Artifact Ratio)**: Measurement of artifacts introduced by separation
- **SI-SDR (Scale-Invariant SDR)**: A more robust version of SDR that is invariant to scaling

Fast implementations like [fast_bss_eval](https://fast-bss-eval.readthedocs.io/) can efficiently compute these metrics.

### Subjective Evaluation

For true quality assessment, subjective listening tests are essential:

- **MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor)**: The gold standard for audio quality evaluation
- **Web-based crowd-sourced alternatives**: More accessible but still provide valuable human feedback

## Hardware Acceleration and Optimization

- **GPU-Accelerated Training**: Modern architectures are designed for efficient GPU training
- **Mixed Precision Training**: Using FP16/BF16 for faster computation with minimal accuracy loss
- **Pruning and Quantization**: Post-training optimization for faster inference
- **Efficient Data Loading**: Techniques like dynamic bucketing improve training efficiency

## Next Steps for Our Project

1. Implement a self-supervised pre-training pipeline using HuBERT or similar models
2. Develop a Transformer-based architecture optimized for source separation
3. Create a comprehensive evaluation framework that includes both objective metrics and subjective testing
4. Consider implementing semantic grouping for improved separation of similar-sounding sources

## References

- HuBERT: https://arxiv.org/abs/2106.07447
- k2SSL/Zipformer: https://arxiv.org/abs/2411.17100
- MAT-SED: https://arxiv.org/abs/2408.08673
- Semantic Grouping Network: https://arxiv.org/abs/2407.03736
- Universal Sound Separation with A-MAE: https://arxiv.org/abs/2407.11745 