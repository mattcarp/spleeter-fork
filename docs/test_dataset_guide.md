# Audio Separation Test Dataset Guide

This guide explains how to use the comprehensive test dataset with our evaluation framework to benchmark and compare audio source separation algorithms.

## Test Dataset Overview

Our test dataset is designed to provide a standardized, diverse, and challenging set of audio files for evaluating source separation algorithms. It includes:

- **Genre-based test cases**: Various musical styles to ensure algorithms work across different sound characteristics
- **Challenge-specific test cases**: Files designed to stress-test specific separation challenges
- **Reference sources**: Each sample includes isolated source tracks for accurate evaluation

## Dataset Structure

```
data/test_dataset/
├── metadata.json          # Complete metadata for all tracks
├── README.md              # Dataset documentation
├── genres/
│   ├── pop/               # Pop music examples
│   ├── rock/              # Rock music examples
│   ├── jazz/              # Jazz music examples
│   ├── classical/         # Classical music examples
│   ├── electronic/        # Electronic music examples
│   └── other/             # Other genres
└── challenges/
    ├── reverberation/     # High reverb examples
    ├── distortion/        # Contains distorted instruments/vocals
    ├── low_volume/        # Sources with significant volume differences
    ├── similar_timbres/   # Contains sources with similar timbres
    └── dynamic_range/     # Examples with wide dynamic range
```

## Populating the Dataset

### Using the Dataset Curation Script

We provide a utility script to help populate the dataset:

```bash
# Install required dependencies
pip install requests

# Add samples to the pop genre from Freesound
python data/test_dataset/fetch_samples.py --target pop --source freesound --query "pop music multitrack" --samples 3

# Add MUSDB18 samples to the rock genre
python data/test_dataset/fetch_samples.py --target rock --source musdb
```

### Adding Your Own Samples

You can also add your own samples manually:

1. Place mixture and source files in the appropriate directory
2. Add metadata to `metadata.json` (you can use the script to update it)

## Using the Dataset for Evaluation

### Basic Evaluation

To evaluate a separation model using the test dataset:

```bash
# Evaluate a single model on a specific genre
python -m spleeter.evaluation.evaluate \
    --models 2stems \
    --audio data/test_dataset/genres/pop/mixture.wav \
    --reference-dir data/test_dataset/genres/pop/references/ \
    --output-dir evaluation_results/pop
```

### Batch Evaluation

To evaluate across multiple test cases:

```bash
# Create a simple script to process all test cases
for genre in pop rock jazz classical electronic; do
    for file in data/test_dataset/genres/$genre/*.wav; do
        if [[ $file == *"mixture"* ]]; then
            ref_dir=$(dirname "$file")/references
            python -m spleeter.evaluation.evaluate \
                --models 2stems 4stems \
                --audio "$file" \
                --reference-dir "$ref_dir" \
                --output-dir "evaluation_results/$genre/$(basename "$file" .wav)"
        fi
    done
done
```

## Analyzing Results

The evaluation framework generates comprehensive reports that can be found in the output directory:

- **metrics.json**: Raw metrics for each model and source
- **Visualization PNG files**: Visual comparisons of separation performance
- **summary_report.txt**: Textual summary of evaluation results

### Key Metrics to Consider

When comparing models, pay attention to:

1. **SDR (Signal-to-Distortion Ratio)**: Higher is better, measures overall separation quality
2. **SIR (Signal-to-Interference Ratio)**: Higher is better, measures isolation from other sources
3. **SAR (Signal-to-Artifacts Ratio)**: Higher is better, measures freedom from artifacts
4. **SI-SDR (Scale-Invariant SDR)**: Higher is better, scale-invariant measure of separation

### Frequency-Dependent Analysis

Our framework provides frequency-dependent analysis to identify where models struggle:

- **Low frequencies (20-200Hz)**: Often challenging for bass/kick drum separation
- **Mid frequencies (200-2000Hz)**: Critical for vocal intelligibility
- **High frequencies (2000-20000Hz)**: Important for percussion and transients

## Expanding the Dataset

The test dataset is designed to be expandable. To contribute:

1. Ensure you have the legal right to share the audio
2. Follow the directory structure
3. Add complete metadata
4. Use the provided script to maintain consistency

## Best Practices for Reproducible Evaluation

1. **Use the same test set** for all compared models
2. **Report all metrics**, not just the most favorable ones
3. **Analyze per-category performance** to identify strengths and weaknesses
4. **Use multiple sources** of test data to ensure robustness

## License and Attribution

When using the test dataset, always respect the license of each individual audio file as specified in the metadata. Many files are available under Creative Commons licenses, which require attribution to the original creators. 