# Spleeter Evaluation Test Dataset

This dataset is designed for comprehensive evaluation of audio source separation algorithms. It includes a diverse collection of multi-track music spanning different genres, production styles, and mixing techniques to enable thorough testing of separation quality.

## Dataset Structure

```
test_dataset/
├── metadata.json          # Complete metadata for all tracks
├── README.md              # This file
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

## Test Case Categories

Each test case is categorized to ensure thorough evaluation across diverse scenarios:

1. **Genre-based**: Tests separation quality across musical styles
2. **Challenge-based**: Tests robustness against specific audio challenges
3. **Production Style**: Tests separation across different production/mixing techniques

## Audio Properties

- **Format**: WAV (lossless)
- **Sample Rate**: 44.1kHz
- **Bit Depth**: 16-bit
- **Channels**: Stereo
- **Duration**: 10-30 seconds per track

## Sources

This dataset includes:

1. **Creative Commons** licensed multi-track recordings
2. **Public domain** classical recordings with isolated sources
3. **Synthesized** test tracks with perfectly isolated sources
4. **Dataset excerpts** from publicly available research datasets (following applicable licenses):
   - [MUSDB18](https://sigsep.github.io/datasets/musdb.html) (sample excerpts)
   - [MedleyDB](https://medleydb.weebly.com/) (sample excerpts)
   - [BACH10](https://labsites.rochester.edu/air/resource.html) (sample excerpts)

## Dataset Population Strategy

This dataset will be populated gradually using the following sources:

1. **Creative Commons audio**: From platforms like Freesound, ccMixter
2. **Open-source STEMS**: From remix competitions and platforms
3. **Generated content**: Using synthesizers and virtual instruments
4. **Academic sources**: Small excerpts from academic datasets (with proper attribution)

## Usage

To use this dataset for evaluation:

```bash
python -m spleeter.evaluation.evaluate \
    --models 2stems 4stems \
    --audio path/to/mixture.wav \
    --reference-dir path/to/reference_sources/ \
    --output-dir evaluation_results
```

## License and Attribution

Each audio sample in this dataset is accompanied by proper licensing information in the metadata.json file. When using this dataset, please ensure proper attribution according to the license of each sample.

## Contribution

To contribute to this test dataset, please ensure:

1. You have the legal right to distribute the audio
2. You provide complete metadata
3. The audio meets the minimum technical specifications
4. You categorize the contribution appropriately 