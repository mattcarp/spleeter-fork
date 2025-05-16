# Guide for Selecting Audio Source Separation Test Samples

This document provides guidance on selecting effective test samples for evaluating audio source separation algorithms.

## Characteristics of Ideal Test Samples

### 1. Distinct Sources with Known Ground Truth

The most valuable test samples have:

- **Perfectly isolated stems/tracks**: Each source (vocals, drums, bass, etc.) available separately
- **Clean recording quality**: Minimal noise or artifacts in the original stems
- **Complete multitracks**: All components that make up the mixture
- **No pre-applied effects**: Stems without heavy processing that could complicate separation

### 2. Diversity in Musical Characteristics

Include samples with:

- **Various instrumentation**: Different instrument combinations and timbres
- **Multiple genres**: Different musical styles have unique separation challenges
- **Different vocal styles**: Male/female, soft/powerful, solo/harmony
- **Production variety**: Various recording and mixing techniques

### 3. Specific Separation Challenges

Include examples that challenge algorithms, such as:

- **Similar timbres**: Instruments with similar spectral characteristics
- **Heavy reverb**: Reverb blurs boundaries between sources
- **Distorted signals**: Distortion creates complex harmonic content
- **Volume imbalance**: Quiet sources mixed with loud sources
- **Rapid transitions**: Fast musical changes and transients
- **Overlapping frequency ranges**: Sources that occupy the same spectral range

### 4. Technical Requirements

Ensure all test samples:

- **Meet licensing requirements**: Creative Commons or other licenses that permit usage
- **Have consistent format**: WAV files with 44.1kHz sampling rate and 16-bit resolution
- **Include proper documentation**: Full metadata about sources and licensing
- **Have appropriate duration**: 10-30 seconds is ideal (long enough to evaluate, short enough to process efficiently)

## Sources for Good Test Samples

1. **Creative Commons Music**:
   - [Freesound](https://freesound.org/) (search for "multitrack" or "stems")
   - [ccMixter](http://ccmixter.org/) (some tracks offer stems)
   - [Open source sound libraries](https://opensoundlibrary.com/)

2. **Academic Datasets**:
   - [MUSDB18](https://sigsep.github.io/datasets/musdb.html) (standard benchmark dataset)
   - [MedleyDB](https://medleydb.weebly.com/) (small samples available)
   - [BACH10](https://labsites.rochester.edu/air/resource.html) (classical music)

3. **Public Domain Music**:
   - Classical recordings where copyright has expired
   - Government-produced music with no copyright restrictions

4. **Synthetically Created Examples**:
   - Generated with software instruments (good for controlled tests)
   - Custom recordings made specifically for testing

5. **Remix Competitions**:
   - Many artists release stems for remix competitions
   - Always verify license terms before using

## Recommended Organization for Test Dataset

For comprehensive evaluation, organize samples into categories:

1. **Genre-based**: pop, rock, jazz, classical, electronic, other
2. **Challenge-based**: reverberation, distortion, low_volume, similar_timbres, dynamic_range

## Evaluation Criteria for Selected Samples

When selecting a sample, ask:

1. Is this legal to use (licensing)?
2. Does it present a meaningful separation challenge?
3. Does it add diversity to our test set?
4. Are all stems/tracks available with good quality?
5. Is it representative of real-world music?

## Contribution Guidelines

When contributing test samples:

1. Always include complete licensing information
2. Document the source of the files
3. Explain why this sample is valuable for testing
4. Categorize appropriately (genre and/or specific challenge)
5. Ensure all stems are properly aligned and normalized 