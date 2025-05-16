# Audio Source Separation Evaluation Framework

This document outlines a comprehensive framework for evaluating improvements in audio source separation quality for our enhanced Spleeter fork.

## Goals

1. Create a rigorous, reproducible evaluation methodology
2. Establish clear metrics for comparing different approaches
3. Enable both objective and subjective quality assessment
4. Ensure statistical validity of results

## Test Dataset Construction

### Criteria for Test Data

- **Diversity**: Include various music genres, instrumentation types, and production styles
- **Balance**: Ensure balanced representation of difficult separation cases (e.g., similar timbres, heavy effects)
- **Ground Truth**: Use professionally recorded multi-track content where true isolated stems are available
- **Real-World Applicability**: Include commercially produced music similar to what end-users will process

### Proposed Test Sets

1. **MUSDB18**: Industry standard dataset with 150 songs (100 for training, 50 for testing)
2. **MUSDB-HQ**: Higher quality version of MUSDB18 (uncompressed WAV files)
3. **MedleyDB**: Multi-track dataset with more instrument variety 
4. **Custom Test Set**: Hand-selected challenging cases representing common use cases:
   - Songs with vocal harmonies/doubling
   - Heavily compressed/limited mixes
   - Songs with heavy effects processing
   - Classical music with similar timbre instruments

## Objective Evaluation

### Primary Metrics

We will use the `fast_bss_eval` library for efficient calculation of the following metrics:

1. **SDR (Source-to-Distortion Ratio)**
   - Overall measure of separation quality
   - Reported in dB, higher values are better

2. **SI-SDR (Scale-Invariant SDR)**
   - More robust to amplitude scaling issues
   - Better correlation with perceived quality in many cases

3. **SIR (Source-to-Interference Ratio)**
   - Measures how well the target source is isolated from other sources
   - Critical for vocal isolation use cases

4. **SAR (Source-to-Artifact Ratio)**
   - Measures artifacts introduced by the separation process
   - Important for music production applications

### Secondary/Advanced Metrics

1. **Frequency-Dependent SDR**
   - Calculate SDR in different frequency bands
   - Helps identify where models struggle (e.g., low frequencies, transients)

2. **PEAQ (Perceptual Evaluation of Audio Quality)**
   - Model-based approach approximating human perception
   - Provides ODG (Objective Difference Grade) scores

3. **Separation Consistency**
   - Measure how separation quality varies over time
   - Track short-time SDR values to identify problematic segments

## Subjective Evaluation

### MUSHRA Tests

For rigorous subjective evaluation, we'll conduct MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor) tests:

1. **Test Design**
   - 10-15 selected music excerpts (10-20 seconds each)
   - Present multiple separation results alongside hidden reference and low-quality anchor
   - Allow listeners to rate quality on 0-100 scale

2. **Participants**
   - Recruit 15-20 trained listeners (audio engineers, musicians)
   - Include both expert and non-expert listeners for comparison

3. **Evaluation Interface**
   - Web-based MUSHRA implementation
   - Randomized presentation order
   - Ability to loop specific sections

### Focused Listening Attributes

Have participants rate specific aspects of separation quality:

1. **Source Isolation**: How well is the target source isolated?
2. **Artifact Level**: Are there noticeable artifacts (warbling, garbling, etc.)?
3. **Transient Preservation**: How well are attack transients preserved?
4. **Timbre Preservation**: Does the separated source maintain natural timbre?

## Comparative Analysis

### Baseline Comparisons

Compare our improved models against:

1. Original Spleeter implementation
2. Commercial solutions (e.g., iZotope RX, AudioSourceRE)
3. Other open-source alternatives (Demucs, Open-Unmix)

### Visualization Tools

Develop visualizations to help understand performance:

1. **Spectral Difference Plots**: Visualize frequency-dependent separation errors
2. **Performance Radar Charts**: Compare models across multiple metrics simultaneously
3. **Error Distribution Plots**: Show statistical distribution of errors across test set

## Statistical Analysis

Ensure rigorous statistical treatment of results:

1. **Significance Testing**
   - Use paired t-tests or ANOVA to determine if improvements are statistically significant
   - Report p-values for key comparisons

2. **Confidence Intervals**
   - Report 95% confidence intervals for all metrics
   - Visualize uncertainty in results

3. **Error Analysis**
   - Identify songs/cases where our approach performs particularly well or poorly
   - Use this to guide further improvements

## Practical Test Cases

Evaluate on specific real-world use cases:

1. **Vocal Isolation for Karaoke**
   - Focus on clean vocal extraction with minimal artifacts
   - Evaluate accompaniment quality separately

2. **Stem Extraction for Remixing**
   - Test separation of drums, bass, vocals, and other instruments
   - Evaluate how well separated stems can be recombined

3. **Music Production Applications**
   - Test effectiveness for corrective EQ, de-bleeding, etc.
   - Evaluate with professional producers/engineers

## Reproducibility and Documentation

1. **Test Protocol Documentation**
   - Detailed description of test methodology
   - Scripts for running standard evaluation

2. **Result Reporting Template**
   - Standardized format for reporting results
   - Include all relevant metrics and statistical analysis

3. **Public Results Database**
   - Maintain repository of test results for comparison
   - Allow community contributions for comparative analysis

## Implementation Plan

1. **Phase 1: Basic Framework Setup**
   - Implement objective metrics calculation (SDR, SIR, SAR, SI-SDR)
   - Create test dataset curation pipeline
   - Develop basic visualization tools

2. **Phase 2: Advanced Metrics and Analysis**
   - Implement frequency-dependent analysis
   - Add statistical analysis tools
   - Create comparative visualization tools

3. **Phase 3: Subjective Testing Infrastructure**
   - Develop web-based MUSHRA test interface
   - Create participant recruitment and testing protocol
   - Implement results analysis pipeline

4. **Phase 4: Integration and Automation**
   - Automate evaluation for continuous integration
   - Create comprehensive reporting dashboards
   - Develop regression testing framework 