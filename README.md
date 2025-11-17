# Utility Sklearn Transformer Functions

Collection of custom scikit-learn compatible transformers for feature engineering, selection, and preprocessing in financial machine learning pipelines.

## Available Transformers

### Feature Selection & Cleaning

#### `FeatureSelector`
**Purpose**: Automated feature selection based on correlation thresholds
- Removes features with low correlation to target
- Eliminates highly correlated features (multicollinearity)
- **Parameters**: `threshold=0.5`, `multicollinearity_threshold=0.9`

#### `HighlyCorrelatedFeaturesRemover`
**Purpose**: Removes highly correlated features to reduce dimensionality
- Identifies feature pairs above correlation threshold
- Configurable keeping strategy (first/last)
- **Parameters**: `threshold=0.9`, `method='pearson'`, `keep='first'`

#### `StationaryOutlierRemover`
**Purpose**: Clips outliers only in stationary time series
- Uses Augmented Dickey-Fuller test for stationarity check
- Applies quantile-based clipping to stationary features
- **Parameters**: `threshold=0.95`

### Feature Generation

#### `RandomFeatureGenerator`
**Purpose**: Creates new features using random numpy operations
- Generates features from random column combinations
- Uses numpy mathematical operations
- **Parameters**: `n_features=5`

#### `OpenFEPipeline`
**Purpose**: Automated feature engineering using OpenFE library
- Generates optimized feature combinations
- Parallel processing support
- **Parameters**: `n_jobs=1`

### Dimensionality Reduction

#### `ParitySplitPCA`
**Purpose**: Applies separate PCA to even and odd samples
- Reduces overfitting through data splitting
- Maintains temporal structure
- **Parameters**: `n_components=None`

#### `ClassBasedPCA`
**Purpose**: Separate PCA transformation for each class
- Class-specific dimensionality reduction
- Concatenates results from both classes
- **Parameters**: `n_components=None`

### Mathematical & Statistical Transformers

#### `StationarityTransformer`
**Purpose**: Converts non-stationary time series to stationary
- Automatic stationarity detection using ADF test
- Multiple transformation methods: diff, pct_change, log, log_diff
- **Parameters**: `method='diff'`, `adf_threshold=0.05`

#### `StatisticalFeatureGenerator`
**Purpose**: Generates comprehensive statistical features
- Rolling statistics (mean, std, min, max, median)
- Advanced metrics (skewness, kurtosis, quantiles, z-scores)
- **Parameters**: `window_sizes=[5,10,20,50]`, `include_advanced=True`

### Rolling Advanced Transformers

#### `RollingFourierTransformer`
**Purpose**: Frequency analysis on rolling windows
- Rolling FFT for time-varying frequency content
- Adaptive frequency feature extraction
- **Parameters**: `window_size=100`, `n_components=3`, `step_size=1`

#### `RollingWaveletTransformer`
**Purpose**: Multi-resolution analysis on rolling windows
- Time-localized frequency decomposition
- Adaptive wavelet coefficients
- **Parameters**: `window_size=64`, `wavelet='db4'`, `level=2`

#### `RollingEntropyCalculator`
**Purpose**: Dynamic complexity measurement
- Rolling entropy for changing market regimes
- Time-varying predictability analysis
- **Parameters**: `window_size=100`, `step_size=10`

#### `MultiScaleFeatureGenerator`
**Purpose**: Multi-timeframe statistical features
- Different window sizes for multi-scale analysis
- Scale ratios and comparative statistics
- **Parameters**: `window_scales=[10,30,60,120]`, `features=['mean','std','min','max','range']`


# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.
