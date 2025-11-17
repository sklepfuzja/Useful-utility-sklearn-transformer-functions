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

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.
