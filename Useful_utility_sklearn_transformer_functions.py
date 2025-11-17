import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from openfe import OpenFE, transform
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, multicollinearity_threshold=0.9):
        self.threshold = threshold
        self.multicollinearity_threshold = multicollinearity_threshold
        self.features_to_remove = []
    
    def fit(self, X, y=None):
        # calculate Pearson correlation matrix
        corr_matrix = X.corr(method='pearson').abs()

        # identify features below threshold with respect to target
        if y is not None:
            target_corr = X.apply(lambda x: x.corr(y)).abs()
            self.features_to_remove = target_corr[target_corr <= self.threshold].index
        else:
            self.features_to_remove = []

        if self.multicollinearity_threshold:
            # calculate correlation matrix again for multicollinearity check
            multicollinearity_matrix = corr_matrix

            # identify features highly correlated with each other
            upper_triangle = multicollinearity_matrix.where(np.triu(np.ones(multicollinearity_matrix.shape), k=1).astype(bool))
            # Append features that are highly correlated with each other
            to_remove = upper_triangle[upper_triangle >= self.multicollinearity_threshold].stack().index
            self.features_to_remove = self.features_to_remove.union(to_remove.get_level_values(1))

        return self
    
    def transform(self, X):
        # drop features to be removed
        return X.drop(columns=self.features_to_remove)
    
class StationaryOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def check_stationarity(self, X):
        if np.all(X == X.iloc[0]):
            return False  # Seria jest stała, więc nie jest stacjonarna
        result = adfuller(X)
        p_value = result[1]
        return p_value < 0.05

    def fit(self, X, y=None):
        # Assuming X is a DataFrame and we apply the check to each column
        self.bounds_ = {}
        for column in X.columns:
            if self.check_stationarity(X[column]):
                upper_bound = X[column].quantile(self.threshold)
                lower_bound = X[column].quantile(1 - self.threshold)
                self.bounds_[column] = (lower_bound, upper_bound)
        return self

    def transform(self, X):
        X_clipped = X.copy()
        for column, bounds in self.bounds_.items():
            lower_bound, upper_bound = bounds
            X_clipped[column] = X[column].clip(lower_bound, upper_bound)
        return X_clipped
    
from sklearn.base import BaseEstimator, TransformerMixin

class HighlyCorrelatedFeaturesRemover(BaseEstimator, TransformerMixin):
    """
    Usuwa wysoko skorelowane cechy z danych.
    
    Parametry:
    ----------
    threshold : float, domyślnie 0.9
        Próg korelacji powyżej którego cechy są uznawane za wysoko skorelowane.
    
    method : str, domyślnie 'pearson'
        Metoda obliczania korelacji ('pearson', 'kendall', 'spearman').
        
    keep : str, domyślnie 'first'
        Którą cechę zachować z pary skorelowanych cech:
        - 'first': zachowuje pierwszą cechę
        - 'last': zachowuje ostatnią cechę
    """
    
    def __init__(self, threshold=0.9, method='pearson', keep='first'):
        self.threshold = threshold
        self.method = method
        self.keep = keep
        self.correlated_features_to_drop_ = None
        
    def fit(self, X, y=None):
        """
        Identyfikuje pary wysoko skorelowanych cech.
        
        Parametry:
        ----------
        X : pd.DataFrame lub np.ndarray
            Dane wejściowe.
        y : ignorowane
            Dla zgodności z interfejsem scikit-learn.
            
        Zwraca:
        -------
        self : obiekt
            Instancja transformer'a.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        corr_matrix = X.corr(method=self.method).abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        if self.keep == 'first':
            self.correlated_features_to_drop_ = [column for column in upper.columns 
                                               if any(upper[column] > self.threshold)]
        elif self.keep == 'last':
            # Odwracamy kolejność, aby zachować ostatnią cechę
            self.correlated_features_to_drop_ = []
            for column in reversed(upper.columns):
                if any(upper[column] > self.threshold):
                    self.correlated_features_to_drop_.append(column)
        else:
            raise ValueError("Parametr 'keep' musi być 'first' lub 'last'")
            
        return self
    
    def transform(self, X, y=None):
        """
        Usuwa wysoko skorelowane cechy.
        
        Parametry:
        ----------
        X : pd.DataFrame lub np.ndarray
            Dane wejściowe.
        y : ignorowane
            Dla zgodności z interfejsem scikit-learn.
            
        Zwraca:
        -------
        X_transformed : pd.DataFrame lub np.ndarray
            Dane z usuniętymi cechami.
        """
        if self.correlated_features_to_drop_ is None:
            raise RuntimeError("Najpierw wywołaj fit() przed transform()")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.drop(columns=self.correlated_features_to_drop_)
        
        # Zwracamy ten sam typ, co wejście
        if isinstance(X, pd.DataFrame):
            return X_transformed
        else:
            return X_transformed.values
    
    def fit_transform(self, X, y=None):
        """Wywołuje fit() a następnie transform()."""
        return self.fit(X, y).transform(X, y)
    
class RandomFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=5):
        self.n_features = n_features
        self.numpy_ops = np.__dict__['__all__']
        self.operations = [op for op in self.numpy_ops if callable(np[op])]

    def fit(self, X, y=None):
        self.input_dim = X.shape[1]
        return self

    def transform(self, X):
        result = np.zeros((X.shape[0], self.n_features))
        for i in range(self.n_features):
            index = np.random.choice(self.input_dim, 2, replace=False)
            operation = np.random.choice(self.operations)
            result[..., i] = getattr(np, operation)(X[:, index[0]], X[:, index[1]])
        return result

class OpenFEPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.openfe = OpenFE()  # Initialize OpenFE instance
        self.features = None  # Placeholder for generated features
        
    def fit(self, X, y=None):
        # Fit method to generate new features with OpenFE and store in self.features
        self.features = self.openfe.fit(data=X, label=y, n_jobs=self.n_jobs)
        return self
    
    def transform(self, X):
        # Apply OpenFE transform to X using the generated features stored in self.features
        X_transformed = transform(X, None, self.features, n_jobs=self.n_jobs)
        return X_transformed
    
class ParitySplitPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None): 
        self.n_components = n_components
        self.pca_even = PCA(n_components=self.n_components)
        self.pca_odd = PCA(n_components=self.n_components)
        
    def fit(self, X, y=None):
        # Podziel X i y na próbki parzyste i nieparzyste
        X_even, y_even = X[::2], (y[::2] if y is not None else None)
        X_odd, y_odd = X[1::2], (y[1::2] if y is not None else None)
        
        # Dopasuj PCA oddzielnie do parzystych i nieparzystych próbek
        self.pca_even.fit(X_even, y_even)
        self.pca_odd.fit(X_odd, y_odd)
        
        return self
    
    def transform(self, X, y=None):
        # Podziel X na próbki parzyste i nieparzyste
        X_even, X_odd = X[::2], X[1::2]
        
        # Przekształć dane za pomocą odpowiednich PCA
        X_even_transformed = self.pca_even.transform(X_even)
        X_odd_transformed = self.pca_odd.transform(X_odd)
        
        # Połącz przekształcone dane z powrotem w jeden zbiór
        X_transformed = np.empty_like(X)
        X_transformed[::2] = X_even_transformed
        X_transformed[1::2] = X_odd_transformed
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_params(self, deep=True):
        return {'n_components': self.n_components}
    
    def set_params(self, **params):
        if 'n_components' in params:
            self.n_components = params['n_components']
        # Re-inicjalizacja PCA z nową liczbą komponentów, jeśli jest to konieczne
        if 'n_components' in params:
            self.pca_even = PCA(n_components=self.n_components)
            self.pca_odd = PCA(n_components=self.n_components)
        return self
    
class ClassBasedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca_class_1 = PCA(n_components=self.n_components)
        self.pca_class_0 = PCA(n_components=self.n_components)
    
    def fit(self, X, y):
        # Fit the PCA models for each class
        self.pca_class_1.fit(X[y == 1])
        self.pca_class_0.fit(X[y == 0])
        return self
    
    def transform(self, X, y=None):
        # Apply the transformations and concatenate the results
        X_class_1_transformed = self.pca_class_1.transform(X)
        X_class_0_transformed = self.pca_class_0.transform(X)
        return np.concatenate((X_class_1_transformed, X_class_0_transformed), axis=1)
    
class StationarityTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms non-stationary time series to stationary using various methods.
    
    Parameters:
    -----------
    method : str, default='diff'
        Transformation method: 'diff', 'pct_change', 'log', 'log_diff'
    adf_threshold : float, default=0.05
        ADF test p-value threshold for stationarity
    columns_to_transform : list, optional
        Specific columns to transform (if None, transforms all numeric columns)
    """
    
    def __init__(self, method='diff', adf_threshold=0.05, columns_to_transform=None):
        self.method = method
        self.adf_threshold = adf_threshold
        self.columns_to_transform = columns_to_transform
        self.stationary_columns_ = []
        self.non_stationary_columns_ = []
        self.original_dtypes_ = {}
        
    def _is_stationary(self, series):
        """Check stationarity using Augmented Dickey-Fuller test."""
        if len(series) < 10 or series.nunique() <= 1:
            return False
            
        try:
            result = adfuller(series.dropna())
            return result[1] < self.adf_threshold
        except:
            return False
    
    def _transform_series(self, series, method):
        """Apply transformation to a single series."""
        if method == 'diff':
            return series.diff()
        elif method == 'pct_change':
            return series.pct_change()
        elif method == 'log':
            return np.log(series.replace(0, np.nan))  # Avoid log(0)
        elif method == 'log_diff':
            return np.log(series.replace(0, np.nan)).diff()
        else:
            return series
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.original_dtypes_ = X.dtypes.to_dict()
        
        # Determine which columns to transform
        if self.columns_to_transform is None:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
        else:
            numeric_columns = [col for col in self.columns_to_transform if col in X.columns]
        
        # Identify stationary vs non-stationary columns
        for col in numeric_columns:
            if self._is_stationary(X[col]):
                self.stationary_columns_.append(col)
            else:
                self.non_stationary_columns_.append(col)
        
        print(f"📊 Stationarity Analysis:")
        print(f"   Stationary columns: {len(self.stationary_columns_)}")
        print(f"   Non-stationary columns: {len(self.non_stationary_columns_)}")
        
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        
        # Apply transformations only to non-stationary columns
        for col in self.non_stationary_columns_:
            if col in X_transformed.columns:
                X_transformed[col] = self._transform_series(X_transformed[col], self.method)
        
        return X_transformed.fillna(0)

class StatisticalFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates statistical features for time series data.
    
    Parameters:
    -----------
    window_sizes : list, default=[5, 10, 20, 50]
        Rolling window sizes for statistical calculations
    include_advanced : bool, default=True
        Whether to include advanced statistical features
    """
    
    def __init__(self, window_sizes=[5, 10, 20, 50], include_advanced=True):
        self.window_sizes = window_sizes
        self.include_advanced = include_advanced
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for window in self.window_sizes:
                # Basic statistical features
                X_transformed[f'{col}_rolling_mean_{window}'] = X[col].rolling(window).mean()
                X_transformed[f'{col}_rolling_std_{window}'] = X[col].rolling(window).std()
                X_transformed[f'{col}_rolling_min_{window}'] = X[col].rolling(window).min()
                X_transformed[f'{col}_rolling_max_{window}'] = X[col].rolling(window).max()
                X_transformed[f'{col}_rolling_median_{window}'] = X[col].rolling(window).median()
                
                # Advanced statistical features
                if self.include_advanced:
                    # Skewness and Kurtosis
                    X_transformed[f'{col}_rolling_skew_{window}'] = X[col].rolling(window).skew()
                    X_transformed[f'{col}_rolling_kurtosis_{window}'] = X[col].rolling(window).kurt()
                    
                    # Quantiles
                    X_transformed[f'{col}_rolling_q25_{window}'] = X[col].rolling(window).quantile(0.25)
                    X_transformed[f'{col}_rolling_q75_{window}'] = X[col].rolling(window).quantile(0.75)
                    
                    # Z-score (current value vs rolling stats)
                    rolling_mean = X[col].rolling(window).mean()
                    rolling_std = X[col].rolling(window).std()
                    X_transformed[f'{col}_zscore_{window}'] = (X[col] - rolling_mean) / rolling_std
                    
                    # Momentum and change features
                    X_transformed[f'{col}_momentum_{window}'] = X[col] - X[col].shift(window)
                    X_transformed[f'{col}_roc_{window}'] = (X[col] / X[col].shift(window) - 1) * 100
        
        return X_transformed.fillna(0)
    
class RollingFourierTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Fourier Transform on rolling windows.
    
    Parameters:
    -----------
    window_size : int, default=100
        Size of rolling window
    n_components : int, default=3
        Number of Fourier components to extract
    step_size : int, default=1
        Step size between windows
    """
    
    def __init__(self, window_size=100, n_components=3, step_size=1):
        self.window_size = window_size
        self.n_components = n_components
        self.step_size = step_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = X[col].fillna(0).values
            
            # Initialize result arrays
            fft_features = np.full((len(series), self.n_components), np.nan)
            
            # Rolling FFT
            for i in range(self.window_size, len(series), self.step_size):
                window = series[i-self.window_size:i]
                
                try:
                    # Apply FFT to window
                    fft_values = np.fft.fft(window)
                    magnitudes = np.abs(fft_values)
                    
                    # Take first n_components (excluding DC component)
                    for j in range(1, min(self.n_components + 1, len(magnitudes))):
                        fft_features[i, j-1] = magnitudes[j]
                        
                except:
                    continue
            
            # Add to DataFrame
            for j in range(self.n_components):
                X_transformed[f'{col}_rolling_fft_{j+1}'] = fft_features[:, j]
        
        return X_transformed.fillna(0)

class RollingWaveletTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Wavelet Transform on rolling windows.
    
    Parameters:
    -----------
    window_size : int, default=64
        Size of rolling window (should be power of 2 for wavelets)
    wavelet : str, default='db4'
        Type of wavelet to use
    level : int, default=2
        Decomposition level
    """
    
    def __init__(self, window_size=64, wavelet='db4', level=2):
        self.window_size = window_size
        self.wavelet = wavelet
        self.level = level
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        try:
            import pywt
        except ImportError:
            print("⚠️ pywt package required for RollingWaveletTransformer")
            return X
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = X[col].fillna(0).values
            
            # Initialize result arrays for approximation coefficients
            wavelet_features = np.full((len(series), self.level + 1), np.nan)
            
            # Rolling wavelet transform
            for i in range(self.window_size, len(series)):
                window = series[i-self.window_size:i]
                
                try:
                    # Ensure window length is sufficient
                    if len(window) >= 2**self.level:
                        coeffs = pywt.wavedec(window, self.wavelet, level=self.level)
                        
                        # Store approximation coefficients
                        for j, coeff in enumerate(coeffs):
                            if j < wavelet_features.shape[1]:
                                wavelet_features[i, j] = np.mean(coeff) if len(coeff) > 0 else 0
                                
                except Exception as e:
                    continue
            
            # Add to DataFrame
            for j in range(self.level + 1):
                X_transformed[f'{col}_rolling_wavelet_cA_{j}'] = wavelet_features[:, j]
        
        return X_transformed.fillna(0)

class RollingEntropyCalculator(BaseEstimator, TransformerMixin):
    """
    Calculates entropy measures on rolling windows.

    Parameters:
    -----------
    window_size : int, default=100
        Size of rolling window
    step_size : int, default=10
        Step size between windows
    """

    def __init__(self, window_size=100, step_size=10):
        self.window_size = window_size
        self.step_size = step_size

    def _approximate_entropy(self, series, m=2, r=None):
        """Calculate approximate entropy for a window."""
        if r is None:
            r = 0.2 * np.std(series)
        if np.std(series) == 0:
            return 0

        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[series[j] for j in range(i, i + m)] for i in range(len(series) - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (len(series) - m + 1.0) for x_i in x]
            return np.sum(np.log(C)) / (len(series) - m + 1.0)

        return abs(_phi(m + 1) - _phi(m))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            series = X[col].fillna(0).values

            # Initialize result array
            entropy_values = np.full(len(series), np.nan)

            # Rolling entropy calculation
            for i in range(self.window_size, len(series), self.step_size):
                window = series[i - self.window_size : i]   # only past values

                try:
                    entropy = self._approximate_entropy(window)

                    # NEW: assign ONLY to the current index i
                    entropy_values[i] = entropy

                except:
                    continue

            X_transformed[f'{col}_rolling_entropy'] = entropy_values

        return X_transformed.fillna(0)

class MultiScaleFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates multi-scale statistical features using different rolling windows.
    
    Parameters:
    -----------
    window_scales : list, default=[10, 30, 60, 120]
        Different window sizes for multi-scale analysis
    features : list, default=['mean', 'std', 'min', 'max', 'range']
        Statistical features to calculate
    """
    
    def __init__(self, window_scales=[10, 30, 60, 120], features=['mean', 'std', 'min', 'max', 'range']):
        self.window_scales = window_scales
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_transformed = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for window in self.window_scales:
                # Basic rolling statistics
                if 'mean' in self.features:
                    X_transformed[f'{col}_scale_{window}_mean'] = X[col].rolling(window).mean()
                
                if 'std' in self.features:
                    X_transformed[f'{col}_scale_{window}_std'] = X[col].rolling(window).std()
                
                if 'min' in self.features:
                    X_transformed[f'{col}_scale_{window}_min'] = X[col].rolling(window).min()
                
                if 'max' in self.features:
                    X_transformed[f'{col}_scale_{window}_max'] = X[col].rolling(window).max()
                
                if 'range' in self.features:
                    rolling_min = X[col].rolling(window).min()
                    rolling_max = X[col].rolling(window).max()
                    X_transformed[f'{col}_scale_{window}_range'] = rolling_max - rolling_min
                
                # Advanced multi-scale features
                X_transformed[f'{col}_scale_{window}_zscore'] = (
                    (X[col] - X[col].rolling(window).mean()) / X[col].rolling(window).std()
                )
                
                # Scale ratio (short-term vs long-term)
                if window > min(self.window_scales):
                    short_window = min(self.window_scales)
                    X_transformed[f'{col}_scale_ratio_{window}'] = (
                        X[col].rolling(short_window).std() / X[col].rolling(window).std()
                    )
        
        return X_transformed.fillna(0)