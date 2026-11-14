"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, normalization, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """
    Preprocesses cloud resource data for machine learning models.
    """
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        scaler_type : str
            'minmax' for MinMaxScaler (0-1 range) or 'standard' for StandardScaler
        """
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.feature_scaler = None
        self.target_scaler = None
        self.is_fitted = False
        self.feature_columns = None
        self.target_columns = None
    
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        
        Returns:
        --------
        pd.DataFrame : Loaded dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records from {file_path}")
        return df
    
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'forward_fill', 'backward_fill', 'mean', or 'drop'
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe
        """
        original_len = len(df)
        
        if method == 'drop':
            df = df.dropna()
        elif method == 'forward_fill':
            df = df.ffill()  # Forward fill (newer pandas syntax)
        elif method == 'backward_fill':
            df = df.bfill()  # Backward fill (newer pandas syntax)
        elif method == 'mean':
            df = df.fillna(df.mean())
        
        df = df.bfill()  # Fill any remaining NaN
        
        if len(df) < original_len:
            print(f"⚠️  Removed {original_len - len(df)} rows with missing values")
        else:
            print("✅ No missing values found")
        
        return df
    
    def create_sequences(self, data, sequence_length=24, prediction_horizon=1):
        """
        Create time-series sequences for LSTM/RNN models.
        
        Parameters:
        -----------
        data : np.array
            Time-series data (features)
        sequence_length : int
            Number of time steps to look back
        prediction_horizon : int
            Number of steps ahead to predict
        
        Returns:
        --------
        tuple : (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence (past values)
            X.append(data[i:i + sequence_length])
            # Target (future values)
            y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, df, sequence_length=24, test_size=0.2,
                          features=None, targets=None):
        """Prepare chronological LSTM data without fitting scalers on test rows."""
        if features is None:
            features = ['cpu_usage', 'ram_usage', 'requests']
        if targets is None:
            targets = ['cpu_usage', 'ram_usage']

        missing = [c for c in list(features) + list(targets) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing preprocessing columns: {missing}")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if len(df) <= sequence_length + 1:
            raise ValueError("Not enough rows to create LSTM sequences")

        self.feature_columns = list(features)
        self.target_columns = list(targets)
        feature_data = df[features].to_numpy(dtype=float)
        target_data = df[targets].to_numpy(dtype=float)

        # Split by time before fitting scalers to avoid test-set leakage.
        train_row_end = int(len(df) * (1 - test_size))
        if train_row_end <= sequence_length:
            raise ValueError("Training partition is too small for the sequence length")

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_scaler.fit(feature_data[:train_row_end])
        self.target_scaler.fit(target_data[:train_row_end])
        feature_scaled = self.feature_scaler.transform(feature_data)
        target_scaled = self.target_scaler.transform(target_data)

        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(feature_scaled[i:i + sequence_length])
            y.append(target_scaled[i + sequence_length])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # A sample belongs to train only when its target row is before the
        # chronological split. Test sequences may use preceding history.
        target_rows = np.arange(sequence_length, len(df))
        train_mask = target_rows < train_row_end
        test_mask = ~train_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        self.is_fitted = True

        print("Prepared LSTM data (chronological split; scalers fitted on training rows):")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Features: {features}")
        print(f"   - Targets: {targets}")
        return X_train, X_test, y_train, y_test

    def prepare_rf_data(self, df, test_size=0.2, lookback=24,
                       features=['cpu_usage', 'ram_usage', 'assigned_mem_usage'],
                       targets=['cpu_usage', 'ram_usage']):
        """
        Prepare data for Random Forest model (with lag features).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        test_size : float
            Proportion of data for testing
        lookback : int
            Number of lag features to create
        features : list
            Feature columns to use
        targets : list
            Target columns to predict
        
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, scaler)
        """
        # Create lag features
        self.feature_columns = list(features)
        self.target_columns = list(targets)
        df_processed = df.copy()
        
        for col in features:
            for lag in range(1, lookback + 1):
                df_processed[f'{col}_lag_{lag}'] = df_processed[col].shift(lag)
        
        # Drop rows with NaN (from lag creation)
        df_processed = df_processed.dropna()
        
        # Prepare features (all lag features)
        feature_cols = [f'{col}_lag_{lag}' for col in features for lag in range(1, lookback + 1)]
        X = df_processed[feature_cols].values
        y = df_processed[targets].values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Scale targets
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
        
        self.is_fitted = True
        
        print(f"✅ Prepared Random Forest data:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Lookback window: {lookback}")
        print(f"   - Total features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_targets(self, y_scaled):
        """
        Inverse transform scaled predictions back to original scale.
        
        Parameters:
        -----------
        y_scaled : np.array
            Scaled predictions
        
        Returns:
        --------
        np.array : Predictions in original scale
        """
        if not self.is_fitted or self.target_scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_*_data first.")
        
        return self.target_scaler.inverse_transform(y_scaled)


if __name__ == "__main__":
    # Test preprocessing
    print("🔄 Testing data preprocessing...")
    
    preprocessor = DataPreprocessor()
    
    # Load data (assuming it exists)
    try:
        df = preprocessor.load_data("data/simulated_cloud_data.csv")
        df = preprocessor.handle_missing_values(df)
        
        # Test LSTM preparation
        print("\n--- LSTM Data Preparation ---")
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = \
            preprocessor.prepare_lstm_data(df, sequence_length=24)
        
        # Test RF preparation
        print("\n--- Random Forest Data Preparation ---")
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = \
            preprocessor.prepare_rf_data(df, lookback=24)
        
        print("\n✅ Preprocessing test completed successfully!")
        
    except FileNotFoundError:
        print("⚠️  Data file not found. Run data_simulation.py first.")

