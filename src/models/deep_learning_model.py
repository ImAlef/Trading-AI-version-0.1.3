import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Attention,
    MultiHeadAttention, LayerNormalization, Concatenate,
    Flatten, Reshape, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSignalModel:
    def __init__(self, sequence_length: int = 60, confidence_threshold: float = 0.45):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_columns = []
        self.scaler = None
        self.model_path = 'models/'
        self.ensure_model_directory()
        
    def ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        os.makedirs(self.model_path, exist_ok=True)
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build the deep learning model architecture"""
        logger.info(f"Building model with input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape, name='price_sequence')
        
        # CNN layers for pattern recognition
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # LSTM layers for sequence learning
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(conv2)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention = MultiHeadAttention(
            num_heads=8, 
            key_dim=64,
            dropout=0.2
        )(lstm2, lstm2)
        attention = LayerNormalization()(attention)
        
        # Combine LSTM and Attention outputs
        combined = Concatenate()([lstm2, attention])
        
        # Global pooling to reduce sequence dimension
        pooled = GlobalMaxPooling1D()(combined)
        
        # Dense layers for final prediction
        dense1 = Dense(256, activation='relu')(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        # Multi-output model
        # Signal classification (Hold=0, Long=1, Short=2)
        signal_output = Dense(3, activation='softmax', name='signal')(dense2)
        
        # Entry price prediction (regression)
        entry_price = Dense(64, activation='relu')(dense2)
        entry_price = Dense(1, activation='linear', name='entry_price')(entry_price)
        
        # Take profit prediction (regression)
        take_profit = Dense(64, activation='relu')(dense2)
        take_profit = Dense(1, activation='linear', name='take_profit')(take_profit)
        
        # Stop loss prediction (regression)
        stop_loss = Dense(64, activation='relu')(dense2)
        stop_loss = Dense(1, activation='linear', name='stop_loss')(stop_loss)
        
        # Confidence score (regression, 0-1)
        confidence = Dense(32, activation='relu')(dense2)
        confidence = Dense(1, activation='sigmoid', name='confidence')(confidence)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[signal_output, entry_price, take_profit, stop_loss, confidence],
            name='TradingSignalModel'
        )
        
        return model
    
    def compile_model(self, model: Model):
        """Compile the model with appropriate losses and metrics"""
        # Define custom loss for trading signals
        def trading_loss(y_true, y_pred):
            # Weighted categorical crossentropy
            cce = SparseCategoricalCrossentropy()(y_true, y_pred)
            return cce
        
        def profit_aware_loss(y_true, y_pred):
            # MSE with profit awareness
            mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            return mse
        
        def confidence_loss(y_true, y_pred):
            # Binary crossentropy for confidence
            return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        
        # Compile with multiple losses
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'signal': trading_loss,
                'entry_price': 'mse',
                'take_profit': profit_aware_loss,
                'stop_loss': profit_aware_loss,
                'confidence': 'mse'
            },
            loss_weights={
                'signal': 2.0,        # Higher weight for signal prediction
                'entry_price': 0.5,
                'take_profit': 1.0,
                'stop_loss': 1.0,
                'confidence': 1.5
            },
            metrics={
                'signal': ['accuracy'],
                'entry_price': ['mae'],
                'take_profit': ['mae'],
                'stop_loss': ['mae'],
                'confidence': ['mae']
            }
        )
        
        logger.info("Model compiled successfully")
        return model
    
    def prepare_training_data(self, training_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for training"""
        logger.info("Preparing training data")
        
        all_X = []
        all_signals = []
        all_entry_prices = []
        all_take_profits = []
        all_stop_losses = []
        all_confidences = []
        
        for symbol, df in training_data.items():
            if df.empty:
                continue
            
            logger.info(f"Processing {symbol}: {len(df)} samples")
            
            # Select feature columns (exclude target columns)
            target_cols = ['signal', 'entry_price', 'take_profit', 'stop_loss', 'future_return', 'hold_period']
            feature_cols = [col for col in df.columns if col not in target_cols]
            
            if not self.feature_columns:
                self.feature_columns = feature_cols
            
            # Prepare features
            features = df[feature_cols].values
            
            # Prepare targets
            signals = df['signal'].values + 1  # Convert -1,0,1 to 0,1,2
            entry_prices = df['entry_price'].values
            take_profits = df['take_profit'].values
            stop_losses = df['stop_loss'].values
            
            # Calculate confidence based on future returns
            future_returns = df['future_return'].values
            confidences = np.clip(np.abs(future_returns) * 10, 0, 1)  # Scale to 0-1
            
            # Create sequences
            if len(features) > self.sequence_length:
                X_seq, y_signals = self.create_sequences(features, signals)
                _, y_entry = self.create_sequences(features, entry_prices)
                _, y_tp = self.create_sequences(features, take_profits)
                _, y_sl = self.create_sequences(features, stop_losses)
                _, y_conf = self.create_sequences(features, confidences)
                
                all_X.append(X_seq)
                all_signals.append(y_signals)
                all_entry_prices.append(y_entry)
                all_take_profits.append(y_tp)
                all_stop_losses.append(y_sl)
                all_confidences.append(y_conf)
        
        if not all_X:
            raise ValueError("No valid training data found")
        
        # Combine all data
        X = np.concatenate(all_X, axis=0)
        y = {
            'signal': np.concatenate(all_signals, axis=0),
            'entry_price': np.concatenate(all_entry_prices, axis=0),
            'take_profit': np.concatenate(all_take_profits, axis=0),
            'stop_loss': np.concatenate(all_stop_losses, axis=0),
            'confidence': np.concatenate(all_confidences, axis=0)
        }
        
        logger.info(f"Total training samples: {len(X)}")
        logger.info(f"Signal distribution: {np.bincount(y['signal'])}")
        
        return X, y
    
    def train(self, training_data: Dict[str, pd.DataFrame], 
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> dict:
        """Train the model"""
        logger.info("Starting model training")
        
        # Prepare data
        X, y = self.prepare_training_data(training_data)
        
        # Split data
        indices = np.arange(len(X))
        X_train_idx, X_val_idx = train_test_split(
            indices, test_size=validation_split, random_state=42, stratify=y['signal']
        )
        
        X_train = X[X_train_idx]
        X_val = X[X_val_idx]
        
        # Split targets
        y_train = {}
        y_val = {}
        for key in y.keys():
            y_train[key] = y[key][X_train_idx]
            y_val[key] = y[key][X_val_idx]
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train['signal']),
            y=y_train['signal']
        )
        class_weight_dict = {0: {i: weight for i, weight in enumerate(class_weights)}}  # Only for signal output
        
        # Build and compile model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        self.model = self.compile_model(self.model)
        
        # Print model summary
        self.model.summary()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        
        # Save model and metadata
        self.save_model()
        
        return history.history
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(X)
        
        # Unpack predictions
        signal_probs, entry_prices, take_profits, stop_losses, confidences = predictions
        
        # Convert signal probabilities to classes
        signals = np.argmax(signal_probs, axis=1) - 1  # Convert back to -1,0,1
        
        # Get confidence scores for the predicted signals
        predicted_confidences = np.max(signal_probs, axis=1)
        
        return {
            'signal': signals,
            'signal_probabilities': signal_probs,
            'entry_price': entry_prices.flatten(),
            'take_profit': take_profits.flatten(),
            'stop_loss': stop_losses.flatten(),
            'confidence': confidences.flatten(),
            'prediction_confidence': predicted_confidences
        }
    
    def predict_single(self, sequence: np.ndarray) -> Dict[str, float]:
        """Make prediction for a single sequence"""
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        predictions = self.predict(sequence)
        
        # Extract single prediction
        result = {
            'signal': int(predictions['signal'][0]),
            'entry_price': float(predictions['entry_price'][0]),
            'take_profit': float(predictions['take_profit'][0]),
            'stop_loss': float(predictions['stop_loss'][0]),
            'confidence': float(predictions['confidence'][0]),
            'prediction_confidence': float(predictions['prediction_confidence'][0])
        }
        
        # Check confidence threshold
        if result['prediction_confidence'] < self.confidence_threshold:
            result['signal'] = 0  # Hold if confidence is low
        
        return result
    
    def save_model(self):
        """Save model and metadata"""
        model_file = os.path.join(self.model_path, 'trading_model.h5')
        weights_file = os.path.join(self.model_path, 'model_weights.h5')
        metadata_file = os.path.join(self.model_path, 'model_metadata.pkl')
        
        # Save model weights only (avoids custom loss function issues)
        self.model.save_weights(weights_file)
        
        # Save model architecture
        model_json = self.model.to_json()
        with open(os.path.join(self.model_path, 'model_architecture.json'), 'w') as f:
            f.write(model_json)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'confidence_threshold': self.confidence_threshold,
            'feature_columns': self.feature_columns
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {weights_file}")
    
    def load_model(self):
        """Load model and metadata"""
        weights_file = os.path.join(self.model_path, 'model_weights.h5')
        architecture_file = os.path.join(self.model_path, 'model_architecture.json')
        metadata_file = os.path.join(self.model_path, 'model_metadata.pkl')
        
        if not os.path.exists(weights_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError("Model files not found")
        
        # Load metadata first
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.sequence_length = metadata['sequence_length']
        self.confidence_threshold = metadata['confidence_threshold']
        self.feature_columns = metadata['feature_columns']
        
        # Rebuild model architecture
        if os.path.exists(architecture_file):
            # Load from JSON
            with open(architecture_file, 'r') as f:
                model_json = f.read()
            self.model = tf.keras.models.model_from_json(model_json)
        else:
            # Rebuild from scratch
            input_shape = (self.sequence_length, len(self.feature_columns))
            self.model = self.build_model(input_shape)
            self.model = self.compile_model(self.model)
        
        # Load weights
        self.model.load_weights(weights_file)
        
        logger.info("Model loaded successfully")
    
    def evaluate_model(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model performance")
        
        X_test, y_test = self.prepare_training_data(test_data)
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        signal_accuracy = np.mean(predictions['signal'] == (y_test['signal'] - 1))
        
        # Calculate win rate for trading signals
        trading_signals = predictions['signal'] != 0
        if np.sum(trading_signals) > 0:
            correct_trades = np.sum(
                (predictions['signal'][trading_signals] == (y_test['signal'][trading_signals] - 1))
            )
            win_rate = correct_trades / np.sum(trading_signals)
        else:
            win_rate = 0.0
        
        # Calculate confidence accuracy
        high_conf_mask = predictions['prediction_confidence'] >= self.confidence_threshold
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(
                predictions['signal'][high_conf_mask] == (y_test['signal'][high_conf_mask] - 1)
            )
        else:
            high_conf_accuracy = 0.0
        
        metrics = {
            'overall_accuracy': signal_accuracy,
            'win_rate': win_rate,
            'high_confidence_accuracy': high_conf_accuracy,
            'total_signals': np.sum(trading_signals),
            'high_confidence_signals': np.sum(high_conf_mask)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = TradingSignalModel(sequence_length=60, confidence_threshold=0.45)
    
    # Load training data (this would come from feature engineering)
    # training_data = load_training_data()
    
    # Train model
    # history = model.train(training_data, epochs=50)
    
    # Evaluate model
    # metrics = model.evaluate_model(test_data)
    
    print("Trading model ready for training")