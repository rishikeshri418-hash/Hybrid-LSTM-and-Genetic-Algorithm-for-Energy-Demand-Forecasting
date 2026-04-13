# ga_lstm_optimizer.py
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms
import pickle
import os

class GALSTMOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Hyperparameter ranges
        self.param_ranges = {
            'lstm_units1': (32, 256),
            'lstm_units2': (32, 256),
            'dropout': (0.1, 0.5),
            'learning_rate': (0.0001, 0.01),
            'batch_size': [32, 64, 128]
        }
        
    def create_model(self, individual):
        """Create LSTM model from hyperparameters"""
        lstm_units1 = int(individual[0])
        lstm_units2 = int(individual[1])
        dropout_rate = individual[2]
        learning_rate = individual[3]
        batch_size = self.param_ranges['batch_size'][int(individual[4])]
        
        model = Sequential([
            LSTM(lstm_units1, return_sequences=True, input_shape=(168, 1)),
            Dropout(dropout_rate),
            LSTM(lstm_units2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model, batch_size
    
    def evaluate(self, individual):
        """Evaluate fitness (lower MAPE is better)"""
        try:
            model, batch_size = self.create_model(individual)
            
            # Train briefly to evaluate
            history = model.fit(
                self.X_train[:10000], self.y_train[:10000],  # Use subset for GA speed
                validation_data=(self.X_val[:2000], self.y_val[:2000]),
                epochs=5,
                batch_size=batch_size,
                verbose=0,
                callbacks=[EarlyStopping(patience=2)]
            )
            
            # Use validation MAPE as fitness
            val_mape = history.history['val_mape'][-1]
            
            # Clean up
            del model
            return (val_mape,)
            
        except Exception as e:
            return (100.0,)  # Return high error if fails
    
    def optimize(self, population_size=20, generations=10):
        """Run GA optimization"""
        # Create DEAP classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Define gene ranges
        toolbox.register("attr_lstm1", random.randint, 32, 256)
        toolbox.register("attr_lstm2", random.randint, 32, 256)
        toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
        toolbox.register("attr_lr", random.uniform, 0.0001, 0.01)
        toolbox.register("attr_batch", random.randint, 0, 2)  # 0,1,2 for 32,64,128
        
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_lstm1, toolbox.attr_lstm2,
                          toolbox.attr_dropout, toolbox.attr_lr,
                          toolbox.attr_batch), n=1)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population
        pop = toolbox.population(n=population_size)
        
        # Run GA
        print(f"Running GA for {generations} generations...")
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, 
                            verbose=True)
        
        # Get best individual
        best = tools.selBest(pop, 1)[0]
        best_params = {
            'lstm_units1': int(best[0]),
            'lstm_units2': int(best[1]),
            'dropout': best[2],
            'learning_rate': best[3],
            'batch_size': self.param_ranges['batch_size'][int(best[4])]
        }
        
        return best_params