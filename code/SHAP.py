import numpy as np
from typing import Callable, Union, Optional, List
from itertools import combinations
import warnings


class Explainer:
    
    def __init__(self, 
                 model: Callable[[np.ndarray], Union[np.ndarray, float]], 
                 data: np.ndarray,
                 max_evals: int = 2000,
                 feature_names: Optional[List[str]] = None):
        self.model = model
        self.max_evals = max_evals
        self.feature_names = feature_names
        
        if len(data) > 100:
            warnings.warn(f"Background data has {len(data)} samples. Using random sample of 100.")
            indices = np.random.choice(len(data), 100, replace=False)
            self.data = data[indices]
        else:
            self.data = data
        
        self.expected_value = np.mean(self.model(self.data))
    
    def __call__(self, X: np.ndarray, max_evals: Optional[int] = None) -> 'Explanation':
        return self.shap_values(X, max_evals)
    
    def shap_values(self, X: np.ndarray, max_evals: Optional[int] = None) -> 'Explanation':

        X = np.atleast_2d(X)
        max_evals = max_evals or self.max_evals
        n_instances, n_features = X.shape
        
        total_evals = n_instances * (2 ** n_features)
        if total_evals > max_evals and n_features > 10:
            raise ValueError(f"Exact computation requires {total_evals} evaluations "
                           f"but max_evals is {max_evals}. Reduce features or use approximation.")
        
        shap_values = np.array([self._compute_shapley_values(instance) 
                               for instance in X])
        
        return Explanation(
            values=shap_values,
            base_values=np.full(n_instances, self.expected_value),
            data=X,
            feature_names=self.feature_names
        )
    
    def _compute_shapley_values(self, instance: np.ndarray) -> np.ndarray:
        n_features = len(instance)
        
        coalition_values = self._get_coalition_values(instance)
        
        shapley_values = np.zeros(n_features)
        
        for i in range(n_features):
            shapley_value = 0.0

            for coalition_size in range(n_features):
                weight = self._shapley_weight(coalition_size, n_features)
                
                for coalition in combinations(
                    [j for j in range(n_features) if j != i], coalition_size
                ):
                    coalition_tuple = tuple(sorted(coalition))
                    coalition_with_i = tuple(sorted(coalition + (i,)))
                    
                    marginal_contribution = (coalition_values[coalition_with_i] - 
                                           coalition_values[coalition_tuple])
                    shapley_value += weight * marginal_contribution
            
            shapley_values[i] = shapley_value
        
        return shapley_values
    
    def _get_coalition_values(self, instance: np.ndarray) -> dict:
        n_features = len(instance)
        coalition_values = {tuple(): self.expected_value}
        
        for size in range(1, n_features + 1):
            for coalition in combinations(range(n_features), size):
                coalition_tuple = tuple(coalition)
                coalition_values[coalition_tuple] = self._evaluate_coalition(
                    coalition_tuple, instance
                )
        
        return coalition_values
    
    def _evaluate_coalition(self, coalition: tuple, instance: np.ndarray) -> float:
        masked_data = self.data.copy()
        for feature_idx in coalition:
            masked_data[:, feature_idx] = instance[feature_idx]
        
        return np.mean(self.model(masked_data))
    
    @staticmethod
    def _shapley_weight(coalition_size: int, n_features: int) -> float:
        if n_features == 1:
            return 1.0
        
        weight = 1.0
        for i in range(coalition_size):
            weight *= (n_features - coalition_size) / (n_features - i)
        return weight / n_features


class Explanation:
    
    def __init__(self, 
                 values: np.ndarray,
                 base_values: np.ndarray,
                 data: np.ndarray,
                 feature_names: Optional[List[str]] = None):
        self.values = values
        self.base_values = base_values  
        self.data = data
        self.feature_names = feature_names
    
    def __getitem__(self, key) -> 'Explanation':
        return Explanation(
            values=self.values[key],
            base_values=self.base_values[key] if self.base_values.ndim > 0 else self.base_values,
            data=self.data[key],
            feature_names=self.feature_names
        )
    
    def __len__(self) -> int:
        return len(self.values)
    
    @property
    def shape(self) -> tuple:
        return self.values.shape
    
    def abs(self) -> 'Explanation':
        return Explanation(
            values=np.abs(self.values),
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names
        )


def sample(X: np.ndarray, nsamples: int) -> np.ndarray:
    if len(X) <= nsamples:
        return X
    indices = np.random.choice(len(X), nsamples, replace=False)
    return X[indices]


def summary_plot(explanation: Explanation, max_display: int = 10) -> None:
    print("=== SHAP Summary ===")
    print(f"Shape: {explanation.shape}")
    print(f"Base value: {explanation.base_values[0]:.4f}")
    
    feature_importance = np.mean(np.abs(explanation.values), axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    print(f"\nTop {min(max_display, len(feature_importance))} features:")
    for i, idx in enumerate(sorted_idx[:max_display]):
        feature_name = (explanation.feature_names[idx] if explanation.feature_names 
                       else f"Feature {idx}")
        print(f"{i+1:2d}. {feature_name}: {feature_importance[idx]:.4f}")


def waterfall_plot(explanation: Explanation, instance_idx: int = 0) -> None:
    if instance_idx >= len(explanation):
        raise IndexError(f"Instance index {instance_idx} out of range")
    
    shap_vals = explanation.values[instance_idx]
    base_val = (explanation.base_values[instance_idx] if explanation.base_values.ndim > 0 
               else explanation.base_values)
    
    print(f"=== Waterfall Plot (Instance {instance_idx}) ===")
    print(f"Base value: {base_val:.4f}")
    
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    
    cumulative = base_val
    for idx in sorted_idx:
        if abs(shap_vals[idx]) > 1e-6:
            feature_name = (explanation.feature_names[idx] if explanation.feature_names 
                           else f"Feature {idx}")
            cumulative += shap_vals[idx]
            print(f"{feature_name}: {shap_vals[idx]:+.4f} → {cumulative:.4f}")
    
    prediction = base_val + np.sum(shap_vals)
    print(f"\nFinal prediction: {prediction:.4f}")
    print(f"SHAP values sum: {np.sum(shap_vals):.4f}")


ExactExplainer = Explainer
KernelExplainer = Explainer