from dataclasses import dataclass
from typing import Optional, Set, List  # Import List explicitly
import numpy as np
from numpy.typing import NDArray
import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image

@dataclass
class Split:
    feature: int
    threshold: np.float64
    left: Set[NDArray[np.float64]]  # X and y of left subtree
    right: Set[NDArray[np.float64]]  # X and y of right subtree

@dataclass
class TreeNode:
    value: Optional[np.float64] = None
    feature: Optional[np.float64] = None
    threshold: Optional[np.float64] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

class DecisionTreeRegressor:
    def __init__(self, max_depth: int = 5, min_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_split = min_split
        self.tree = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        assert len(X.shape) > 1
        self.tree = self._build_tree(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([self._predict(inputs, self.tree) for inputs in X])
    
    def visualize_tree(self, feature_names: Optional[List[str]] = None) -> None:
        dot_data = export_graphviz(self.tree, out_file=None, feature_names=feature_names, filled=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        Image(graph.create_png())

    def _mse(self, y: NDArray[np.float64]) -> float: 
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _weighted_mse(self, left_y: NDArray[np.float64], right_y: NDArray[np.float64]) -> float:
        n = len(left_y) + len(right_y)
        left_weight = len(left_y) / n
        right_weight = len(right_y) / n
        return left_weight * self._mse(left_y) + right_weight * self._mse(right_y)
    
    def _find_best_split(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Optional[Split]:
        best_mse = float('inf')
        best_split = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) < self.min_split or len(right_y) < self.min_split:
                    continue

                current_mse = self._weighted_mse(left_y, right_y)
                if current_mse < best_mse:
                    best_mse = current_mse
                    best_split = Split(feature, threshold, (X[left_mask], left_y), (X[right_mask], right_y))
        return best_split

    def _build_tree(self, X: NDArray[np.float64], y: NDArray[np.float64], depth: int = 0) -> TreeNode:
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_split:
            return TreeNode(value=np.mean(y))

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return TreeNode(value=np.mean(y))

        left_subtree = self._build_tree(best_split.left[0], best_split.left[1], depth + 1)
        right_subtree = self._build_tree(best_split.right[0], best_split.right[1], depth + 1)
        return TreeNode(
            feature=best_split.feature,
            threshold=best_split.threshold,
            left=left_subtree,
            right=right_subtree
        )

    def _predict(self, inputs: NDArray[np.float64], node: TreeNode) -> float:
        if node.value is not None:
            return node.value
        if inputs[node.feature] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
