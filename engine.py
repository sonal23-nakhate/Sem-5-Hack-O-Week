import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Linear Algebra Helpers
def dot(vec_a, vec_b):
    return sum((a * b for a, b in zip(vec_a, vec_b)), Value(0))

def matvec_mul(matrix, vec):
    return [dot(row, vec) for row in matrix]

def power_iteration(matrix, num_simulations: int = 20):
    """
    Computes the dominant eigenvalue and eigenvector using Power Iteration.
    A * v = lambda * v
    """
    # Start with an initial guess vector [1.0, 1.0]
    b_k = [1.0, 1.0]
    
    for _ in range(num_simulations):
        # Multiply matrix by vector: A * v
        y = [
            matrix[0][0] * b_k[0] + matrix[0][1] * b_k[1],
            matrix[1][0] * b_k[0] + matrix[1][1] * b_k[1]
        ]
        # Compute magnitude (Euclidean norm)
        norm = math.sqrt(y[0]**2 + y[1]**2)
        
        # Normalize vector
        b_k = [y[0] / norm, y[1] / norm]
        
    # Rayleigh quotient for dominant eigenvalue: (v^T * A * v) / (v^T * v)
    y = [
        matrix[0][0] * b_k[0] + matrix[0][1] * b_k[1],
        matrix[1][0] * b_k[0] + matrix[1][1] * b_k[1]
    ]
    eigenvalue = b_k[0] * y[0] + b_k[1] * y[1]
    
    return eigenvalue, b_k