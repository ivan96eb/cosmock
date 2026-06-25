import numpy as np
import pytest

from cosmock.transforms import evaluate_transform


def test_evaluate_order2_transform_matches_formula():
    x = np.array([-1.0, 0.0, 1.0])
    alpha = 0.4
    beta = 1.7

    actual = evaluate_transform(x, 2, [alpha, beta])
    expected = beta * np.exp(alpha * x - 0.5 * alpha**2) - beta

    np.testing.assert_allclose(actual, expected)


def test_evaluate_order3_transform_matches_formula():
    x = np.array([-1.0, 0.0, 1.0])
    a, b, c = 0.3, 0.2, 0.8

    actual = evaluate_transform(x, 3, [a, b, c])
    expected = (np.exp(a * x - 0.5 * a**2) + b * x + c) / (1.0 + c) - 1.0

    np.testing.assert_allclose(actual, expected)


def test_evaluate_transform_rejects_unsupported_order():
    with pytest.raises(ValueError, match="order"):
        evaluate_transform([0.0], 4, [1.0, 1.0, 1.0, 1.0])
