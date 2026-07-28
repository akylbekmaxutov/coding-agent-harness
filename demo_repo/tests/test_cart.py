import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cart import Cart


def make_cart() -> Cart:
    cart = Cart()
    cart.add("book", 12.00, 4)
    cart.add("pen", 2.00, 2)
    return cart


def test_subtotal():
    assert make_cart().subtotal() == 52.00


def test_apply_discount():
    assert make_cart().apply_discount(10) == 46.80


def test_rejects_bad_quantity():
    with pytest.raises(ValueError):
        Cart().add("book", 10.00, 0)


def test_free_shipping_above_threshold():
    assert make_cart().shipping_fee() == 0.00


def test_discount_can_lose_free_shipping():
    # 52.00 - 10% = 46.80, which is below the 50.00 threshold, so shipping
    # is charged: 46.80 * 1.20 + 5.00 = 61.16
    assert make_cart().total(discount_percent=10) == 61.16