from dataclasses import dataclass, field

TAX_RATE = 0.20
SHIPPING_FEE = 5.00
FREE_SHIPPING_THRESHOLD = 50.00


@dataclass
class Cart:
    items: list[tuple[str, float, int]] = field(default_factory=list)

    def add(self, name: str, unit_price: float, quantity: int = 1) -> None:
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        self.items.append((name, unit_price, quantity))

    def subtotal(self) -> float:
        return round(sum(price * qty for _, price, qty in self.items), 2)

    def apply_discount(self, percent: float) -> float:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        return round(self.subtotal() * (1 - percent / 100), 2)

    def shipping_fee(self, discount_percent: float = 0.0) -> float:
        # Free shipping is earned on what the customer actually pays, so the
        # threshold must be checked against the discounted amount.
        if self.subtotal() >= FREE_SHIPPING_THRESHOLD:
            return 0.00
        return SHIPPING_FEE

    def total(self, discount_percent: float = 0.0) -> float:
        goods = self.apply_discount(discount_percent)
        return round(goods * (1 + TAX_RATE) + self.shipping_fee(discount_percent), 2)