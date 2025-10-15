import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of rows
n_orders = 20000
n_customers = 3000
n_products = 1000

# Generate customers
customers = pd.DataFrame({
    "customer_id": [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
    "join_date": pd.to_datetime(np.random.choice(pd.date_range("2020-01-01", "2024-12-31"), n_customers)),
    "age": np.random.randint(18, 65, n_customers),
    "gender": np.random.choice(["Male", "Female", "Other"], n_customers, p=[0.48, 0.48, 0.04]),
    "location": np.random.choice(["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata"], n_customers),
    "preferred_payment": np.random.choice(["UPI", "CreditCard", "DebitCard", "COD", "Wallet"], n_customers),
    "total_orders": np.random.randint(1, 100, n_customers),
    "total_returns": np.random.randint(0, 20, n_customers),
    "avg_order_value": np.round(np.random.uniform(300, 3000, n_customers), 2)
})

# Generate products
categories = ["Apparel", "Electronics", "Home", "Beauty", "Sports"]
subcategories = {
    "Apparel": ["Shirts", "Shoes", "Jeans", "Dresses"],
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Cameras"],
    "Home": ["Furniture", "Kitchen", "Decor", "Appliances"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance"],
    "Sports": ["Fitness", "Outdoor", "TeamSports", "Accessories"]
}
brands = ["Nike", "Adidas", "Samsung", "Apple", "Sony", "LG", "Levis", "Puma", "L'Oreal", "Philips"]

products = []
for i in range(1, n_products + 1):
    cat = random.choice(categories)
    subcat = random.choice(subcategories[cat])
    brand = random.choice(brands)
    products.append([
        f"P{str(i).zfill(5)}",
        f"{brand} {subcat}",
        cat,
        subcat,
        brand,
        round(np.random.uniform(3.0, 5.0), 1),
        round(np.random.uniform(0.05, 0.3), 2)
    ])

products = pd.DataFrame(products, columns=["product_id", "product_name", "product_category",
                                           "product_subcategory", "brand", "avg_rating", "return_rate_category"])

# Generate orders
order_dates = pd.date_range("2023-01-01", "2025-09-01", freq="h")
orders = []
for i in range(1, n_orders + 1):
    cust = customers.sample(1).iloc[0]
    prod = products.sample(1).iloc[0]
    order_date = random.choice(order_dates)
    delivery_days = random.randint(2, 10)
    delivery_date = order_date + timedelta(days=delivery_days)
    price = round(np.random.uniform(300, 5000), 2)
    discount = round(np.random.uniform(0, 50), 2)
    quantity = np.random.randint(1, 5)
    orders.append([
        f"O{str(i).zfill(6)}",
        cust.customer_id,
        prod.product_id,
        order_date,
        delivery_date,
        random.choice(["CreditCard", "COD", "UPI", "DebitCard", "Wallet"]),
        price,
        discount,
        quantity,
        random.choice(["North", "South", "East", "West"]),
        delivery_days,
        round(np.random.uniform(1.0, 5.0), 1),
        prod.product_category,
        prod.product_subcategory,
        prod.brand,
        np.random.randint(0, 500),
        np.random.choice([True, False], p=[0.1, 0.9]),
        random.choice(["Mobile", "Desktop", "Tablet"]),
        random.choice(["CampaignA", "CampaignB", "Organic", "SocialMedia"])
    ])

orders = pd.DataFrame(orders, columns=[
    "order_id", "customer_id", "product_id", "order_date", "delivery_date", "payment_type",
    "price", "discount_percent", "quantity", "shipping_region", "expected_delivery_days",
    "product_rating", "product_category", "product_subcategory", "brand", "loyalty_points",
    "is_first_order", "device_type", "referral"
])

# Generate returns (~20% return rate)
returns = []
for i, row in orders.sample(frac=0.2, random_state=42).iterrows():
    return_date = row.delivery_date + timedelta(days=np.random.randint(1, 15))
    returns.append([
        f"R{str(i).zfill(6)}",
        row.order_id,
        return_date,
        random.choice(["Size Issue", "Defective", "Changed Mind", "Wrong Item", "Other"]),
        row.price * row.quantity,
        np.random.randint(1, 7)
    ])

returns = pd.DataFrame(returns, columns=["return_id", "order_id", "return_date",
                                         "return_reason", "refund_amount", "pickup_delay_days"])

# Save datasets
orders.to_csv("./data/orders.csv", index=False)
returns.to_csv("./data/returns.csv", index=False)
customers.to_csv("./data/customers.csv", index=False)
products.to_csv("./data/products.csv", index=False)

