import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealtimeDataSimulator:
    """Simulate real-time data updates for the Superstore dataset"""
    
    def __init__(self, base_file_path):
        self.base_file_path = base_file_path
        self.base_data = pd.read_csv(base_file_path, encoding='latin-1')
        self.base_data['Order Date'] = pd.to_datetime(self.base_data['Order Date'])
        self.base_data['Ship Date'] = pd.to_datetime(self.base_data['Ship Date'])
        
    def generate_new_orders(self, num_orders=10):
        """Generate new simulated orders based on historical patterns"""
        new_orders = []
        
        # Get patterns from existing data
        categories = self.base_data['Category'].unique()
        segments = self.base_data['Segment'].unique()
        regions = self.base_data['Region'].unique()
        ship_modes = self.base_data['Ship Mode'].unique()
        
        for i in range(num_orders):
            # Random selection based on existing patterns
            category = np.random.choice(categories)
            segment = np.random.choice(segments)
            region = np.random.choice(regions)
            ship_mode = np.random.choice(ship_modes)
            
            # Get sub-category and product from selected category
            category_products = self.base_data[self.base_data['Category'] == category]
            sample_product = category_products.sample(1).iloc[0]
            
            # Generate realistic values
            quantity = np.random.randint(1, 10)
            base_price = sample_product['Sales'] / sample_product['Quantity']
            sales = base_price * quantity * np.random.uniform(0.8, 1.2)
            discount = np.random.choice([0, 0.1, 0.15, 0.2, 0.25], p=[0.5, 0.2, 0.15, 0.1, 0.05])
            profit = sales * np.random.uniform(0.1, 0.3) - (sales * discount)
            
            order_date = datetime.now() - timedelta(days=np.random.randint(0, 7))
            ship_date = order_date + timedelta(days=np.random.randint(2, 8))
            
            new_order = {
                'Row ID': len(self.base_data) + i + 1,
                'Order ID': f'RT-2025-{np.random.randint(100000, 999999)}',
                'Order Date': order_date.strftime('%m/%d/%Y'),
                'Ship Date': ship_date.strftime('%m/%d/%Y'),
                'Ship Mode': ship_mode,
                'Customer ID': f'CU-{np.random.randint(10000, 99999)}',
                'Customer Name': f'Customer {np.random.randint(1000, 9999)}',
                'Segment': segment,
                'Country': 'United States',
                'City': sample_product['City'],
                'State': sample_product['State'],
                'Postal Code': sample_product['Postal Code'],
                'Region': region,
                'Product ID': sample_product['Product ID'],
                'Category': category,
                'Sub-Category': sample_product['Sub-Category'],
                'Product Name': sample_product['Product Name'],
                'Sales': round(sales, 2),
                'Quantity': quantity,
                'Discount': discount,
                'Profit': round(profit, 2)
            }
            new_orders.append(new_order)
        
        return pd.DataFrame(new_orders)
    
    def get_realtime_metrics(self, window_hours=24):
        """Get metrics for recent time window"""
        # Simulate by using recent data from base dataset
        cutoff_date = self.base_data['Order Date'].max() - timedelta(hours=window_hours)
        recent_data = self.base_data[self.base_data['Order Date'] > cutoff_date]
        
        return {
            'recent_orders': len(recent_data),
            'recent_sales': float(recent_data['Sales'].sum()),
            'recent_profit': float(recent_data['Profit'].sum()),
            'avg_order_value': float(recent_data['Sales'].mean()) if len(recent_data) > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_realtime_alerts(self):
        """Detect issues requiring immediate attention"""
        alerts = []
        
        # Check for high discount items
        high_discount = self.base_data[self.base_data['Discount'] > 0.3]
        if len(high_discount) > 0:
            alerts.append({
                'type': 'HIGH_DISCOUNT',
                'severity': 'WARNING',
                'message': f'{len(high_discount)} orders with >30% discount',
                'count': len(high_discount)
            })
        
        # Check for negative profit
        negative_profit = self.base_data[self.base_data['Profit'] < 0]
        recent_negative = negative_profit.nlargest(10, 'Order Date')
        if len(negative_profit) > 100:
            alerts.append({
                'type': 'NEGATIVE_PROFIT',
                'severity': 'CRITICAL',
                'message': f'{len(negative_profit)} loss-making orders detected',
                'total_loss': float(negative_profit['Profit'].sum())
            })
        
        # Check for inventory alerts (low quantity)
        low_qty = self.base_data[self.base_data['Quantity'] == 1]
        if len(low_qty) > len(self.base_data) * 0.6:
            alerts.append({
                'type': 'LOW_QUANTITY',
                'severity': 'INFO',
                'message': 'High number of single-unit orders - possible stock issues'
            })
        
        return alerts
    
    def get_live_dashboard_data(self):
        """Get data formatted for live dashboard"""
        recent_7days = self.base_data[
            self.base_data['Order Date'] > (self.base_data['Order Date'].max() - timedelta(days=7))
        ]
        
        daily_stats = recent_7days.groupby(recent_7days['Order Date'].dt.date).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        
        return {
            'daily_trends': daily_stats.to_dict('records'),
            'total_today': float(recent_7days[recent_7days['Order Date'].dt.date == recent_7days['Order Date'].max().date()]['Sales'].sum()),
            'alerts': self.detect_realtime_alerts(),
            'last_updated': datetime.now().isoformat()
        }
