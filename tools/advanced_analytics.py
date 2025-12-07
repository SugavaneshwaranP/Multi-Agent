import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedSalesAnalyzer:
    """Advanced analytics for Superstore dataset with ML capabilities"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.processed_data = None
        
    def load_and_preprocess(self):
        """Load and preprocess the dataset"""
        self.data = pd.read_csv(self.file_path, encoding='latin-1')
        
        # Convert dates
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        self.data['Ship Date'] = pd.to_datetime(self.data['Ship Date'])
        
        # Extract time features
        self.data['Year'] = self.data['Order Date'].dt.year
        self.data['Month'] = self.data['Order Date'].dt.month
        self.data['Quarter'] = self.data['Order Date'].dt.quarter
        self.data['DayOfWeek'] = self.data['Order Date'].dt.dayofweek
        self.data['WeekOfYear'] = self.data['Order Date'].dt.isocalendar().week
        
        # Calculate additional metrics
        self.data['Profit Margin'] = (self.data['Profit'] / self.data['Sales']) * 100
        self.data['Shipping Days'] = (self.data['Ship Date'] - self.data['Order Date']).dt.days
        self.data['Revenue per Quantity'] = self.data['Sales'] / self.data['Quantity']
        
        self.processed_data = self.data.copy()
        return self.data
    
    def get_basic_metrics(self):
        """Get comprehensive basic metrics"""
        return {
            'total_sales': float(self.data['Sales'].sum()),
            'total_profit': float(self.data['Profit'].sum()),
            'total_orders': int(self.data['Order ID'].nunique()),
            'total_customers': int(self.data['Customer ID'].nunique()),
            'total_products': int(self.data['Product ID'].nunique()),
            'avg_order_value': float(self.data.groupby('Order ID')['Sales'].sum().mean()),
            'avg_profit_margin': float(self.data['Profit Margin'].mean()),
            'total_quantity_sold': int(self.data['Quantity'].sum())
        }
    
    def time_series_analysis(self):
        """Analyze sales trends over time"""
        monthly_sales = self.data.groupby([self.data['Order Date'].dt.to_period('M')]).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        
        monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)
        
        return {
            'monthly_data': monthly_sales.to_dict('records'),
            'peak_month': monthly_sales.loc[monthly_sales['Sales'].idxmax(), 'Order Date'],
            'lowest_month': monthly_sales.loc[monthly_sales['Sales'].idxmin(), 'Order Date']
        }
    
    def sales_forecast(self, periods=6):
        """Predict future sales using linear regression"""
        monthly_sales = self.data.groupby(self.data['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly_sales['Month_Num'] = range(len(monthly_sales))
        
        X = monthly_sales[['Month_Num']].values
        y = monthly_sales['Sales'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next periods
        future_months = np.arange(len(monthly_sales), len(monthly_sales) + periods).reshape(-1, 1)
        predictions = model.predict(future_months)
        
        return {
            'forecasted_sales': [float(p) for p in predictions],
            'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
            'monthly_growth_rate': float(model.coef_[0])
        }
    
    def customer_segmentation(self, n_clusters=4):
        """Segment customers using K-Means clustering"""
        customer_features = self.data.groupby('Customer ID').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique',
            'Quantity': 'sum',
            'Discount': 'mean'
        }).reset_index()
        
        # Prepare features for clustering
        features = customer_features[['Sales', 'Profit', 'Order ID', 'Quantity', 'Discount']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_features['Segment'] = kmeans.fit_predict(scaled_features)
        
        # Analyze segments
        segment_analysis = customer_features.groupby('Segment').agg({
            'Sales': 'mean',
            'Profit': 'mean',
            'Order ID': 'mean',
            'Customer ID': 'count'
        }).round(2)
        
        # Label segments
        segment_labels = []
        for idx in range(n_clusters):
            seg_data = segment_analysis.loc[idx]
            if seg_data['Sales'] > segment_analysis['Sales'].mean() and seg_data['Profit'] > segment_analysis['Profit'].mean():
                label = 'VIP Customers'
            elif seg_data['Sales'] > segment_analysis['Sales'].mean():
                label = 'High Value'
            elif seg_data['Order ID'] > segment_analysis['Order ID'].mean():
                label = 'Frequent Buyers'
            else:
                label = 'Regular Customers'
            segment_labels.append(label)
        
        segment_analysis['Label'] = segment_labels
        
        return segment_analysis.to_dict('index')
    
    def product_performance_analysis(self):
        """Analyze product categories and sub-categories"""
        category_analysis = self.data.groupby('Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Order ID': 'nunique'
        }).round(2)
        
        category_analysis['Profit Margin %'] = (category_analysis['Profit'] / category_analysis['Sales'] * 100).round(2)
        
        # Top performing sub-categories
        subcategory_analysis = self.data.groupby('Sub-Category').agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).sort_values('Profit', ascending=False).head(10)
        
        # Bottom performing products
        bottom_products = self.data.groupby('Product Name').agg({
            'Profit': 'sum',
            'Sales': 'sum'
        }).sort_values('Profit').head(10)
        
        return {
            'category_performance': category_analysis.to_dict('index'),
            'top_subcategories': subcategory_analysis.to_dict('index'),
            'worst_products': bottom_products.to_dict('index')
        }
    
    def regional_intelligence(self):
        """Deep regional analysis"""
        regional_analysis = self.data.groupby('Region').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique',
            'Customer ID': 'nunique',
            'Discount': 'mean'
        }).round(2)
        
        regional_analysis.columns = ['_'.join(col).strip() for col in regional_analysis.columns.values]
        
        # State-level analysis
        state_analysis = self.data.groupby(['Region', 'State']).agg({
            'Sales': 'sum',
            'Profit': 'sum'
        }).sort_values('Sales', ascending=False).head(15)
        
        return {
            'regional_summary': regional_analysis.to_dict('index'),
            'top_states': state_analysis.to_dict('index')
        }
    
    def discount_impact_analysis(self):
        """Analyze the impact of discounts on profitability"""
        # Create discount bins
        self.data['Discount_Bin'] = pd.cut(self.data['Discount'], 
                                           bins=[0, 0.1, 0.2, 0.3, 1.0],
                                           labels=['0-10%', '10-20%', '20-30%', '30%+'])
        
        discount_impact = self.data.groupby('Discount_Bin').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count'
        }).round(2)
        
        discount_impact['Profit Margin %'] = (discount_impact['Profit'] / discount_impact['Sales'] * 100).round(2)
        
        # High discount items with losses
        high_discount_losses = self.data[(self.data['Discount'] > 0.2) & (self.data['Profit'] < 0)]
        total_loss = high_discount_losses['Profit'].sum()
        
        return {
            'discount_analysis': discount_impact.to_dict('index'),
            'high_discount_loss': float(total_loss),
            'loss_orders_count': len(high_discount_losses),
            'recommendation': 'Review discount strategy for items with >20% discount' if total_loss < -10000 else 'Discount strategy is healthy'
        }
    
    def shipping_efficiency_analysis(self):
        """Analyze shipping performance"""
        shipping_analysis = self.data.groupby('Ship Mode').agg({
            'Shipping Days': 'mean',
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'count'
        }).round(2)
        
        # Late shipments (>5 days)
        late_shipments = self.data[self.data['Shipping Days'] > 5]
        late_shipment_rate = (len(late_shipments) / len(self.data)) * 100
        
        return {
            'shipping_mode_performance': shipping_analysis.to_dict('index'),
            'late_shipment_rate': float(late_shipment_rate),
            'avg_shipping_days': float(self.data['Shipping Days'].mean())
        }
    
    def customer_lifetime_value(self):
        """Calculate CLV metrics"""
        customer_metrics = self.data.groupby('Customer ID').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique',
            'Order Date': ['min', 'max']
        })
        
        customer_metrics.columns = ['Total_Sales', 'Total_Profit', 'Order_Count', 'First_Order', 'Last_Order']
        customer_metrics['Customer_Lifetime_Days'] = (customer_metrics['Last_Order'] - customer_metrics['First_Order']).dt.days
        customer_metrics['Avg_Order_Value'] = customer_metrics['Total_Sales'] / customer_metrics['Order_Count']
        
        # Top customers
        top_customers = customer_metrics.nlargest(10, 'Total_Sales')[['Total_Sales', 'Total_Profit', 'Order_Count']]
        
        return {
            'avg_customer_lifetime_value': float(customer_metrics['Total_Sales'].mean()),
            'avg_orders_per_customer': float(customer_metrics['Order_Count'].mean()),
            'top_customers': top_customers.to_dict('index'),
            'one_time_customers': int((customer_metrics['Order_Count'] == 1).sum()),
            'repeat_customer_rate': float(((customer_metrics['Order_Count'] > 1).sum() / len(customer_metrics)) * 100)
        }
    
    def anomaly_detection(self):
        """Detect anomalies and outliers"""
        # Sales anomalies
        Q1 = self.data['Sales'].quantile(0.25)
        Q3 = self.data['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        
        sales_outliers = self.data[(self.data['Sales'] < (Q1 - 1.5 * IQR)) | 
                                   (self.data['Sales'] > (Q3 + 1.5 * IQR))]
        
        # Negative profit items
        negative_profit = self.data[self.data['Profit'] < 0]
        
        return {
            'sales_outliers_count': len(sales_outliers),
            'negative_profit_orders': len(negative_profit),
            'total_negative_profit': float(negative_profit['Profit'].sum()),
            'outlier_percentage': float((len(sales_outliers) / len(self.data)) * 100)
        }
    
    def generate_comprehensive_report(self):
        """Generate a complete analytics report"""
        self.load_and_preprocess()
        
        return {
            'basic_metrics': self.get_basic_metrics(),
            'time_series': self.time_series_analysis(),
            'forecast': self.sales_forecast(),
            'customer_segments': self.customer_segmentation(),
            'product_performance': self.product_performance_analysis(),
            'regional_intelligence': self.regional_intelligence(),
            'discount_impact': self.discount_impact_analysis(),
            'shipping_efficiency': self.shipping_efficiency_analysis(),
            'customer_lifetime_value': self.customer_lifetime_value(),
            'anomalies': self.anomaly_detection()
        }
