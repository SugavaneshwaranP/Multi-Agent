import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_sales_trend_chart(data):
    """Create interactive sales trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines+markers',
        name='Sales',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8, color='#60a5fa'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title='Sales Trend Analysis',
        xaxis_title='Period',
        yaxis_title='Sales ($)',
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_customer_segment_chart(segments):
    """Create customer segmentation visualization"""
    labels = [seg['Label'] for seg in segments.values()]
    values = [seg['Customer ID'] for seg in segments.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']),
        textinfo='label+percent',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title='Customer Segmentation',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_category_performance_chart(data, value_col=None):
    """Create category performance chart - dynamic"""
    if not data or len(data) == 0:
        # Return empty chart
        fig = go.Figure()
        fig.update_layout(
            title='No categorical data available',
            template='plotly_dark',
            height=400
        )
        return fig
    
    categories = list(data.keys())
    
    # Get first numeric column from data
    first_cat_data = data[categories[0]]
    numeric_cols = [k for k, v in first_cat_data.items() if isinstance(v, (int, float))]
    
    if len(numeric_cols) == 0:
        fig = go.Figure()
        fig.update_layout(title='No numeric data', template='plotly_dark', height=400)
        return fig
    
    # Use first two numeric columns or duplicate if only one
    col1 = numeric_cols[0]
    col2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    
    values1 = [data[cat].get(col1, 0) for cat in categories]
    values2 = [data[cat].get(col2, 0) for cat in categories]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{col1} by Category', f'{col2} by Category'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=values1, name=col1, marker_color='#3b82f6'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=categories, y=values2, name=col2, marker_color='#10b981'),
        row=1, col=2
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def create_skills_chart(skills_data):
    """Create skills frequency chart"""
    if not skills_data or not skills_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No skills data', template='plotly_dark', height=400)
        return fig
    
    tech_skills = skills_data.get('top_technical_skills', [])
    soft_skills = skills_data.get('top_soft_skills', [])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Top Technical Skills', 'Top Soft Skills'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    if tech_skills:
        skills, counts = zip(*tech_skills)
        fig.add_trace(
            go.Bar(x=list(counts), y=list(skills), orientation='h', marker_color='#3b82f6', name='Technical'),
            row=1, col=1
        )
    
    if soft_skills:
        skills, counts = zip(*soft_skills)
        fig.add_trace(
            go.Bar(x=list(counts), y=list(skills), orientation='h', marker_color='#10b981', name='Soft Skills'),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Skills Analysis',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    return fig

def create_experience_chart(exp_data):
    """Create experience distribution chart"""
    if not exp_data or not exp_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No experience data', template='plotly_dark', height=400)
        return fig
    
    distribution = exp_data.get('distribution', {})
    categories = list(distribution.keys())
    values = list(distribution.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        hole=0.4,
        marker=dict(colors=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']),
        textinfo='label+value',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title=f'Experience Distribution (Avg: {exp_data.get("average_experience", 0):.1f} years)',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_education_chart(edu_data):
    """Create education distribution chart"""
    if not edu_data or not edu_data.get('available'):
        fig = go.Figure()
        fig.update_layout(title='No education data', template='plotly_dark', height=400)
        return fig
    
    distribution = edu_data.get('distribution', {})
    categories = list(distribution.keys())
    values = list(distribution.values())
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker=dict(color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444']),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Education Qualifications',
        template='plotly_dark',
        height=400,
        xaxis_title='Qualification',
        yaxis_title='Number of Candidates'
    )
    
    return fig

def create_category_distribution_chart(category_data):
    """Create resume category distribution chart"""
    if not category_data or len(category_data) == 0:
        fig = go.Figure()
        fig.update_layout(title='No category data', template='plotly_dark', height=400)
        return fig
    
    categories = list(category_data.keys())[:15]  # Top 15
    values = [category_data[cat] for cat in categories]
    
    fig = go.Figure(data=[go.Bar(
        y=categories,
        x=values,
        orientation='h',
        marker=dict(color=values, colorscale='Viridis'),
        text=values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Resume Categories Distribution',
        template='plotly_dark',
        height=500,
        xaxis_title='Number of Resumes',
        yaxis_title='Category'
    )
    
    return fig
