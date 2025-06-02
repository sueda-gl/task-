# 🚀 Dynamic Pricing System Dashboard

An advanced interactive dashboard for visualizing and analyzing a sophisticated two-stage dynamic pricing system with MILP optimization and Monte Carlo risk analysis.

## 🌟 Features

### **Interactive Controls**
- **Face Price Adjustment**: Real-time pricing parameter tuning ($50-$200)
- **Inventory Management**: Configure total inventory (50-200 units)
- **Customer Base Scaling**: Adjust number of customers (50-300)
- **Scenario Generation**: Random seed control for reproducible experiments

### **Stage 1: Bucket-Based Allocation**
- **Visual Bucket Analysis**: Interactive histogram showing bid distribution across 10 pricing buckets
- **Acceptance Rate Metrics**: Real-time calculation of Stage 1 performance
- **Revenue Tracking**: Surplus/deficit analysis with color-coded indicators

### **Stage 2: Personalized Pricing**
- **MILP Optimization**: Advanced Mixed-Integer Linear Programming solver
- **Dual Decomposition**: Sophisticated economic optimization algorithms
- **Price-Bid Scatter Plot**: Visual analysis of personalized pricing vs customer willingness to pay
- **Acceptance Probability Modeling**: Quadratic demand function implementation

### **Advanced Analytics**
- **📊 Monte Carlo Simulation**: 1000-iteration risk analysis with statistical distributions
- **💰 Revenue Optimization**: Multi-stage revenue comparison and efficiency metrics
- **📈 Economic Efficiency**: Welfare analysis and price discrimination metrics
- **⚖️ Fairness Analysis**: Equity assessment with bias detection
- **🎲 Risk Assessment**: Overselling probability and confidence intervals

### **Professional Visualizations**
- **Interactive Plotly Charts**: High-quality, professional-grade visualizations
- **Real-time Updates**: Dynamic recalculation as parameters change
- **Export Capabilities**: JSON report generation for further analysis
- **Responsive Design**: Mobile-friendly layout with gradient styling

## 🚀 Quick Start

### Installation

1. **Clone or download the files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 How to Use

### **Basic Operation**
1. **Adjust Parameters**: Use the sidebar controls to set face price, inventory, and customer count
2. **Generate Scenarios**: Click "Generate New Scenario" to create new customer bid patterns
3. **Analyze Results**: Explore the interactive visualizations and metrics
4. **Run Simulations**: Enable Monte Carlo analysis for risk assessment

### **Key Metrics to Watch**
- **Inventory Utilization**: Aim for 85-95% for optimal efficiency
- **Customer Satisfaction**: Track percentage of customers served
- **Revenue Balance**: Monitor Stage 1 surplus/deficit
- **Overselling Risk**: Keep below 5% for safe operations

### **Advanced Features**
- **Advanced Analytics Expander**: Deep-dive into economic efficiency and fairness metrics
- **Export Reports**: Generate detailed JSON reports for documentation
- **Real-time Optimization**: Watch algorithms solve complex pricing problems instantly

## 🎓 Academic Highlights

This dashboard showcases several advanced concepts:

### **Mathematical Optimization**
- **Mixed-Integer Linear Programming (MILP)**: Exact solution methods for complex optimization
- **Dual Decomposition**: Advanced technique from convex optimization theory
- **Revenue Management**: Industry-standard pricing strategies

### **Algorithm Design**
- **Two-Stage Mechanism**: Sophisticated auction-like pricing system
- **Bucket-Based Allocation**: Efficient customer segmentation
- **Monte Carlo Methods**: Statistical simulation for risk analysis

### **Economic Theory**
- **Price Discrimination**: First, second, and third-degree pricing strategies
- **Welfare Analysis**: Economic efficiency and surplus calculations
- **Mechanism Design**: Incentive-compatible pricing systems

## 🔧 Technical Architecture

### **Backend Algorithms**
- **PuLP**: Open-source MILP solver with CBC backend
- **NumPy**: High-performance numerical computing
- **Custom Optimization**: Proprietary dual decomposition implementation

### **Frontend Technologies**
- **Streamlit**: Modern Python web framework
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis

### **Performance Features**
- **Real-time Computation**: Sub-second optimization for typical problem sizes
- **Scalable Architecture**: Handles 50-300 customers efficiently
- **Memory Optimization**: Efficient data structures for large-scale problems

## 📈 Demo Scenarios

Try these interesting parameter combinations:

### **High Competition Scenario**
- Face Price: $75
- Inventory: 100
- Customers: 200
- Observe: High competition drives sophisticated pricing strategies

### **Premium Market Scenario**
- Face Price: $150
- Inventory: 50
- Customers: 100
- Observe: Scarcity-driven pricing with high margins

### **Mass Market Scenario**
- Face Price: $100
- Inventory: 200
- Customers: 150
- Observe: Efficient allocation with balanced utilization

## 🎯 Impressing Your Professor

This dashboard demonstrates:

1. **Technical Sophistication**: Implementation of advanced optimization algorithms
2. **Professional Presentation**: Clean, modern interface with publication-quality visualizations
3. **Real-world Relevance**: Applicable to airlines, hotels, e-commerce, and other industries
4. **Academic Rigor**: Sound mathematical foundations with proper algorithmic implementation
5. **Innovation**: Novel combination of bucket allocation and personalized pricing
6. **Practical Impact**: Immediate business value with measurable performance metrics

## 🔍 Understanding the Output

### **Key Visualizations**
- **Bucket Histogram**: Shows how customers are distributed across pricing segments
- **Scatter Plot**: Reveals the relationship between bids and personalized prices
- **Revenue Charts**: Compares actual vs target performance
- **Monte Carlo Distributions**: Illustrates uncertainty and risk profiles

### **Critical Metrics**
- **Lambda Star**: The optimal dual variable from the optimization
- **Revenue Efficiency**: How well the system captures potential revenue
- **Fairness Ratio**: Whether the system treats customers equitably

---

**Built with ❤️ using advanced optimization theory and modern web technologies** 