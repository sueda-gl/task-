import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dynamic_pricing_system import stage1_allocation, stage2_pricing, simulate_stage2_outcomes
import io

# Page configuration
st.set_page_config(
    page_title="Dynamic Pricing System Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stage-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

def generate_customer_bids(num_customers, face_price, seed=42):
    """Generate *uniform* bids across the 10 legal buckets (CLI-style).

    Each customer is first assigned a bucket 0…9 with equal probability,
    then a bid is drawn uniformly inside that bucket's strict open interval
    (0.51P–0.60P, 0.61P–0.70P, … , 1.41P–1.50P).
    This mirrors the helper used by `pricing_cli.py -r` so the dashboard and
    console runs become directly comparable.
    """

    rng = np.random.default_rng(seed)

    # Bucket bounds (strict open intervals)
    lower = np.array([0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41])
    upper = np.array([0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50])

    gap_ratio = 0.005  # keep 0.5 % of P away from edges to avoid boundary issues

    bids = []
    for cid in range(num_customers):
        idx = rng.integers(0, 10)  # choose bucket uniformly
        lo = lower[idx] + gap_ratio
        hi = upper[idx] - gap_ratio
        ratio = rng.uniform(lo, hi)
        bids.append((cid, float(ratio * face_price)))

    return bids

def create_bucket_visualization(bids, face_price, accepted_stage1):
    """Create visualization of Stage 1 bucket allocation"""
    
    # Calculate bid ratios
    ratios = [bid / face_price for _, bid in bids]
    accepted_ids = {cid for cid, _ in accepted_stage1}
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': [cid for cid, _ in bids],
        'bid_ratio': ratios,
        'bid_value': [bid for _, bid in bids],
        'status': ['Accepted' if cid in accepted_ids else 'Rejected' 
                  for cid, _ in bids]
    })
    
    # Create histogram with bucket boundaries
    fig = go.Figure()
    
    # Add histogram for all bids
    fig.add_trace(go.Histogram(
        x=df['bid_ratio'],
        nbinsx=50,
        name='All Bids',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Add histogram for accepted bids
    accepted_df = df[df['status'] == 'Accepted']
    fig.add_trace(go.Histogram(
        x=accepted_df['bid_ratio'],
        nbinsx=50,
        name='Accepted (Stage 1)',
        opacity=0.8,
        marker_color='green'
    ))
    
    # Add bucket boundaries
    bucket_boundaries = [0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.50]
    for boundary in bucket_boundaries:
        fig.add_vline(x=boundary, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title="Stage 1: Bucket-Based Allocation",
        xaxis_title="Bid Ratio (Bid / Face Price)",
        yaxis_title="Number of Customers",
        barmode='overlay',
        height=400
    )
    
    return fig

def create_price_distribution_plot(remaining, prices_stage2):
    """Create price distribution visualization for Stage 2"""
    
    if not prices_stage2:
        return go.Figure().add_annotation(text="No Stage 2 customers", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Create DataFrame
    price_dict = {cid: price for cid, price in prices_stage2}
    
    df = pd.DataFrame({
        'customer_id': [cid for cid, _ in remaining],
        'bid_value': [bid for _, bid in remaining],
        'personalized_price': [price_dict[cid] for cid, _ in remaining]
    })
    
    df['price_ratio'] = df['personalized_price'] / df['bid_value']
    df['will_accept'] = df['personalized_price'] <= df['bid_value']
    
    # Create scatter plot
    fig = go.Figure()
    
    # Accepted customers
    accepted = df[df['will_accept']]
    fig.add_trace(go.Scatter(
        x=accepted['bid_value'],
        y=accepted['personalized_price'],
        mode='markers',
        name='Will Accept',
        marker=dict(color='green', size=8),
        text=accepted['customer_id'],
        hovertemplate='Customer %{text}<br>Bid: $%{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Rejected customers
    rejected = df[~df['will_accept']]
    fig.add_trace(go.Scatter(
        x=rejected['bid_value'],
        y=rejected['personalized_price'],
        mode='markers',
        name='Will Reject',
        marker=dict(color='red', size=8),
        text=rejected['customer_id'],
        hovertemplate='Customer %{text}<br>Bid: $%{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add diagonal line (price = bid)
    max_val = max(df['bid_value'].max(), df['personalized_price'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Price = Bid',
        line=dict(color='black', dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Stage 2: Personalized Pricing vs. Customer Bids",
        xaxis_title="Customer Bid Value ($)",
        yaxis_title="Personalized Price ($)",
        height=400
    )
    
    return fig

def create_revenue_analysis(accepted_stage1, remaining, prices_stage2, face_price):
    """Create revenue analysis visualization"""
    
    # Calculate Stage 1 revenue
    stage1_revenue = sum(bid for _, bid in accepted_stage1)
    stage1_units = len(accepted_stage1)
    
    # Calculate Stage 2 expected revenue
    stage2_revenue = 0.0
    stage2_units = 0.0
    
    if prices_stage2:
        price_dict = {cid: price for cid, price in prices_stage2}
        
        for cid, bid in remaining:
            price = price_dict[cid]
            if price <= bid:
                f_i = 1.0
            elif price <= 2 * bid:
                f_i = ((2 * bid - price) / bid) ** 2
            else:
                f_i = 0.0
            
            stage2_revenue += price * f_i
            stage2_units += f_i
    
    # Create comparison chart
    data = {
        'Stage': ['Stage 1 (Actual)', 'Stage 2 (Expected)', 'Face Price Target'],
        'Revenue': [stage1_revenue, stage2_revenue, face_price * (stage1_units + stage2_units)],
        'Units': [stage1_units, stage2_units, stage1_units + stage2_units]
    }
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue Comparison', 'Units Sold'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue chart
    fig.add_trace(
        go.Bar(name='Revenue', x=df['Stage'], y=df['Revenue'], 
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']),
        row=1, col=1
    )
    
    # Units chart
    fig.add_trace(
        go.Bar(name='Units', x=df['Stage'], y=df['Units'],
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'], showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Revenue and Sales Analysis")
    fig.update_xaxes(title_text="Stage", row=1, col=1)
    fig.update_xaxes(title_text="Stage", row=1, col=2)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(title_text="Units Sold", row=1, col=2)
    
    return fig

def run_monte_carlo_simulation(remaining, prices_stage2, face_price, stage1_units, total_inventory):
    """Run Monte Carlo simulation and create visualization"""
    
    if not prices_stage2:
        return None, {}
    
    # Run simulation
    sim_results = simulate_stage2_outcomes(
        remaining, prices_stage2, face_price, stage1_units, total_inventory, 
        n_rep=1000, seed=42
    )
    
    # Create visualization
    np.random.seed(42)
    price_dict = {cid: p for cid, p in prices_stage2}
    bids = np.array([bid for _, bid in remaining], dtype=float)
    prices = np.array([price_dict[cid] for cid, _ in remaining], dtype=float)
    
    prob = np.where(
        prices <= bids,
        1.0,
        np.where(prices <= 2 * bids, ((2 * bids - prices) / bids) ** 2, 0.0),
    )
    
    units_sold = []
    revenues = []
    
    for _ in range(1000):
        accept = np.random.random(prob.size) < prob
        sold_stage2 = accept.sum()
        
        # Apply inventory constraint
        if stage1_units + sold_stage2 > total_inventory:
            accept_indices = np.flatnonzero(accept)
            np.random.shuffle(accept_indices)
            accept[accept_indices[total_inventory - stage1_units:]] = False
            sold_stage2 = accept.sum()
        
        units_sold.append(stage1_units + sold_stage2)
        revenues.append(np.sum(prices[accept]))
    
    # Create histogram
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Units Sold Distribution', 'Revenue Distribution')
    )
    
    fig.add_trace(
        go.Histogram(x=units_sold, name='Units Sold', nbinsx=30, marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=revenues, name='Revenue', nbinsx=30, marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Add mean lines
    fig.add_vline(x=np.mean(units_sold), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=np.mean(revenues), line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(height=400, title_text="Monte Carlo Simulation Results (1000 iterations)")
    fig.update_xaxes(title_text="Units Sold", row=1, col=1)
    fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
    
    return fig, sim_results

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Dynamic Pricing System Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Two-Stage Pricing with MILP Optimization & Monte Carlo Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ System Parameters")

    uploaded_file = st.sidebar.file_uploader("Upload your bids file (CSV or Excel)", type=["csv", "xlsx"])
    user_bids = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        # Assume the bids are in the second column or named 'Initial Bid'
        if 'Initial Bid' in df.columns:
            user_bids = df['Initial Bid'].tolist()
        else:
            user_bids = df.iloc[:, 1].tolist()  # fallback: use second column

    use_fixed_bids = st.sidebar.checkbox("Use Professor's Test Bids", value=False)

    # Fixed input bids from user
    fixed_bids = [
        151, 97, 70, 124, 81, 140, 158, 165, 155, 162, 144, 168, 59, 73, 72, 132, 145, 85, 117, 74, 95, 126, 163, 125, 143, 65, 89, 98, 68, 105, 144, 160, 139, 97, 94, 100, 153, 128, 164, 122, 148, 88, 138, 149, 130, 95, 62, 134, 121, 142, 94, 144, 132, 75, 75, 141, 68, 110, 83, 82, 93, 130, 132, 170, 161, 150, 134, 96, 118, 115, 117, 129, 163, 76, 100, 165, 153, 116, 125, 81, 86, 121, 61, 77, 130, 159, 73, 104, 101, 170, 66
    ]

    if user_bids is not None:
        # Add customer count selection
        max_customers = len(user_bids)
        num_customers = st.sidebar.number_input("Number of Customers", min_value=1, max_value=max_customers, value=max_customers, step=1)
        bids = [(i, float(b)) for i, b in enumerate(user_bids[:num_customers])]
        # face_price, total_inventory, seed remain as user-selectable below
        face_price = st.sidebar.number_input("Face Price ($)", min_value=50, max_value=200, value=100, step=1)
        total_inventory = st.sidebar.number_input("Total Inventory", min_value=50, max_value=200, value=100, step=10)
        st.sidebar.markdown("## 🎲 Simulation Settings")
        seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=100, value=42, step=1)
    elif use_fixed_bids:
        face_price = 116
        total_inventory = 60
        bids = [(i, float(b)) for i, b in enumerate(fixed_bids)]
        num_customers = len(bids)
        # Show parameters as static text
        st.sidebar.markdown(f"**Face Price:** {face_price}")
        st.sidebar.markdown(f"**Total Inventory:** {total_inventory}")
        st.sidebar.markdown(f"**Number of Customers:** {num_customers}")
        seed = 42  # not used, but keep for compatibility
    else:
        face_price = st.sidebar.number_input("Face Price ($)", min_value=50, max_value=200, value=100, step=1)
        total_inventory = st.sidebar.number_input("Total Inventory", min_value=50, max_value=200, value=100, step=10)
        num_customers = st.sidebar.number_input("Number of Customers", min_value=50, max_value=300, value=150, step=1)
        st.sidebar.markdown("## 🎲 Simulation Settings")
        seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=100, value=42, step=1)
        bids = generate_customer_bids(num_customers, face_price, seed)
    
    st.sidebar.markdown("## ⚙️ Boundary Handling")
    boundary_mode = st.sidebar.radio(
        "Bucket boundaries:",
        options=["Strict gaps", "Inclusive"],
        index=0,
        key="boundary_mode",
    )
    strict_flag = boundary_mode == "Strict gaps"
    
    st.sidebar.markdown("## ⚙️ Optimiser Priority")
    priority_mode = st.sidebar.radio(
        "When perfect balance is impossible, optimise for…",
        options=["Max units", "Min gap"],
        index=0,
        key="priority_mode",
    )
    priority_flag = "units" if priority_mode == "Max units" else "gap"
    
    # Generate data
    with st.spinner("🔄 Running pricing optimization..."):
        # Only generate random bids if not using uploaded or fixed bids
        if user_bids is None and not use_fixed_bids:
            bids = generate_customer_bids(num_customers, face_price, seed)
        # Stage 1
        accepted_stage1, remaining, stage1_units, delta1 = stage1_allocation(
            bids, face_price, total_inventory, strict_boundaries=strict_flag
        )
        # Stage 2
        inventory_left = total_inventory - stage1_units
        prices_stage2, lambda_star = stage2_pricing(remaining, face_price, delta1, inventory_left, priority=priority_flag)
    
    # Key Metrics Row
    st.markdown('<div class="stage-header">📊 Key Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Stage 1 Sales", f"{stage1_units}", f"{stage1_units/total_inventory*100:.1f}% of inventory")
    
    with col2:
        expected_stage2 = 0
        if prices_stage2:
            price_dict = {cid: p for cid, p in prices_stage2}
            for cid, bid in remaining:
                price = price_dict[cid]
                if price <= bid:
                    f_i = 1.0
                elif price <= 2 * bid:
                    f_i = ((2 * bid - price) / bid) ** 2
                else:
                    f_i = 0.0
                expected_stage2 += f_i
        
        st.metric("Expected Stage 2 Sales", f"{expected_stage2:.1f}", f"{expected_stage2/total_inventory*100:.1f}% of inventory")
    
    with col3:
        total_expected = stage1_units + expected_stage2
        st.metric("Total Expected Sales", f"{total_expected:.1f}", f"{total_expected/total_inventory*100:.1f}% utilization")
    
    with col4:
        bal_tag = "surplus" if delta1 > 0 else ("deficit" if delta1 < 0 else "balanced")
        st.metric("Revenue Balance", f"${delta1:+.2f}", f"Stage 1 {bal_tag}")
    
    with col5:
        customer_satisfaction = (stage1_units + expected_stage2) / num_customers * 100
        st.metric("Customer Satisfaction", f"{customer_satisfaction:.1f}%", "Customers served")

    # ------------------------------------------------------------------
    # 🎯 TASK-SPECIFIC METRICS
    # ------------------------------------------------------------------

    # Economic totals
    stage1_revenue = sum(b for _, b in accepted_stage1)

    stage2_revenue = 0.0
    if prices_stage2:
        price_dict = {cid: p for cid, p in prices_stage2}
        for cid, bid in remaining:
            price = price_dict[cid]
            if price <= bid:
                f_i = 1.0
            elif price <= 2 * bid:
                f_i = ((2 * bid - price) / bid) ** 2
            else:
                f_i = 0.0
            stage2_revenue += price * f_i

    vendor_revenue = stage1_revenue + stage2_revenue  # expected

    total_expected_sales = stage1_units + expected_stage2  # T (expected)

    # Task metrics
    proportion_customers = total_expected_sales / num_customers if num_customers else 0.0  # T/C
    proportion_items = total_expected_sales / total_inventory if total_inventory else 0.0    # T/N

    target_revenue = face_price * total_expected_sales  # P × T
    remaining_balance = vendor_revenue - target_revenue  # Σ − P×T  (should be ≈0)

    st.markdown('<div class="stage-header">📏 Task-Specific Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("T / C (customers served)", f"{proportion_customers:.3f}")

    with m2:
        st.metric("T / N (items sold)", f"{proportion_items:.3f}")

    with m3:
        st.metric("Vendor Revenue (P×T)", f"${target_revenue:.2f}")

    with m4:
        bal_tag2 = "surplus" if remaining_balance > 0 else ("deficit" if remaining_balance < 0 else "balanced")
        bal_color = "🟢" if abs(remaining_balance) < 1e-2 else ("🟡" if abs(remaining_balance) < 1 else "🔴")
        st.metric("Remaining Balance", f"${remaining_balance:+.6f}", f"{bal_tag2} {bal_color}")
    
    # Stage 1 Analysis
    st.markdown('<div class="stage-header">🎯 Stage 1: Bucket-Based Allocation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        bucket_fig = create_bucket_visualization(bids, face_price, accepted_stage1)
        st.plotly_chart(bucket_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Stage 1 Summary")
        st.write(f"**Customers processed:** {len(bids)}")
        st.write(f"**Customers accepted:** {stage1_units}")
        st.write(f"**Acceptance rate:** {stage1_units/len(bids)*100:.1f}%")
        st.write(f"**Revenue surplus/deficit:** ${delta1:+.2f}")
        st.write(f"**Inventory used:** {stage1_units}/{total_inventory}")
        
        if stage1_units > 0:
            avg_stage1_price = sum(bid for _, bid in accepted_stage1) / stage1_units
            st.write(f"**Average price paid:** ${avg_stage1_price:.2f}")
            st.write(f"**vs Face price:** {avg_stage1_price/face_price:.2%}")

    # --- NEW: Show all Stage 1 bids and acceptance status as a table ---
    st.markdown("#### Stage 1: All Bids and Acceptance Status")
    accepted_ids = {cid for cid, _ in accepted_stage1}
    stage1_tbl = []
    for cid, bid in bids:
        status = "Accepted" if cid in accepted_ids else "Rejected"
        stage1_tbl.append({"Customer": cid, "Bid": f"${bid:.2f}", "Status": status})
    st.dataframe(pd.DataFrame(stage1_tbl))

    # Stage 2 Analysis
    st.markdown('<div class="stage-header">🎨 Stage 2: Personalized Pricing</div>', unsafe_allow_html=True)
    
    if remaining and inventory_left > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            price_dist_fig = create_price_distribution_plot(remaining, prices_stage2)
            st.plotly_chart(price_dist_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Stage 2 Summary")
            st.write(f"**Remaining customers:** {len(remaining)}")
            st.write(f"**Remaining inventory:** {inventory_left}")
            
            if lambda_star != float('inf'):
                st.write(f"**Lambda* (dual variable):** {lambda_star:.6f}")
                st.write(f"**Priority mode:** {priority_flag}")
            
            # Calculate acceptance statistics
            if prices_stage2:
                price_dict = {cid: p for cid, p in prices_stage2}
                will_accept = sum(1 for cid, bid in remaining if price_dict[cid] <= bid)
                st.write(f"**Will definitely accept:** {will_accept}")
                st.write(f"**Expected acceptances:** {expected_stage2:.1f}")
                
                prices_list = [price_dict[cid] for cid, _ in remaining]
                st.write(f"**Price range:** ${min(prices_list):.2f} - ${max(prices_list):.2f}")

        # --- NEW: Show all Stage 2 bids, prices, and acceptance probabilities as a visible table ---
        st.markdown("#### Stage 2: All Bids, Personalized Prices, and Acceptance Probabilities")
        tbl = []
        price_dict = {cid: p for cid, p in prices_stage2}
        for cid, bid in remaining:
            P_i = price_dict[cid]
            if P_i <= bid:
                prob = 1.0
            elif P_i <= 2 * bid:
                prob = ((2 * bid - P_i) / bid) ** 2
            else:
                prob = 0.0
            status = f"Prob={prob:.2f}"
            tbl.append({"Customer": cid, "Bid": f"${bid:.2f}", "Price": f"${P_i:.2f}", "Acceptance Probability": status})
        st.dataframe(pd.DataFrame(tbl))
    else:
        if inventory_left <= 0:
            st.info("📦 All inventory was already sold in Stage 1.")
        else:
            st.info("🙅‍♂️ No customers left for Stage 2.")
    
    # --- NEW: Show all customer outcomes as a table (for professor) ---
    st.markdown("#### Final Customer Outcomes Table")
    outcome_tbl = []
    # Build lookup for Stage 1 and Stage 2
    accepted_stage1_ids = {cid for cid, _ in accepted_stage1}
    price_dict_stage2 = {cid: p for cid, p in prices_stage2} if prices_stage2 else {}
    for cid, bid in bids:
        if cid in accepted_stage1_ids:
            outcome = f"{bid:.2f}"  # Stage 1: price is the bid
        elif cid in price_dict_stage2:
            price = price_dict_stage2[cid]
            # Acceptance probability for Stage 2
            if price <= bid:
                outcome = f"{price:.2f}"
            else:
                outcome = "Incomplete"
        else:
            outcome = "Incomplete"
        outcome_tbl.append({"Customer Index": cid, "Initial Bid": f"{bid:.2f}", "Outcome": outcome})
    st.dataframe(pd.DataFrame(outcome_tbl))

    # --- Export to Excel button ---
    output_df = pd.DataFrame(outcome_tbl)
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name='Outcomes')
    towrite.seek(0)
    st.download_button(
        label="📥 Export Outcomes Table to Excel",
        data=towrite,
        file_name="customer_outcomes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Revenue Analysis
    st.markdown('<div class="stage-header">💰 Revenue Analysis</div>', unsafe_allow_html=True)
    
    revenue_fig = create_revenue_analysis(accepted_stage1, remaining, prices_stage2, face_price)
    st.plotly_chart(revenue_fig, use_container_width=True)
    
    # Monte Carlo Simulation
    if prices_stage2:
        st.markdown('<div class="stage-header">🎲 Monte Carlo Risk Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner("Running Monte Carlo simulation..."):
            mc_fig, sim_stats = run_monte_carlo_simulation(
                remaining, prices_stage2, face_price, stage1_units, total_inventory
            )
        
        if mc_fig:
            st.plotly_chart(mc_fig, use_container_width=True)
            
            # Simulation statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Units Sold", f"{sim_stats['mean_units']:.1f}")
                st.metric("5th-95th Percentile", f"{sim_stats['p5_units']:.1f} - {sim_stats['p95_units']:.1f}")
            
            with col2:
                st.metric("Mean Revenue", f"${sim_stats['mean_revenue']:.2f}")
                st.metric("5th-95th Percentile", f"${sim_stats['p5_revenue']:.2f} - ${sim_stats['p95_revenue']:.2f}")
            
            with col3:
                oversell_risk = sim_stats['oversell_prob'] * 100
                st.metric("Overselling Risk", f"{oversell_risk:.2f}%")
                risk_color = "🟢" if oversell_risk < 5 else "🟡" if oversell_risk < 15 else "🔴"
                st.write(f"{risk_color} Risk Level")
            
            with col4:
                revenue_efficiency = sim_stats['mean_revenue'] / (face_price * sim_stats['mean_units']) * 100
                st.metric("Revenue Efficiency", f"{revenue_efficiency:.1f}%")
                st.write("vs Face Price Baseline")
    
    # Advanced Analytics
    with st.expander("🔬 Advanced Analytics & Insights"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Economic Efficiency Metrics")
            
            # Calculate various efficiency metrics
            total_welfare = sum(bid for _, bid in bids)  # Total customer surplus potential
            captured_welfare_s1 = sum(bid for _, bid in accepted_stage1)
            
            if prices_stage2:
                price_dict = {cid: p for cid, p in prices_stage2}
                captured_welfare_s2 = sum(bid * (1.0 if price_dict[cid] <= bid else 
                                                ((2 * bid - price_dict[cid]) / bid) ** 2 if price_dict[cid] <= 2 * bid else 0.0)
                                        for cid, bid in remaining)
            else:
                captured_welfare_s2 = 0
            
            total_captured = captured_welfare_s1 + captured_welfare_s2
            efficiency = total_captured / total_welfare * 100
            
            st.write(f"**Total potential welfare:** ${total_welfare:.2f}")
            st.write(f"**Captured welfare:** ${total_captured:.2f}")
            st.write(f"**Economic efficiency:** {efficiency:.1f}%")
            
            # Price discrimination analysis
            if accepted_stage1:
                s1_prices = [bid for _, bid in accepted_stage1]
                s1_price_cv = np.std(s1_prices) / np.mean(s1_prices)
                st.write(f"**Stage 1 price variation:** {s1_price_cv:.3f}")
            
            if prices_stage2:
                s2_prices = [price for _, price in prices_stage2]
                s2_price_cv = np.std(s2_prices) / np.mean(s2_prices)
                st.write(f"**Stage 2 price variation:** {s2_price_cv:.3f}")
        
        with col2:
            st.markdown("### ⚖️ Fairness & Equity Analysis")
            
            # Calculate fairness metrics
            if len(bids) > 0:
                bid_values = [bid for _, bid in bids]
                served_bids = [bid for _, bid in accepted_stage1]
                
                if prices_stage2:
                    price_dict = {cid: p for cid, p in prices_stage2}
                    for cid, bid in remaining:
                        if price_dict[cid] <= bid:
                            served_bids.append(bid)
                
                if served_bids:
                    avg_served_bid = np.mean(served_bids)
                    avg_total_bid = np.mean(bid_values)
                    fairness_ratio = avg_served_bid / avg_total_bid
                    
                    st.write(f"**Average bid (all customers):** ${avg_total_bid:.2f}")
                    st.write(f"**Average bid (served customers):** ${avg_served_bid:.2f}")
                    st.write(f"**Fairness ratio:** {fairness_ratio:.3f}")
                    
                    if fairness_ratio > 1.1:
                        st.warning("⚠️ System may favor high-bid customers")
                    elif fairness_ratio < 0.9:
                        st.info("ℹ️ System serves lower-bid customers proportionally")
                    else:
                        st.success("✅ Balanced customer selection")
    
    # Export functionality
    st.markdown('<div class="stage-header">📋 Export Results</div>', unsafe_allow_html=True)
    
    if st.button("📊 Generate Detailed Report"):
        # Create comprehensive report
        report_data = {
            'System Parameters': {
                'Face Price': face_price,
                'Total Inventory': total_inventory,
                'Number of Customers': num_customers,
                'Random Seed': seed
            },
            'Stage 1 Results': {
                'Units Sold': stage1_units,
                'Customers Served': len(accepted_stage1),
                'Revenue Surplus/Deficit': delta1,
                'Acceptance Rate': f"{stage1_units/len(bids)*100:.1f}%"
            },
            'Stage 2 Results': {
                'Remaining Customers': len(remaining),
                'Expected Sales': f"{expected_stage2:.1f}",
                'Lambda Star': lambda_star if lambda_star != float('inf') else 'Infinity'
            },
            'Overall Performance': {
                'Total Expected Sales': f"{total_expected:.1f}",
                'Inventory Utilization': f"{total_expected/total_inventory*100:.1f}%",
                'Customer Satisfaction': f"{customer_satisfaction:.1f}%"
            }
        }
        
        st.json(report_data)
        st.success("📋 Report generated! You can copy the JSON data above.")

if __name__ == "__main__":
    main() 