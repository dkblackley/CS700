import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def load_results(N_values):
    """Load results from individual JSON files with better error handling"""
    results = {'simple_pir': {}, 'sublinear_pir': {}}
    for N in N_values:
        try:
            with open(f'pir_benchmark_N{N}.json', 'r') as f:
                data = json.load(f)
                
                results['simple_pir'][N] = [{'time': data['simple']['time']['mean'],
                                           'bytes_sent': data['simple']['bytes_sent'],
                                           'bytes_received': data['simple']['bytes_received']}]
                
                results['sublinear_pir'][N] = [{'time': data['sublinear']['time']['mean'],
                                              'bytes_sent': data['sublinear']['bytes_sent'],
                                              'bytes_received': data['sublinear']['bytes_received']}]
                
        except FileNotFoundError:
            print(f"Warning: No data file found for N={N}")
        except KeyError as e:
            print(f"Warning: Unexpected data structure in file for N={N}. Missing key: {e}")
        except Exception as e:
            print(f"Error processing N={N}: {str(e)}")
    
    return results

def calculate_statistics(data):
    """Calculate statistics from the simplified data structure"""
    time_mean = data['time']['mean']
    time_ci = (data['time']['ci_low'], data['time']['ci_high'])
    bytes_total = data['bytes_sent'] + data['bytes_received']
    
    return {
        'time': {'mean': time_mean, 'std': (time_ci[1] - time_ci[0])/3.92, 'ci': time_ci},
        'bytes': {'mean': bytes_total, 'std': 0, 'ci': (bytes_total, bytes_total)}
    }

def add_theoretical_curves(ax, N_values, max_value):
    """Add theoretical complexity curves (sqrt(n) and log(n))"""
    x = np.linspace(min(N_values), max(N_values), 100)
    
    # Calculate sqrt(n) curve
    sqrt_curve = np.sqrt(x)
    # Normalize to match the scale of the actual data
    sqrt_curve = sqrt_curve * (max_value / np.max(sqrt_curve))
    
    # Calculate log(n) curve
    log_curve = np.log2(x)
    # Normalize to match the scale of the actual data
    log_curve = log_curve * (max_value / np.max(log_curve))
    
    ax.plot(x, sqrt_curve, ':', color='gray', alpha=0.3, label='√n complexity')
    ax.plot(x, log_curve, ':', color='darkgray', alpha=0.3, label='log(n) complexity')

def fit_polynomial_regression(x, y, degree=3):
    """Fit polynomial regression with specified degree"""
    X = np.array(x).reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate points for smooth curve
    X_smooth = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    
    return X_smooth.flatten(), y_smooth

def plot_results(results, N_values):
    """Create visualization plots with curved regression lines and theoretical curves"""
    # Prepare data
    simple_stats = {N: calculate_statistics(results[N]['simple']) for N in N_values}
    sublinear_stats = {N: calculate_statistics(results[N]['sublinear']) for N in N_values}
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # SimplePIR Time Plot (top left)
    simple_times = [simple_stats[N]['time']['mean'] for N in N_values]
    simple_times_err = [simple_stats[N]['time']['std'] for N in N_values]
    
    ax1.errorbar(N_values, simple_times, yerr=simple_times_err, fmt='o', 
                color='blue', alpha=0.6, label='Data points')
    
    X_smooth, y_simple = fit_polynomial_regression(N_values, simple_times)
    ax1.plot(X_smooth, y_simple, '-', color='blue', alpha=0.8, label='Regression curve')
    add_theoretical_curves(ax1, N_values, max(simple_times))
    
    ax1.set_xlabel('Database Size (N)')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('SimplePIR Query Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SublinearPIR Time Plot (top right)
    sublinear_times = [sublinear_stats[N]['time']['mean'] for N in N_values]
    sublinear_times_err = [sublinear_stats[N]['time']['std'] for N in N_values]
    
    ax2.errorbar(N_values, sublinear_times, yerr=sublinear_times_err, fmt='o', 
                color='orange', alpha=0.6, label='Data points')
    
    X_smooth, y_sublinear = fit_polynomial_regression(N_values, sublinear_times)
    ax2.plot(X_smooth, y_sublinear, '-', color='orange', alpha=0.8, label='Regression curve')
    add_theoretical_curves(ax2, N_values, max(sublinear_times))
    
    ax2.set_xlabel('Database Size (N)')
    ax2.set_ylabel('Query Time (s)')
    ax2.set_title('SublinearPIR Query Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # SimplePIR Bytes Plot (bottom left)
    simple_bytes = [simple_stats[N]['bytes']['mean'] for N in N_values]
    
    ax3.scatter(N_values, simple_bytes, color='blue', alpha=0.6, label='Data points')
    
    X_smooth, y_simple = fit_polynomial_regression(N_values, simple_bytes)
    ax3.plot(X_smooth, y_simple, '-', color='blue', alpha=0.8, label='Regression curve')
    add_theoretical_curves(ax3, N_values, max(simple_bytes))
    
    ax3.set_xlabel('Database Size (N)')
    ax3.set_ylabel('Total Bytes Transferred')
    ax3.set_title('SimplePIR Network Usage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # SublinearPIR Bytes Plot (bottom right)
    sublinear_bytes = [sublinear_stats[N]['bytes']['mean'] for N in N_values]
    
    ax4.scatter(N_values, sublinear_bytes, color='orange', alpha=0.6, label='Data points')
    
    X_smooth, y_sublinear = fit_polynomial_regression(N_values, sublinear_bytes)
    ax4.plot(X_smooth, y_sublinear, '-', color='orange', alpha=0.8, label='Regression curve')
    add_theoretical_curves(ax4, N_values, max(sublinear_bytes))
    
    ax4.set_xlabel('Database Size (N)')
    ax4.set_ylabel('Total Bytes Transferred')
    ax4.set_title('SublinearPIR Network Usage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pir_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    N_values = [4, 16]
    results = {}
    
    # Load the data
    for N in N_values:
        try:
            with open(f'pir_benchmark_N{N}.json', 'r') as f:
                results[N] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: No data file found for N={N}")
    
    if results:
        plot_results(results, N_values)
        print("\nPlots have been generated!")
    else:
        print("No data was loaded. Please check your benchmark result files.")

if __name__ == "__main__":
    main()
