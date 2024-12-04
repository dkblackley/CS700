import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression


def load_results(N_values):
    """Load results from individual JSON files"""
    results = {'simple_pir': {}, 'sublinear_pir': {}}
    for N in N_values:
        try:
            with open(f'pir_benchmark_results_N{N}.json', 'r') as f:
                data = json.load(f)
                results['simple_pir'][N] = data['simple_pir'][str(N)]
                results['sublinear_pir'][N] = data['sublinear_pir'][str(N)]
        except FileNotFoundError:
            print(f"Warning: No data file found for N={N}")
    return results


def calculate_statistics(data):
    """Calculate mean, std, and confidence intervals"""
    times = [d['time'] for d in data]
    bytes_total = [d['bytes_sent'] + d['bytes_received'] for d in data]

    time_mean = np.mean(times)
    time_std = np.std(times)
    time_ci = stats.t.interval(
        0.95,
        len(times) - 1,
        loc=time_mean,
        scale=stats.sem(times))

    bytes_mean = np.mean(bytes_total)
    bytes_std = np.std(bytes_total)
    bytes_ci = stats.t.interval(
        0.95,
        len(bytes_total) - 1,
        loc=bytes_mean,
        scale=stats.sem(bytes_total))

    return {
        'time': {'mean': time_mean, 'std': time_std, 'ci': time_ci},
        'bytes': {'mean': bytes_mean, 'std': bytes_std, 'ci': bytes_ci}
    }


def plot_results(results, N_values):
    """Create visualization plots"""
    # Prepare data
    simple_stats = {N: calculate_statistics(
        results['simple_pir'][N]) for N in N_values}
    sublinear_stats = {N: calculate_statistics(
        results['sublinear_pir'][N]) for N in N_values}

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plots with error bars
    width = 0.35
    simple_times = [simple_stats[N]['time']['mean'] for N in N_values]
    simple_times_err = [simple_stats[N]['time']['std'] for N in N_values]
    sublinear_times = [sublinear_stats[N]['time']['mean'] for N in N_values]
    sublinear_times_err = [sublinear_stats[N]['time']['std'] for N in N_values]

    x = np.arange(len(N_values))
    ax1.bar(
        x - width / 2,
        simple_times,
        width,
        yerr=simple_times_err,
        label='SimplePIR',
        capsize=5)
    ax1.bar(
        x + width / 2,
        sublinear_times,
        width,
        yerr=sublinear_times_err,
        label='SublinearPIR',
        capsize=5)
    ax1.set_xlabel('Database Size (N)')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('Query Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(N_values)
    ax1.legend()

    # Network usage comparison
    simple_bytes = [simple_stats[N]['bytes']['mean'] for N in N_values]
    simple_bytes_err = [simple_stats[N]['bytes']['std'] for N in N_values]
    sublinear_bytes = [sublinear_stats[N]['bytes']['mean'] for N in N_values]
    sublinear_bytes_err = [sublinear_stats[N]
                           ['bytes']['std'] for N in N_values]

    ax2.bar(
        x - width / 2,
        simple_bytes,
        width,
        yerr=simple_bytes_err,
        label='SimplePIR',
        capsize=5)
    ax2.bar(
        x + width / 2,
        sublinear_bytes,
        width,
        yerr=sublinear_bytes_err,
        label='SublinearPIR',
        capsize=5)
    ax2.set_xlabel('Database Size (N)')
    ax2.set_ylabel('Total Bytes Transferred')
    ax2.set_title('Network Usage Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(N_values)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('pir_comparison_bars.png')
    plt.close()

    # Linear regression plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time regression
    X = np.array(N_values).reshape(-1, 1)

    # SimplePIR
    model = LinearRegression()
    model.fit(X, simple_times)
    ax1.scatter(N_values, simple_times, label='SimplePIR actual')
    ax1.plot(
        N_values,
        model.predict(X),
        '--',
        label=f'SimplePIR trend (R²={model.score(X, simple_times):.3f})')

    # SublinearPIR
    model.fit(X, sublinear_times)
    ax1.scatter(N_values, sublinear_times, label='SublinearPIR actual')
    ax1.plot(
        N_values,
        model.predict(X),
        '--',
        label=f'SublinearPIR trend (R²={model.score(X, sublinear_times):.3f})')

    ax1.set_xlabel('Database Size (N)')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('Query Time Regression')
    ax1.legend()

    # Bytes regression
    # SimplePIR
    model.fit(X, simple_bytes)
    ax2.scatter(N_values, simple_bytes, label='SimplePIR actual')
    ax2.plot(
        N_values,
        model.predict(X),
        '--',
        label=f'SimplePIR trend (R²={model.score(X, simple_bytes):.3f})')

    # SublinearPIR
    model.fit(X, sublinear_bytes)
    ax2.scatter(N_values, sublinear_bytes, label='SublinearPIR actual')
    ax2.plot(
        N_values,
        model.predict(X),
        '--',
        label=f'SublinearPIR trend (R²={model.score(X, sublinear_bytes):.3f})')

    ax2.set_xlabel('Database Size (N)')
    ax2.set_ylabel('Bytes Transferred')
    ax2.set_title('Network Usage Regression')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('pir_comparison_regression.png')
    plt.close()

    # Q-Q plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

    # Combine all data across N values
    simple_times_all = [item['time']
                        for N in N_values for item in results['simple_pir'][N]]
    sublinear_times_all = [item['time']
                           for N in N_values for item in results['sublinear_pir'][N]]
    simple_bytes_all = [item['bytes_sent'] + item['bytes_received']
                        for N in N_values for item in results['simple_pir'][N]]
    sublinear_bytes_all = [item['bytes_sent'] + item['bytes_received']
                           for N in N_values for item in results['sublinear_pir'][N]]

    # Time Q-Q plots
    stats.probplot(simple_times_all, dist="norm", plot=ax1)
    ax1.set_title('SimplePIR Time Q-Q Plot')

    stats.probplot(sublinear_times_all, dist="norm", plot=ax2)
    ax2.set_title('SublinearPIR Time Q-Q Plot')

    # Bytes Q-Q plots
    stats.probplot(simple_bytes_all, dist="norm", plot=ax3)
    ax3.set_title('SimplePIR Bytes Q-Q Plot')

    stats.probplot(sublinear_bytes_all, dist="norm", plot=ax4)
    ax4.set_title('SublinearPIR Bytes Q-Q Plot')

    plt.tight_layout()
    plt.savefig('pir_qq_plots.png')
    plt.close()


def print_statistics(results, N_values):
    """Print statistical summary"""
    print("\nStatistical Summary:")
    for N in N_values:
        print(f"\nDatabase Size N = {N}")
        print("SimplePIR:")
        stats = calculate_statistics(results['simple_pir'][N])
        print(
            f"  Time: {stats['time']['mean']:.3f}s ± {stats['time']['std']:.3f} (95% CI: {stats['time']['ci']})")
        print(
            f"  Bytes: {stats['bytes']['mean']:.0f} ± {stats['bytes']['std']:.0f} (95% CI: {stats['bytes']['ci']})")

        print("SublinearPIR:")
        stats = calculate_statistics(results['sublinear_pir'][N])
        print(
            f"  Time: {stats['time']['mean']:.3f}s ± {stats['time']['std']:.3f} (95% CI: {stats['time']['ci']})")
        print(
            f"  Bytes: {stats['bytes']['mean']:.0f} ± {stats['bytes']['std']:.0f} (95% CI: {stats['bytes']['ci']})")


def main():
    # Match N values from benchmark script
    N_values = [4, 16, 64]
    results = load_results(N_values)
    plot_results(results, N_values)
    print_statistics(results, N_values)


if __name__ == "__main__":
    main()
