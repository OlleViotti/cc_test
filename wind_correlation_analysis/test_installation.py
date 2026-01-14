#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")

    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'yaml',
        'requests',
        'xarray'
    ]

    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            failed.append(package)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages installed")
        return True


def test_modules():
    """Test that custom modules can be imported"""
    print("\nTesting custom modules...")

    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    modules = [
        ('data_acquisition.metar_downloader', 'METARDownloader'),
        ('data_acquisition.era5_downloader', 'ERA5Downloader'),
        ('analysis.spatial_utils', 'SpatialCalculations'),
        ('analysis.correlation_analysis', 'CrossCorrelationAnalysis'),
        ('visualization.polar_plots', 'PolarCorrelationPlot'),
    ]

    failed = []
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name} - ERROR: {e}")
            failed.append(module_name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All custom modules loaded successfully")
        return True


def test_directory_structure():
    """Test that required directories exist"""
    print("\nChecking directory structure...")

    required_dirs = [
        'config',
        'data/raw/metar',
        'data/raw/era5',
        'data/processed',
        'src/data_acquisition',
        'src/analysis',
        'src/visualization',
        'results/data',
        'results/figures',
        'notebooks'
    ]

    base_path = Path(__file__).parent
    missing = []

    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            missing.append(dir_path)

    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All required directories exist")
        return True


def test_config():
    """Test that configuration file is valid"""
    print("\nTesting configuration...")

    try:
        import yaml
        config_path = Path(__file__).parent / 'config' / 'stations.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required keys
        if 'stations' in config and isinstance(config['stations'], list):
            print(f"  ✓ Found {len(config['stations'])} stations in config")

            # Check first station has required fields
            if config['stations']:
                required_fields = ['id', 'name', 'lat', 'lon']
                first_station = config['stations'][0]

                for field in required_fields:
                    if field in first_station:
                        print(f"  ✓ Station field '{field}' present")
                    else:
                        print(f"  ✗ Station field '{field}' missing")
                        return False

            print("\n✓ Configuration file is valid")
            return True
        else:
            print("  ✗ Invalid configuration format")
            return False

    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with synthetic data"""
    print("\nTesting basic functionality...")

    try:
        import numpy as np
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).parent / 'src'))

        from analysis.spatial_utils import SpatialCalculations, CircularStatistics

        # Test distance calculation
        lat1, lon1 = 59.3544, 17.9419  # Stockholm
        lat2, lon2 = 57.6628, 12.2798  # Göteborg

        distance = SpatialCalculations.haversine_distance(lat1, lon1, lat2, lon2)
        print(f"  ✓ Distance calculation: {distance:.1f} km")

        bearing = SpatialCalculations.bearing(lat1, lon1, lat2, lon2)
        print(f"  ✓ Bearing calculation: {bearing:.1f}°")

        # Test circular statistics
        angles = np.array([350, 10, 5, 355])
        mean_angle = CircularStatistics.circular_mean(angles)
        print(f"  ✓ Circular mean: {mean_angle:.1f}°")

        print("\n✓ Basic functionality works")
        return True

    except Exception as e:
        print(f"  ✗ Error in functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("Wind Correlation Analysis - Installation Test")
    print("=" * 80)

    results = []

    results.append(("Package imports", test_imports()))
    results.append(("Custom modules", test_modules()))
    results.append(("Directory structure", test_directory_structure()))
    results.append(("Configuration", test_config()))
    results.append(("Basic functionality", test_basic_functionality()))

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed! Installation is complete.")
        print("\nYou can now run the analysis with:")
        print("  python run_analysis.py")
        print("\nOr explore the Jupyter notebook:")
        print("  jupyter notebook notebooks/example_analysis.ipynb")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
