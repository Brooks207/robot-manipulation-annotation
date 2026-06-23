#!/usr/bin/env python3
"""Quick validation script to verify pipeline installation and basic functionality."""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are importable."""
    print("Checking imports...")
    missing = []

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "pillow"),
        ("yaml", "pyyaml"),
        ("pycocotools", "pycocotools"),
        ("matplotlib", "matplotlib"),
    ]

    for module, package_name in packages:
        try:
            __import__(module)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: uv pip install " + " ".join(missing))
        return False

    print("✅ All imports OK\n")
    return True


def check_modules():
    """Check if pipeline modules are loadable."""
    print("Checking pipeline modules...")

    try:
        from annotation.types import InstanceMask, DepthResult
        print("  ✓ Types")

        from annotation.discovery.base import ObjectDiscoverer
        from annotation.discovery.rule_extractor import RuleBasedExtractor
        from annotation.discovery.qwen_extractor import QwenExtractor
        print("  ✓ Discovery layer")

        from annotation.segmentation.base import Segmenter
        from annotation.segmentation.mock_segmenter import MockSegmenter
        from annotation.segmentation.sam3 import SAM3Segmenter
        print("  ✓ Segmentation layer")

        from annotation.depth.base import DepthEstimator
        from annotation.depth.mock_depth_estimator import MockDepthEstimator
        from annotation.depth.depth_anything3 import DepthAnything3Estimator
        print("  ✓ Depth layer")

        from annotation.storage.base import MaskWriter, DepthWriter
        from annotation.storage.mask_writer import ParquetMaskWriter
        from annotation.storage.depth_writer import PNG16DepthWriter
        print("  ✓ Storage layer")

        from annotation.qc.visualize import visualize_annotations
        print("  ✓ QC visualization")

        from pipeline import AnnotationPipeline, load_config_from_yaml
        print("  ✓ Pipeline orchestration")

        print("✅ All modules loadable\n")
        return True

    except Exception as e:
        print(f"❌ Module loading failed: {e}\n")
        return False


def check_configs():
    """Check if config files exist."""
    print("Checking config files...")

    # Try both relative to script and relative to cwd
    script_dir = Path(__file__).parent

    configs = [
        "configs/anygrasp_dryrun.yaml",
        "configs/anygrasp_full.yaml",
    ]

    all_exist = True
    for config_path in configs:
        path1 = Path(config_path)
        path2 = script_dir / config_path

        if path1.exists() or path2.exists():
            print(f"  ✓ {config_path}")
        else:
            print(f"  ✗ {config_path} - MISSING")
            all_exist = False

    if all_exist:
        print("✅ All configs present\n")
    else:
        print("❌ Some configs missing\n")

    return all_exist


def run_quick_test():
    """Run a minimal pipeline test."""
    print("Running quick pipeline test...")

    try:
        from pipeline import load_config_from_yaml
        from annotation.discovery.rule_extractor import RuleBasedExtractor

        # Test discovery only
        discoverer = RuleBasedExtractor()
        queries = discoverer.discover_objects(
            "pick up the red cup and place it on the table",
            {"always_include": ["robot hand"]}
        )

        print(f"  ✓ Discovery test passed")
        print(f"    Extracted queries: {queries}")

        print("✅ Quick test passed\n")
        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("annotation-pipeline - Installation Validation")
    print("=" * 60)
    print()

    results = []

    # Check imports
    results.append(("Imports", check_imports()))

    # Check modules
    results.append(("Modules", check_modules()))

    # Check configs
    results.append(("Configs", check_configs()))

    # Quick test
    results.append(("Quick test", run_quick_test()))

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Dry-run: python run_dryrun.py configs/anygrasp_dryrun.yaml")
        print("  2. Full test: python run_annotate.py configs/anygrasp_full.yaml --use-mock")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
