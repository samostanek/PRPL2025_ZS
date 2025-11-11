"""
Test script for new GOLEM data loader features.
Run this in an environment where pandas, numpy, and h5py are installed.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from golem_data_loader import GolemDataLoader


def test_basic_diagnostics(loader):
    """Test basic diagnostics loading."""
    print("\n" + "=" * 70)
    print("TEST: Basic Diagnostics")
    print("=" * 70)

    try:
        basic = loader.load_basic_diagnostics()

        print(f"✓ Loaded basic diagnostics")
        print(f"  Toroidal field (Bt): {'✓' if basic.toroidal_field else '✗'}")
        if basic.toroidal_field:
            print(f"    - {len(basic.toroidal_field.time)} points")

        print(f"  Plasma current (Ip): {'✓' if basic.plasma_current else '✗'}")
        if basic.plasma_current:
            print(f"    - {len(basic.plasma_current.time)} points")

        print(f"  Chamber current (Ich): {'✓' if basic.chamber_current else '✗'}")
        if basic.chamber_current:
            print(f"    - {len(basic.chamber_current.time)} points")

        print(f"  Loop voltage: {'✓' if basic.loop_voltage else '✗'}")
        if basic.loop_voltage:
            print(f"    - {len(basic.loop_voltage.time)} points")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_mirnov_coils(loader):
    """Test Mirnov coils loading."""
    print("\n" + "=" * 70)
    print("TEST: Mirnov Coils")
    print("=" * 70)

    try:
        mirnov = loader.load_mirnov_coils()

        print(f"✓ Loaded {len(mirnov.coils)} coils")
        for coil_num, signal in sorted(mirnov.coils.items()):
            print(f"  Coil {coil_num}: {len(signal.time)} points")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_mhd_ring(loader):
    """Test MHD ring loading."""
    print("\n" + "=" * 70)
    print("TEST: MHD Ring")
    print("=" * 70)

    try:
        mhd = loader.load_mhd_ring()

        print(f"✓ Loaded {len(mhd.rings)} rings")
        for ring_num, signal in sorted(mhd.rings.items()):
            print(f"  Ring {ring_num}: {len(signal.time)} points")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_plasma_detection(loader):
    """Test plasma detection loading."""
    print("\n" + "=" * 70)
    print("TEST: Plasma Detection")
    print("=" * 70)

    try:
        plasma = loader.load_plasma_detection()

        signals = [
            ("BT Coil", plasma.bt_coil),
            ("Int BT Coil", plasma.int_bt_coil),
            ("Rogowski Coil", plasma.rog_coil),
            ("Int Rogowski Coil", plasma.int_rog_coil),
            ("Leybold Photocell", plasma.leyb_phot),
            ("Loop", plasma.loop),
        ]

        available = [name for name, sig in signals if sig is not None]
        print(f"✓ Loaded {len(available)}/{len(signals)} signals")

        for name, signal in signals:
            if signal:
                print(f"  {name}: {len(signal.time)} points")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_shot_info(loader):
    """Test shot info loading."""
    print("\n" + "=" * 70)
    print("TEST: Shot Info")
    print("=" * 70)

    try:
        info = loader.load_shot_info()

        print(f"✓ Shot #{info.shot_number}")
        print(f"  Logbook: {len(info.logbook) if info.logbook else 0} characters")

        avail_count = sum(1 for v in info.available_diagnostics.values() if v)
        total_count = len(info.available_diagnostics)
        print(f"  Available diagnostics: {avail_count}/{total_count}")

        if info.logbook:
            print(f"\n  Logbook excerpt:")
            print(f"  {'-'*66}")
            for line in info.logbook[:300].split("\n")[:5]:
                print(f"  {line[:66]}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_load_all(loader):
    """Test loading all diagnostics."""
    print("\n" + "=" * 70)
    print("TEST: Load All Diagnostics")
    print("=" * 70)

    try:
        all_data = loader.load_all_diagnostics()

        print(f"✓ Loaded {len(all_data)} diagnostic categories:")
        for key, value in all_data.items():
            print(f"  {key}: {type(value).__name__}")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GOLEM Data Loader - Extended Features Test Suite")
    print("=" * 70)

    shot_number = 50377
    print(f"\nTesting with shot #{shot_number}")

    loader = GolemDataLoader(shot_number)

    tests = [
        ("Basic Diagnostics", test_basic_diagnostics),
        ("Mirnov Coils", test_mirnov_coils),
        ("MHD Ring", test_mhd_ring),
        ("Plasma Detection", test_plasma_detection),
        ("Shot Info", test_shot_info),
        ("Load All", test_load_all),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func(loader)
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
