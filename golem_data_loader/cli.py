#!/usr/bin/env python3
"""
GOLEM Data Loader CLI

Command-line interface for loading and inspecting GOLEM tokamak diagnostic data.

Usage:
    golem-cli info <shot_number>              # Show shot information
    golem-cli list <shot_number>              # List available diagnostics
    golem-cli load <shot_number> <diagnostic> # Load specific diagnostic
    golem-cli export <shot_number> <output>   # Export all data to directory
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

try:
    from golem_data_loader import (
        GolemDataLoader,
        SpectroscopyLine,
        DataLoadError,
        FileNotFoundError as GolemFileNotFoundError,
    )
except ImportError:
    print(
        "Error: golem_data_loader package not found. Make sure it's in your PYTHONPATH."
    )
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)
    # Suppress golem_data_loader module logs unless verbose
    logging.getLogger("golem_data_loader").setLevel(level)
    logging.getLogger("golem_data_loader.golem_data_loader").setLevel(level)


def cmd_info(args):
    """Show shot information and logbook."""
    setup_logging(args.verbose)

    print(f"Loading information for shot {args.shot}...")
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    loader = GolemDataLoader(args.shot, log_level=log_level)

    try:
        info = loader.load_shot_info()

        print(f"\n{'='*70}")
        print(f"GOLEM Shot #{info.shot_number} Information")
        print(f"{'='*70}\n")

        # Show available diagnostics
        print("Available Diagnostics:")
        print("-" * 70)

        diag_groups = {
            "Fast Spectrometry": [],
            "Basic Diagnostics": [],
            "Magnetic Diagnostics": [],
            "Advanced Diagnostics": [],
        }

        for diag, available in sorted(info.available_diagnostics.items()):
            status = "✓" if available else "✗"
            if diag in ["Hα", "Hβ", "He I", "Whole"]:
                diag_groups["Fast Spectrometry"].append(f"  {status} {diag}")
            elif diag in ["BasicDiagnostics", "PlasmaDetection"]:
                diag_groups["Basic Diagnostics"].append(f"  {status} {diag}")
            elif diag in ["MirnovCoils", "MHDRing"]:
                diag_groups["Magnetic Diagnostics"].append(f"  {status} {diag}")
            else:
                diag_groups["Advanced Diagnostics"].append(f"  {status} {diag}")

        for group, items in diag_groups.items():
            if items:
                print(f"\n{group}:")
                for item in items:
                    print(item)

        # Show logbook excerpt
        if info.logbook:
            print(f"\n{'='*70}")
            print("Shot Logbook (first 1000 characters):")
            print("-" * 70)
            print(info.logbook[:1000])
            if len(info.logbook) > 1000:
                print(f"\n... ({len(info.logbook) - 1000} more characters)")
        else:
            print("\nNo logbook data available.")

        print(f"\n{'='*70}\n")

    except DataLoadError as e:
        print(f"Error loading shot info: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_list(args):
    """List available diagnostics for a shot."""
    setup_logging(args.verbose)

    print(f"Checking available diagnostics for shot {args.shot}...")
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    loader = GolemDataLoader(args.shot, log_level=log_level)

    try:
        available = loader.get_available_diagnostics()

        print(f"\n{'='*70}")
        print(f"Available Diagnostics for Shot #{args.shot}")
        print(f"{'='*70}\n")

        available_count = sum(1 for v in available.values() if v)
        total_count = len(available)

        for diag, is_available in sorted(available.items()):
            status = "✓ Available" if is_available else "✗ Not available"
            print(f"{status:20} {diag}")

        print(f"\n{available_count}/{total_count} diagnostics available\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_load(args):
    """Load specific diagnostic data."""
    setup_logging(args.verbose)

    print(f"Loading {args.diagnostic} data for shot {args.shot}...")
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    loader = GolemDataLoader(args.shot, log_level=log_level)

    try:
        diagnostic = args.diagnostic.lower()

        if diagnostic in ["basic", "basicdiagnostics"]:
            data = loader.load_basic_diagnostics()
            print("\nBasic Diagnostics Data:")
            print("-" * 50)

            if data.toroidal_field:
                print(f"✓ Toroidal Field (Bt): {len(data.toroidal_field.time)} points")
            if data.plasma_current:
                print(f"✓ Plasma Current (Ip): {len(data.plasma_current.time)} points")
            if data.chamber_current:
                print(
                    f"✓ Chamber Current (Ich): {len(data.chamber_current.time)} points"
                )
            if data.loop_voltage:
                print(f"✓ Loop Voltage: {len(data.loop_voltage.time)} points")

        elif diagnostic in ["spectrometry", "fastspectrometry"]:
            data = loader.load_fast_spectrometry()
            print("\nFast Spectrometry Data:")
            print("-" * 50)

            for name, signal in data.items():
                print(
                    f"✓ {name}: {len(signal.time)} points, "
                    f"range: [{signal.intensity.min():.3e}, {signal.intensity.max():.3e}]"
                )

        elif diagnostic in ["minispectrometer", "minispec"]:
            data = loader.load_minispectrometer_h5()
            print("\nMini-Spectrometer Data:")
            print("-" * 50)
            print(f"✓ Spectra shape: {data.spectra.shape}")
            print(
                f"✓ Wavelength range: {data.wavelengths[0]:.2f} - {data.wavelengths[-1]:.2f} nm"
            )
            data.cleanup()

        elif diagnostic in ["mirnov", "mirnovcoils"]:
            data = loader.load_mirnov_coils()
            print("\nMirnov Coils Data:")
            print("-" * 50)

            for coil_num, signal in sorted(data.coils.items()):
                print(f"✓ Coil {coil_num}: {len(signal.time)} points")

        elif diagnostic in ["mhd", "mhdring"]:
            data = loader.load_mhd_ring()
            print("\nMHD Ring Data:")
            print("-" * 50)

            for ring_num, signal in sorted(data.rings.items()):
                print(f"✓ Ring {ring_num}: {len(signal.time)} points")

        elif diagnostic in ["plasma", "plasmadetection"]:
            data = loader.load_plasma_detection()
            print("\nPlasma Detection Data:")
            print("-" * 50)

            signals = [
                ("BT Coil", data.bt_coil),
                ("Integrated BT Coil", data.int_bt_coil),
                ("Rogowski Coil", data.rog_coil),
                ("Integrated Rogowski Coil", data.int_rog_coil),
                ("Leybold Photocell", data.leyb_phot),
                ("Loop", data.loop),
            ]

            for name, signal in signals:
                if signal:
                    print(f"✓ {name}: {len(signal.time)} points")

        elif diagnostic == "all":
            print("\nLoading all diagnostics...")
            data = loader.load_all_diagnostics()

            print("\n" + "=" * 70)
            print("All Available Data:")
            print("=" * 70 + "\n")

            for key, value in data.items():
                print(f"• {key}: {type(value).__name__}")

        else:
            print(f"Unknown diagnostic: {args.diagnostic}", file=sys.stderr)
            print(
                "\nAvailable diagnostics: basic, spectrometry, minispectrometer, "
                "mirnov, mhd, plasma, all"
            )
            return 1

        print()

    except GolemFileNotFoundError as e:
        print(f"Error: Diagnostic not available for this shot: {e}", file=sys.stderr)
        return 1
    except DataLoadError as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def cmd_export(args):
    """Export all data to directory."""
    setup_logging(args.verbose)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting shot {args.shot} data to {output_dir}...")
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    loader = GolemDataLoader(args.shot, log_level=log_level)

    try:
        all_data = loader.load_all_diagnostics()

        # Export each diagnostic type
        exported = []

        # Basic diagnostics
        if "basic" in all_data:
            basic = all_data["basic"]
            for name, df in basic.raw_dataframes.items():
                output_file = output_dir / f"basic_{name}.csv"
                df.to_csv(output_file, index=False)
                exported.append(output_file.name)

        # Fast spectrometry
        if "spectrometry" in all_data:
            for name, signal in all_data["spectrometry"].items():
                safe_name = (
                    name.replace(" ", "_").replace("α", "alpha").replace("β", "beta")
                )
                output_file = output_dir / f"spectrometry_{safe_name}.csv"
                signal.raw_dataframe.to_csv(output_file, index=False)
                exported.append(output_file.name)

        # Mirnov coils
        if "mirnov" in all_data:
            for coil_num, signal in all_data["mirnov"].coils.items():
                output_file = output_dir / f"mirnov_coil_{coil_num}.csv"
                signal.raw_dataframe.to_csv(output_file, index=False)
                exported.append(output_file.name)

        # MHD ring
        if "mhd_ring" in all_data:
            for ring_num, signal in all_data["mhd_ring"].rings.items():
                output_file = output_dir / f"mhd_ring_{ring_num}.csv"
                signal.raw_dataframe.to_csv(output_file, index=False)
                exported.append(output_file.name)

        # Plasma detection
        if "plasma_detection" in all_data:
            for name, df in all_data["plasma_detection"].raw_dataframes.items():
                output_file = output_dir / f"plasma_{name}.csv"
                df.to_csv(output_file, index=False)
                exported.append(output_file.name)

        # Shot info
        if "info" in all_data:
            info_file = output_dir / "shot_info.txt"
            with open(info_file, "w") as f:
                f.write(f"GOLEM Shot #{all_data['info'].shot_number}\n")
                f.write("=" * 70 + "\n\n")
                if all_data["info"].logbook:
                    f.write("Logbook:\n")
                    f.write("-" * 70 + "\n")
                    f.write(all_data["info"].logbook)
                    f.write("\n\n")
                f.write("Available Diagnostics:\n")
                f.write("-" * 70 + "\n")
                for diag, avail in sorted(
                    all_data["info"].available_diagnostics.items()
                ):
                    f.write(f"{'✓' if avail else '✗'} {diag}\n")
            exported.append(info_file.name)

        print(f"\nExported {len(exported)} files:")
        for filename in sorted(exported):
            print(f"  • {filename}")

        print(f"\nAll data exported to: {output_dir.absolute()}\n")

    except Exception as e:
        print(f"Error exporting data: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def cmd_export_hdf5(args):
    """Export all diagnostics to a single HDF5 file."""
    setup_logging(args.verbose)

    print(f"Exporting shot {args.shot} to HDF5 format...")
    log_level = logging.DEBUG if args.verbose else logging.ERROR
    loader = GolemDataLoader(args.shot, log_level=log_level)

    try:
        # Determine output path
        output_path = args.output
        if not output_path.endswith(".h5") and not output_path.endswith(".hdf5"):
            output_path = f"{output_path}.h5"

        # Determine which diagnostics to include
        include_diagnostics = None
        if hasattr(args, "diagnostics") and args.diagnostics:
            include_diagnostics = args.diagnostics.split(",")

        # Set compression options
        compression = "gzip"
        compression_opts = (
            args.compression_level if hasattr(args, "compression_level") else 4
        )

        # Export to HDF5
        loader.export_to_hdf5(
            output_path,
            include_diagnostics=include_diagnostics,
            compression=compression,
            compression_opts=compression_opts,
        )

        # Show file info
        file_path = Path(output_path)
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"\n{'='*70}")
            print(f"✓ Successfully exported shot {args.shot}")
            print(f"  Output file: {output_path}")
            print(f"  File size: {file_size:.2f} MB")
            print(f"  Compression: {compression} (level {compression_opts})")
            print(f"{'='*70}\n")

            # Show how to read the file
            print("To read this HDF5 file in Python:")
            print(f"  >>> import h5py")
            print(f"  >>> f = h5py.File('{output_path}', 'r')")
            print(f"  >>> list(f.keys())  # Show available groups")
            print(f"  >>> plasma_current = f['basic/plasma_current/intensity'][:]")
            print(f"  >>> f.close()")
            print()

        return 0

    except Exception as e:
        print(f"Error exporting to HDF5: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GOLEM Data Loader CLI - Load and inspect GOLEM tokamak diagnostic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info 50377                    # Show shot information
  %(prog)s list 50377                    # List available diagnostics
  %(prog)s load 50377 basic              # Load basic diagnostics
  %(prog)s load 50377 spectrometry       # Load fast spectrometry
  %(prog)s load 50377 all                # Load all diagnostics
  %(prog)s export 50377 ./data/shot50377 # Export all data to CSV files
  %(prog)s export-hdf5 50377 shot.h5     # Export all data to HDF5 file
  %(prog)s export-hdf5 50377 shot.h5 -d basic,spectrometry  # Export specific diagnostics
  
Diagnostic types:
  basic, spectrometry, minispectrometer, mirnov, mhd, plasma, all
        """,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output and debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show shot information and logbook"
    )
    info_parser.add_argument("shot", type=int, help="Shot number")
    info_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List available diagnostics for a shot"
    )
    list_parser.add_argument("shot", type=int, help="Shot number")
    list_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # Load command
    load_parser = subparsers.add_parser("load", help="Load specific diagnostic data")
    load_parser.add_argument("shot", type=int, help="Shot number")
    load_parser.add_argument(
        "diagnostic",
        help="Diagnostic type (basic, spectrometry, minispectrometer, mirnov, mhd, plasma, all)",
    )
    load_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export all diagnostic data to a directory"
    )
    export_parser.add_argument("shot", type=int, help="Shot number")
    export_parser.add_argument("output", help="Output directory path")
    export_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # Export HDF5 command
    hdf5_parser = subparsers.add_parser(
        "export-hdf5", help="Export all diagnostics to a single HDF5 file"
    )
    hdf5_parser.add_argument("shot", type=int, help="Shot number")
    hdf5_parser.add_argument("output", help="Output HDF5 file path (.h5 or .hdf5)")
    hdf5_parser.add_argument(
        "-d",
        "--diagnostics",
        help="Comma-separated list of diagnostics to include (default: all)",
        default=None,
    )
    hdf5_parser.add_argument(
        "-c",
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=4,
        help="HDF5 compression level 0-9 (default: 4)",
    )
    hdf5_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    commands = {
        "info": cmd_info,
        "list": cmd_list,
        "load": cmd_load,
        "export": cmd_export,
        "export-hdf5": cmd_export_hdf5,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
