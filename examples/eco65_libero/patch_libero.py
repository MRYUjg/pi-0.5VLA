"""Runtime patcher: replace Panda robot with ECO65 in LIBERO + robosuite.

1. Finds the exact robot name strings used in LIBERO source
2. Replaces them with ECO65 equivalents
3. Registers ECO65 (and mounted variants) in robosuite ROBOT_CLASS_MAPPING
"""

import pathlib
import re
import site
import sys


def find_libero_sources():
    """Find all LIBERO Python files that may reference robot names."""
    sources = []
    # Installed package
    for site_dir in site.getsitepackages():
        libero_dir = pathlib.Path(site_dir) / "libero"
        if libero_dir.exists():
            for f in libero_dir.rglob("*.py"):
                sources.append(f)
    # Mounted third_party
    tp = pathlib.Path("/app/third_party/libero")
    if tp.exists():
        for f in tp.rglob("*.py"):
            sources.append(f)
    return sources


def find_robot_strings(text):
    """Extract all quoted strings containing 'Panda'."""
    strings = set()
    # Double-quoted
    for m in re.finditer(r'"([^"]*?Panda[^"]*?)"', text):
        strings.add(m.group(1))
    # Single-quoted
    for m in re.finditer(r"'([^']*?Panda[^']*?)'", text):
        strings.add(m.group(1))
    return strings


def register_eco65_in_robosuite():
    """Ensure ECO65 and its mounted variants exist in robosuite ROBOT_CLASS_MAPPING."""
    try:
        import robosuite
        from robosuite.robots import ROBOT_CLASS_MAPPING, Manipulator
    except ImportError:
        print("  (robosuite not importable, skipping robot registration)")
        return

    # Check what ECO65 variants we have
    eco65_variants = [k for k in ROBOT_CLASS_MAPPING if "ECO65" in k]
    print(f"  Existing ECO65 entries in ROBOT_CLASS_MAPPING: {eco65_variants}")

    if "ECO65" not in ROBOT_CLASS_MAPPING:
        print("  WARNING: 'ECO65' not found in ROBOT_CLASS_MAPPING!")
        print("  Expecting ECO65 to be pre-registered in robosuite.")
        print("  Available robots:", sorted(ROBOT_CLASS_MAPPING.keys()))
        return

    eco65_cls = ROBOT_CLASS_MAPPING["ECO65"]
    print(f"  ECO65 class: {eco65_cls}")

    # Register mounted variant if needed
    for variant_name in ["MountedECO65"]:
        if variant_name not in ROBOT_CLASS_MAPPING:
            # Check if Panda has a mounted variant to copy the pattern
            if "MountedPanda" in ROBOT_CLASS_MAPPING:
                print(f"  Registering {variant_name} in ROBOT_CLASS_MAPPING...")
                # ECO65 might need its own mounted wrapper, but let us
                # try registering the base class first
                ROBOT_CLASS_MAPPING[variant_name] = eco65_cls
                print(f"  [OK] {variant_name} registered")
            else:
                print(f"  Note: 'MountedPanda' not found either, "
                      f"LIBERO may not use mount prefix")


def patch_libero():
    print("=" * 50)
    print("LIBERO Patch: Panda -> ECO65")
    print("=" * 50)

    # Step 1: find all Panda strings in LIBERO
    print("
[1] Scanning LIBERO sources for Panda references...")
    sources = find_libero_sources()
    all_panda_strings = set()
    files_to_patch = []

    for src in sources:
        try:
            text = src.read_text()
        except Exception:
            continue
        panda_strings = find_robot_strings(text)
        if panda_strings:
            files_to_patch.append((src, panda_strings))
            all_panda_strings.update(panda_strings)

    if not all_panda_strings:
        print("  No Panda strings found in any LIBERO file!")
    else:
        print(f"  Found {len(all_panda_strings)} unique Panda strings:")
        for s in sorted(all_panda_strings):
            print(f"    "{s}"")

    # Step 2: register ECO65 in robosuite
    print("
[2] Registering ECO65 in robosuite...")
    register_eco65_in_robosuite()

    # Step 3: replace in each file
    print("
[3] Replacing Panda -> ECO65 in source files...")
    total_replacements = 0
    for src, panda_strings in files_to_patch:
        text = src.read_text()
        modified = False
        for s in sorted(panda_strings, key=len, reverse=True):
            new_s = s.replace("Panda", "ECO65").replace(
                "PandaGripper", "RethinkGripper"
            )
            if new_s != s:
                # Replace quoted occurrences
                for quote in ('"', "'"):
                    old_quoted = f"{quote}{s}{quote}"
                    new_quoted = f"{quote}{new_s}{quote}"
                    if old_quoted in text:
                        text = text.replace(old_quoted, new_quoted)
                        print(f"  {src.name}: {s} -> {new_s}")
                        total_replacements += 1
                        modified = True
        if modified:
            src.write_text(text)

    print(f"
[4] Done: {total_replacements} replacements across "
          f"{len(files_to_patch)} files")

    if total_replacements == 0:
        print("
*** MANUAL ACTION NEEDED ***")
        print("Could not automatically replace robot names.")
        print("Please check the LIBERO source files listed above.")
        sys.exit(1)


if __name__ == "__main__":
    patch_libero()
