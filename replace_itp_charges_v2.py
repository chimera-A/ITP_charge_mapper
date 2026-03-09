#Version 2, 10 March 2026.
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SECTION_HEADER_RE = re.compile(r"^\s*\[\s*(?P<name>[^\]]+?)\s*\]")
ELEMENT_RE = re.compile(r"^[A-Za-z]+")


def infer_element(atom_name: str) -> str:
    m = ELEMENT_RE.match(atom_name)
    if not m:
        raise ValueError(f"Cannot infer element from atom name: {atom_name!r}")
    return m.group(0).capitalize()


def parse_mol2_atom_names(path: Path) -> List[str]:
    names: List[str] = []
    in_atom_block = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "@<TRIPOS>ATOM":
            in_atom_block = True
            continue
        if in_atom_block and line.startswith("@<TRIPOS>"):
            break
        if in_atom_block and line:
            parts = raw.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed MOL2 atom line in {path}: {raw!r}")
            names.append(parts[1])
    if not names:
        raise ValueError(f"No atoms found in MOL2 file: {path}")
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Atom names in MOL2 are not unique: {dupes}")
    return names


def parse_chg_charges(path: Path) -> List[float]:
    charges: List[float] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed CHG line {i} in {path}: {raw!r}")
        try:
            charges.append(float(parts[4]))
        except ValueError as exc:
            raise ValueError(f"Invalid charge on CHG line {i} in {path}: {parts[4]!r}") from exc
    if not charges:
        raise ValueError(f"No charges found in CHG file: {path}")
    return charges


def build_name_to_charge(mmff_mol2: Path, chg: Path) -> Dict[str, float]:
    atom_names = parse_mol2_atom_names(mmff_mol2)
    charges = parse_chg_charges(chg)
    if len(atom_names) != len(charges):
        raise ValueError(
            f"Atom count mismatch: MOL2 has {len(atom_names)} atoms, CHG has {len(charges)} charges"
        )
    return dict(zip(atom_names, charges))


def adjust_total_charge(
    name_to_charge: Dict[str, float],
    target_total: float,
    preferred_elements: List[str] | None = None,
    preferred_atoms: List[str] | None = None,
    max_correction_abs: float | None = 1e-4,
) -> Tuple[Dict[str, float], str, float, float]:
    adjusted = dict(name_to_charge)
    current_total = sum(adjusted.values())
    delta = target_total - current_total

    if abs(delta) < 1e-15:
        return adjusted, "", delta, current_total

    if max_correction_abs is not None and abs(delta) > max_correction_abs:
        raise ValueError(
            f"Required charge correction ({delta:+.10f}) exceeds the allowed threshold "
            f"({max_correction_abs:.10f}). This is likely more than harmless rounding noise; "
            "please inspect the input charges or override with --max-correction-abs."
        )

    candidate_name = None

    if preferred_atoms:
        for atom in preferred_atoms:
            if atom in adjusted:
                candidate_name = atom
                break
        if candidate_name is None:
            raise ValueError(
                "None of the requested --adjust-on-atoms entries were found in the MOL2/CHG atom names"
            )

    if candidate_name is None:
        elements = [e.capitalize() for e in (preferred_elements or ["H"])]
        for elem in elements:
            for atom_name in adjusted:
                if infer_element(atom_name) == elem:
                    candidate_name = atom_name
                    break
            if candidate_name is not None:
                break

    if candidate_name is None:
        raise ValueError(
            "Could not find a suitable atom for charge correction. "
            "Try --adjust-on-atoms with a specific atom name."
        )

    adjusted[candidate_name] += delta
    return adjusted, candidate_name, delta, current_total



def replace_itp_charges(
    itp_in: Path,
    itp_out: Path,
    name_to_charge: Dict[str, float],
    charge_decimals: int = 8,
) -> Tuple[int, float, float, List[Tuple[int, str, float, float]]]:
    lines = itp_in.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    in_atoms = False
    replaced = 0
    old_total = 0.0
    new_total = 0.0
    seen_names = set()
    mapping_preview: List[Tuple[int, str, float, float]] = []
    out_lines: List[str] = []

    for raw in lines:
        stripped = raw.strip()
        header_match = SECTION_HEADER_RE.match(raw)
        if header_match:
            section_name = header_match.group("name").strip().lower()
            in_atoms = section_name == "atoms"
            out_lines.append(raw)
            continue

        if not in_atoms or not stripped or stripped.startswith(";"):
            out_lines.append(raw)
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            in_atoms = False
            out_lines.append(raw)
            continue

        comment = ""
        content = raw.rstrip("\n")
        if ";" in content:
            content, comment = content.split(";", 1)
            comment = ";" + comment

        parts = content.split()
        if len(parts) < 8:
            out_lines.append(raw)
            continue

        atom_name = parts[4]
        if atom_name not in name_to_charge:
            raise KeyError(f"Atom {atom_name!r} in ITP not found in MOL2/CHG mapping")

        old_charge = float(parts[6])
        new_charge = name_to_charge[atom_name]
        old_total += old_charge
        new_total += new_charge
        replaced += 1
        seen_names.add(atom_name)
        if len(mapping_preview) < 12:
            mapping_preview.append((int(parts[0]), atom_name, old_charge, new_charge))

        parts[6] = f"{new_charge:.{charge_decimals}f}"
        rebuilt = (
            f"{int(parts[0]):>4d} {parts[1]:<5} {int(parts[2]):>2d} {parts[3]:<4} "
            f"{parts[4]:<6} {int(parts[5]):>4d} {parts[6]:>12} {float(parts[7]):>8.4f} "
        )
        if comment:
            rebuilt += comment
        if raw.endswith("\n"):
            rebuilt += "\n"
        out_lines.append(rebuilt)

    missing = sorted(set(name_to_charge) - seen_names)
    if missing:
        raise KeyError(f"These MOL2 atoms were not found in ITP [ atoms ]: {', '.join(missing)}")

    itp_out.write_text("".join(out_lines), encoding="utf-8")
    return replaced, old_total, new_total, mapping_preview



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace the [ atoms ] charge column in a GROMACS ITP using CHG charges mapped via MMFF MOL2 atom names."
    )
    parser.add_argument("--mmff-mol2", required=True, type=Path, help="MOL2 with unique atom names matching the ITP")
    parser.add_argument("--chg", required=True, type=Path, help="CHG file containing RESP/RESP2 charges in MOL2 atom order")
    parser.add_argument("--itp-in", required=True, type=Path, help="Input ITP file")
    parser.add_argument("--itp-out", required=True, type=Path, help="Output ITP file")
    parser.add_argument("--decimals", type=int, default=8, help="Decimal places for written charges (default: 8)")
    parser.add_argument(
        "--target-total-charge",
        type=float,
        default=None,
        help="Force the mapped charges to sum exactly to this value by applying the residual to one atom",
    )
    parser.add_argument(
        "--max-correction-abs",
        type=float,
        default=1e-4,
        help=(
            "Maximum absolute residual allowed for automatic total-charge correction "
            "(default: 1e-4). Set a negative value to disable this safety threshold."
        ),
    )
    parser.add_argument(
        "--adjust-on-elements",
        nargs="+",
        default=["H"],
        help="Element priority list for choosing the correction atom (default: H)",
    )
    parser.add_argument(
        "--adjust-on-atoms",
        nargs="+",
        default=None,
        help="Specific atom names to try first for absorbing the residual, e.g. H12 H15",
    )
    args = parser.parse_args()

    try:
        name_to_charge = build_name_to_charge(args.mmff_mol2, args.chg)

        correction_atom = None
        correction_delta = 0.0
        raw_total = sum(name_to_charge.values())
        threshold = None if args.max_correction_abs < 0 else args.max_correction_abs
        if args.target_total_charge is not None:
            name_to_charge, correction_atom, correction_delta, raw_total = adjust_total_charge(
                name_to_charge,
                target_total=args.target_total_charge,
                preferred_elements=args.adjust_on_elements,
                preferred_atoms=args.adjust_on_atoms,
                max_correction_abs=threshold,
            )

        replaced, old_total, new_total, preview = replace_itp_charges(
            args.itp_in, args.itp_out, name_to_charge, charge_decimals=args.decimals
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote: {args.itp_out}")
    print(f"Replaced {replaced} charges in [ atoms ]")
    print(f"Old total charge in ITP [ atoms ]: {old_total:.10f}")
    print(f"New total charge from CHG      : {raw_total:.10f}")
    if args.target_total_charge is not None:
        print(f"Target total charge           : {args.target_total_charge:.10f}")
        if threshold is None:
            print("Auto-correction threshold     : disabled")
        else:
            print(f"Auto-correction threshold     : {threshold:.10f}")
        print(f"Final written total charge    : {new_total:.10f}")
        if correction_atom:
            print(
                f"Residual correction applied to {correction_atom}: "
                f"{correction_delta:+.{max(args.decimals + 2, 10)}f}"
            )
    else:
        print(f"Final written total charge    : {new_total:.10f}")

    if preview:
        print("\nPreview of first mapped atoms:")
        for nr, atom_name, old_q, new_q in preview:
            print(f"  {nr:>4d} {atom_name:<6} {old_q:>12.8f} -> {new_q:>12.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
