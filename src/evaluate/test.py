from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def get_lattice_param (input_file):
    # Read a QE input (e.g., scf.in)
    pw = PWInput.from_file(input_file)   
    s = pw.structure 
    
    # 2) Lattice constants
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    print(f"a={a:.6f} Å, b={b:.6f} Å, c={c:.6f} Å")
    print(f"α={alpha:.6f}°, β={beta:.6f}°, γ={gamma:.6f}°")
    return a, b, c, alpha, beta, gamma
    
def get_symm (input_file):
    # Space group info
    pw = PWInput.from_file(input_file) 
    s = pw.structure 
    sga = SpacegroupAnalyzer(s, symprec=1e-3, angle_tolerance=5)
    SG = sga.get_space_group_symbol()
    SG_num = sga.get_space_group_number()
    PG = sga.get_point_group_symbol()
    CS = sga.get_crystal_system()
    print("SG symbol:", SG)
    print("SG number:", SG_num)
    print("Point group:", PG)
    print("Crystal system:", CS)
    return SG, SG_num, PG, CS

def _cli():
    get_lattice_param("input_1.in")
    get_symm("input_1.in")

if __name__ == "__main__":
    _cli()
