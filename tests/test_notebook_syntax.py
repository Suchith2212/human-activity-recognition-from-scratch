"""Test script to parse and verify the python syntax of all code cells in the notebooks."""

import os
import json
import py_compile
import tempfile

def test_notebooks_syntax():
    notebooks_dir = "notebooks"
    notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")]
    
    print(f"Checking {len(notebook_files)} notebooks in {notebooks_dir}...")
    
    for nb_file in notebook_files:
        path = os.path.join(notebooks_dir, nb_file)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        print(f"  {nb_file}: found {len(code_cells)} code cells.")
        
        # Merge all code cells into a single python script
        py_code = []
        for idx, cell in enumerate(code_cells):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            # Skip pip installs or shell commands
            lines = [line for line in source.splitlines() if not line.strip().startswith("!") and "pip install" not in line]
            py_code.append(f"# --- Cell {idx} ---")
            py_code.extend(lines)
            py_code.append("")
            
        full_code = "\n".join(py_code)
        
        # Write to temp file and compile
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_f:
            temp_f.write(full_code)
            temp_path = temp_f.name
            
        try:
            py_compile.compile(temp_path, doraise=True)
            print(f"  [PASS] {nb_file} compiled successfully.")
        except py_compile.PyCompileError as e:
            print(f"  [FAIL] {nb_file} compilation failed:")
            print(e)
            raise e
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    test_notebooks_syntax()
    print("\nALL NOTEBOOKS COMPILED SUCCESSFULLY!")
