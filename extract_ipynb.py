import json
import traceback

try:
    with open("3_modeling.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    with open("3_modeling_code_and_output.txt", "w", encoding="utf-8") as out_f:
        cell_count = 1
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                out_f.write(f"--- Code Cell {cell_count} ---\n")
                out_f.write("[CODE]\n")
                out_f.write("".join(cell.get("source", [])) + "\n\n")
                
                out_f.write("[OUTPUT]\n")
                if "outputs" in cell and cell["outputs"]:
                    for output in cell["outputs"]:
                        if output.get("output_type") == "stream":
                            out_f.write("".join(output.get("text", [])))
                        elif output.get("output_type") in ["execute_result", "display_data"]:
                            data = output.get("data", {})
                            if "text/plain" in data:
                                out_f.write("".join(data.get("text/plain", [])) + "\n")
                        elif output.get("output_type") == "error":
                            out_f.write("ERROR:\n")
                            out_f.write("".join(output.get("traceback", [])) + "\n")
                else:
                    out_f.write("No output\n")
                out_f.write("\n" + "="*80 + "\n\n")
                cell_count += 1

    print("Extraction complete. Saved to 3_modeling_code_and_output.txt")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
