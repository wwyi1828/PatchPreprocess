import pickle
import argparse
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

def convert_gdata_to_simplenamespace(input_pkl, output_pkl):
    """
    Convert torch_geometric.Data objects to SimpleNamespace objects in pkl file

    Args:
        input_pkl: Path to input pkl file containing torch_geometric.Data objects
        output_pkl: Path to output pkl file with SimpleNamespace objects
    """
    print(f"Loading data from {input_pkl}...")

    # Load original data
    try:
        with open(input_pkl, 'rb') as f:
            original_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {input_pkl}: {e}")
        return False

    print(f"Found {len(original_data)} objects to convert")

    # Convert each object
    converted_data = []
    for i, obj in enumerate(tqdm(original_data, desc="Converting objects")):
        try:
            # Extract only essential attributes from torch_geometric.Data object
            converted_obj = SimpleNamespace()

            # Only copy essential attributes
            essential_attrs = ['x', 'pos', 'y', 'graph_y', 'slide_index']

            for attr_name in essential_attrs:
                if hasattr(obj, attr_name):
                    setattr(converted_obj, attr_name, getattr(obj, attr_name))

            converted_data.append(converted_obj)

        except Exception as e:
            print(f"Error converting object {i}: {e}")
            return False

    # Save converted data
    print(f"Saving converted data to {output_pkl}...")
    try:
        with open(output_pkl, 'wb') as f:
            pickle.dump(converted_data, f)
        print(f"Successfully saved {len(converted_data)} converted objects")
        return True
    except Exception as e:
        print(f"Error saving {output_pkl}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert torch_geometric.Data to SimpleNamespace in pkl files')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input pkl file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output pkl file path')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup of original file')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    # Create backup if requested
    if args.backup:
        backup_path = input_path.with_suffix('.backup.pkl')
        print(f"Creating backup at {backup_path}")
        try:
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"Backup created successfully")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Perform conversion
    success = convert_gdata_to_simplenamespace(input_path, output_path)

    if success:
        print("Conversion completed successfully!")

        # Show file size comparison
        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {input_size:.2f} MB -> {output_size:.2f} MB")
    else:
        print("Conversion failed!")

if __name__ == "__main__":
    main()