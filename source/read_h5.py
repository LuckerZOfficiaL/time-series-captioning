import h5py
import numpy as np  # For handling numerical data (optional)

def read_h5_file(file_path):
    """Reads an H5 file and prints its contents."""

    try:
        with h5py.File(file_path, 'r') as hf:
            def print_content(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"    {key}: {val}")

                if isinstance(obj, h5py.Dataset):
                    print("    Data:")
                    if obj.ndim == 0:
                        print("        ", obj[()]) # Print scalar values
                    elif obj.ndim == 1:
                        print("        ", obj[:]) # Print 1D arrays
                    else:
                        print("        ", obj[:]) #Print multidimensional arrays.
                    #If the arrays are very large, you may want to only print a small portion.
                    #print("        ", obj[0:10])

            hf.visititems(print_content)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: Replace 'your_file.h5' with the actual path to your H5 file
def main():
    read_h5_file('/home/ubuntu/thesis/data/datasets/Traffic/METR-LA.h5')

if __name__ == "__main__":
    main()