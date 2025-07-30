import os

def count_txt_files_recursive(directory,file_format=".txt"):
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(file_format):
                count += 1
    return count

def count_txt_files(directory):
    directory = input("Enter the directory path: ")
    if os.path.isdir(directory):
        txt_file_count = count_txt_files_recursive(directory)
        print(f"Number of .txt files in '{directory}': {txt_file_count}")
    else:
        print("Invalid directory path.")

if __name__ == "__main__":
    directory = "/home/ling/zyt/cifar100_images"
    file_format = ".png"
    txt_file_count=count_txt_files_recursive(directory,file_format)
    print(f"Number of {file_format} files in '{directory}': {txt_file_count}")