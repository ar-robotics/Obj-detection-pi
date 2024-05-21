import os
import shutil


def move_matching_files(source_folder, xml_folder, target_folder):
    # Ensure target folder exists, if not, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)
    files_xml = os.listdir(xml_folder)
    # Create a set of base names without file extensions
    base_names = set(os.path.splitext(file)[0] for file in files_xml)

    # Iterate over each unique file name
    for name in base_names:
        jpg_file = f"{name}.jpg"
        xml_file = f"{name}.xml"

        # Check if both .jpg and .xml files exist
        if jpg_file in files and xml_file in files_xml:
            # Move both files to the target folder
            shutil.move(
                os.path.join(source_folder, jpg_file),
                os.path.join(target_folder, jpg_file),
            )
            print(f"Moved: {jpg_file} and {xml_file}")
        else:
            print(f"Not moved: {jpg_file} or {xml_file} is missing")


# Define your source and target folders
source_folder = "C:/Users/aditi/Downloads/Person Dataset.v1i.voc/train/data"
xml_folder = "C:/Users/aditi/Downloads/Person Dataset.v1i.voc/train/annotations"
target_folder = "C:/Users/aditi/Downloads/Person Dataset.v1i.voc/train/dataa"

# Run the function
move_matching_files(source_folder, xml_folder, target_folder)
