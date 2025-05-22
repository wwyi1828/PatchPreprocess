import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import os
import glob
from pathlib import Path

class AnnotationConverter:
    def __init__(self):
        self.next_group_id = 0
    
    def convert_coordinates(self, vertices: List[ET.Element]) -> List[Dict[str, str]]:
        """Convert vertex coordinates to ASAP format"""
        coordinates = []
        for i, vertex in enumerate(vertices):
            coordinates.append({
                'Order': str(i),
                'X': vertex.get('X', '0'),
                'Y': vertex.get('Y', '0')
            })
        return coordinates

    def create_asap_xml(self, input_xml: str) -> str:
        """Convert input annotation XML to ASAP format"""
        # Parse input XML
        root = ET.fromstring(input_xml)
        
        # Create ASAP XML structure
        asap_root = ET.Element("ASAP_Annotations")
        annotations = ET.SubElement(asap_root, "Annotations")
        annotation_groups = ET.SubElement(asap_root, "AnnotationGroups")
        
        # Process each annotation
        for region in root.findall(".//Region"):
            # Create annotation element
            annotation = ET.SubElement(annotations, "Annotation")
            annotation.set("Name", f"_{self.next_group_id}")
            annotation.set("Type", "Spline")
            annotation.set("PartOfGroup", f"_{self.next_group_id}")
            annotation.set("Color", "#F4FA58")
            
            # Create coordinates section
            coordinates = ET.SubElement(annotation, "Coordinates")
            
            # Convert vertices to coordinates
            coord_list = self.convert_coordinates(region.findall("Vertices/Vertex"))
            for coord in coord_list:
                coord_elem = ET.SubElement(coordinates, "Coordinate")
                coord_elem.set("Order", coord['Order'])
                coord_elem.set("X", coord['X'])
                coord_elem.set("Y", coord['Y'])
            
            # Create annotation group
            group = ET.SubElement(annotation_groups, "Group")
            group.set("Name", f"_{self.next_group_id}")
            group.set("PartOfGroup", "None")
            group.set("Color", "#00ff00")
            
            # Add empty attributes
            attributes = ET.SubElement(group, "Attributes")
            
            self.next_group_id += 1
        
        # Convert to string with proper formatting
        def indent(elem: ET.Element, level: int = 0) -> None:
            i = "\n" + level * "\t"
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "\t"
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for subelem in elem:
                    indent(subelem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        indent(asap_root)
        return '<?xml version="1.0"?>\n' + ET.tostring(asap_root, encoding='unicode')

def process_directory(input_dir: str, output_dir: str):
    """Process all XML files in the input directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files in input directory
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    
    converter = AnnotationConverter()
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process each XML file
    for xml_file in xml_files:
        try:
            # Get the filename
            filename = os.path.basename(xml_file)
            print(f"Processing {filename}...")
            
            # Read input XML
            with open(xml_file, 'r') as f:
                input_xml = f.read()
            
            # Convert to ASAP format
            output_xml = converter.create_asap_xml(input_xml)
            
            # Create output file path
            output_file = os.path.join(output_dir, filename)
            
            # Save converted XML
            with open(output_file, 'w') as f:
                f.write(output_xml)
                
            print(f"Successfully converted {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print("\nConversion complete!")

def main():
    # Define input and output directories
    input_dir = "/data/data/BACH"
    output_dir = "/data/data/BACH/annots"
    
    # Process all XML files
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()