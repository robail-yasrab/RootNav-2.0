import xml.etree.cElementTree as ET
from xml.dom import minidom
from xml.etree import ElementTree
import os.path

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

class RSMLWriter():
    @staticmethod
    def save(key, output_dir, plants):
        root = ET.Element('rsml') 
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata,  'version').text = "1"
        ET.SubElement(metadata, 'unit').text = "pixel"
        ET.SubElement(metadata, 'resolution').text = "1"
        ET.SubElement(metadata, 'last-modified').text = "1"
        ET.SubElement(metadata, 'software').text = "ROOT_NAV.2.0"
        ET.SubElement(metadata, 'user').text = "Robi"
        ET.SubElement(metadata, 'file-key').text = key
        scene = ET.SubElement(root, 'scene')

        for plant_id, p in enumerate(plants):
            plant = ET.SubElement(scene, 'plant', id=str(plant_id+1), label="wheat_bluepaper")

            for primary_id, pri in enumerate(p.roots):
                priroot = ET.SubElement(plant, 'root', id=str(primary_id+1), label="primary", poaccession="1")
                geometry = ET.SubElement(priroot, 'geometry')
                polyline = ET.SubElement(geometry, 'polyline')

                spline = pri.spline
                rootnavspline = ET.SubElement(geometry, 'rootnavspline', controlpointseparation= str(spline.knot_spacing), tension=str(spline.tension))
                for c in spline.knots:
                    point = ET.SubElement(rootnavspline, 'point', x=str(c[0]), y=str(c[1]))

                poly = spline.polyline(sample_spacing = 1)
                for pt in poly:
                    point = ET.SubElement(polyline, 'point', x=str(pt[0]), y=str(pt[1]))

                for lateral_id, lat in enumerate(pri.roots):
                    latroot = ET.SubElement(priroot, 'root', id=str(primary_id+1)+"."+str(lateral_id+1), label="lateral")
                    lat_geometry = ET.SubElement(latroot, 'geometry')
                    lat_polyline = ET.SubElement(lat_geometry, 'polyline')
                    
                    lat_spline = lat.spline
                    lat_rootnavspline = ET.SubElement(lat_geometry, 'rootnavspline', controlpointseparation=str(lat_spline.knot_spacing), tension=str(lat_spline.tension))
                    for c in lat_spline.knots:
                        point = ET.SubElement(lat_rootnavspline, 'point', x=str(c[0]), y=str(c[1]))

                    lat_poly = lat_spline.polyline(sample_spacing = 1)
                    for pt in lat_poly:
                        point = ET.SubElement(lat_polyline, 'point', x=str(pt[0]), y=str(pt[1]))

        tree = ET.ElementTree(root)
        rsml_text = prettify(root)

        output_path = os.path.join(output_dir, "{0}.rsml".format(key))
        with open (output_path, 'w') as f:
            f.write(rsml_text)
