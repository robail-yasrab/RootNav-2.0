import xml

class RSMLParser():
    @staticmethod
    def parse(path, round_points = False):
        e = xml.etree.ElementTree.parse(path).getroot()
        metadata = e.find('metadata')
        scene = e.find('scene')
        # Only returns plants in current implementation
        return [Plant(p, round_points) for p in scene.findall('plant')]

class Plant():
    def __init__(self, xml_node, round_points = False):
        assert(xml_node.tag == 'plant')
        self.id = xml_node.attrib.get('ID')
        self.label = xml_node.attrib.get('label')
        self.roots = [Root(child_node, round_points) for child_node in xml_node.findall('root')]

        self.seed = self.roots[0].start if self.roots else None

    def all_roots(self):
        for r in self.roots:
            # Return current primary
            yield r
            # Return all child roots
            for c in r.roots:
                yield c

    def primary_roots(self):
        for r in self.roots:
            # Return only primary
            yield r

    def lateral_roots(self):
        for r in self.roots:
            # Return only child roots
            for c in r.roots:
                yield c

class Root():
    def __init__(self, xml_node, round_points = False):
        assert(xml_node.tag == 'root')
        self.id = xml_node.attrib.get('ID')
        self.label = xml_node.attrib.get('label')
        
        self.points = [(float(p.attrib['x']), float(p.attrib['y'])) for p in xml_node.find('geometry').find('polyline')]
        
        if round_points:
            self.points = [(int(round(p[0])),int(round(p[1]))) for p in self.points]

        self.roots = [Root(child_node, round_points) for child_node in xml_node.findall('root')]

        self.start = self.points[0] if self.points else None
        self.end = self.points[-1] if self.points else None

    def pairwise(self):
        return zip(self.points[:-1],self.points[1:])
