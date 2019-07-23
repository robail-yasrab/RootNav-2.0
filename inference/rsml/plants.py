from .splines import Spline

class Plant():
    def __init__(self, id, label, seed=None, roots = []):
        self.id = id
        self.label = label
        self.seed = seed
        self.roots = roots

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
    def __init__(self, points, roots = [], spline_tension = 0.5, spline_knot_spacing = 50):
        self.roots = roots
        self.start = points[0]
        self.end = points[-1]
        self.spline = Spline(points, spline_tension, spline_knot_spacing)