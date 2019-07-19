import math

def norm(p0,p1):
    """Calculate norm_2 distance between two points"""
    return math.sqrt((p1[0]-p0[0]) ** 2 + (p1[1]-p0[1]) ** 2)

def lerp(t,p0,p1):
    """ Linear interpolation between two points"""
    return (p0[0] * (1-t) + p1[0] * t, p0[1] * (1-t) + p1[1] * t)

def spline_interpolate(s,t,p1,p2,p3,p4):
    """Cubic hermite spline interpolation.

    Arguments:
    s -- tension parameter, usually should be calculated as (1-t)/2
    t -- time parameter between 0.0 and 1.0, representing distance along the spline from p2 to p3.
    p1,p2,p3,p4 -- Four control points necessary for interpolation.
    """

    # Cubic hermite spline, re-arranged and optimised
    t2 = t ** 2
    t3 = t ** 3
    a = s * (-t3 + 2 * t2 - t)
    b = s * (t2 - t3)
    c = b + (2 * t3 - 3 * t2 + 1)
    d = s * (t3 - 2 * t2 + t) + (-2 * t3 + 3 * t2)

    x = a * p1[0] + c * p2[0] + d * p3[0] - b * p4[0]
    y = a * p1[1] + c * p2[1] + d * p3[1] - b * p4[1]

    return (x,y)

def linear_sample(pts, max_segment_length):
    """Linearly resample a polyline at regular intervals.

    Arguments:
    pts -- list of points (x,y) from which to sample evenly
    max_segment_length -- The maximum spacing between new sample points. The number of points will be chosen to meet, but not exceed, this value.
    """

    # Resample control points
    cumul = [0.0] * len(pts)
    for i in range(1, len(pts)):
        p0 = pts[i-1]
        p1 = pts[i]
        cumul[i] = cumul[i-1] + norm(p0,p1)

    segment_count = int(math.ceil(cumul[-1] / max_segment_length))
    knot_count = segment_count + 1
    #print segment_count, max_segment_length
    if segment_count == 0:
        segment_count = 1
    control_spacing = cumul[-1] / segment_count

    # Sample from points at control_spacing
    knots = [pts[0]]
    current_sample_distance = 0.0
    search_pos = 0
    for i in range(1,knot_count-1):
        current_sample_distance += control_spacing
        # Find cumulative position
        seg_idx = 0
        for j in range(search_pos, len(cumul)):
            if cumul[j] > current_sample_distance:
                seg_idx = j;
                search_pos = j
                break;

        t = 1 - (cumul[seg_idx] - current_sample_distance) / (cumul[seg_idx] - cumul[seg_idx-1])
        new_point = lerp(t, pts[seg_idx-1], pts[seg_idx])
        knots.append(new_point)

    knots.append(pts[-1])
    return knots;

def spline_sample(knots, tension, segment_count):
    """Samples a spine at regular intervals.

    Arguments:
    knots -- list of (x,y) points representing the knots of the spline.
    tension -- Tension parameter [-1,1].
    segment_count -- The number of segments to create between each pair of knots.
    """

    s = (1-tension) / 2
    output_points = [knots[0]]

    reflect_start = (knots[0][0] - (knots[1][0] - knots[0][0]), knots[0][1] - (knots[1][1] - knots[0][1]))
    reflect_end = (knots[-1][0] - (knots[-2][0] - knots[-1][0]), knots[-1][1] - (knots[-2][1] - knots[-1][1]))
    
    for idx in range(len(knots) - 1):
        points = [None,knots[idx],knots[idx+1],None]
        points[0] = knots[idx-1] if idx > 0 else reflect_start
        points[3] = knots[idx+2] if idx < len(knots) - 2 else reflect_end

        for i in range(1, segment_count + 1):
            t = i / segment_count
            pt = spline_interpolate(s,t,
                                    points[0],
                                    points[1],
                                    points[2],
                                    points[3])
            output_points.append(pt)

    return output_points

class Spline():
    """Spline class for smooth resampling between polylines."""
    def __init__(self, points, tension = 0, knot_spacing = 10):
        """Create a new spline from a polyline.

        Arguments:
        points -- the original polyline from which to sample.
        tension (default: 0) -- Tension parameter in the range [-1,1].
        knot_spacing (default: 10) - Distance between knots sampled from polyline.
        """

        self.tension = tension
        self.knot_spacing = knot_spacing
        self.knots = linear_sample(points, self.knot_spacing)

    def polyline(self, sample_spacing = 1.0):
        """Returns a polyline representation of this spline.

        Arguments:
        sample_spacing (default: 1.0) -- The maximum distance between sample points. Will be adjusted automatically to preserve regular spacing.
        """
        
        # Oversample spline points
        spline_subsample_count = 2 * self.knot_spacing
        spline_points = spline_sample(self.knots, self.tension, spline_subsample_count)
        
        # Resample as polyline
        return linear_sample(spline_points, sample_spacing)

'''if __name__ == "__main__":
    # Create example data
    points = [(27,39),(54,26),(83,37),(108,29),(135,42),(161,67),(151,100),(116,80),(69,69),(30,91),(21,139),(61,170),(113,174),(148,149),(119,115),(71,112),(73,139),(106,146)]

    # Create a new spline based on "points" with a tension of 0 and
    # knots spaced at 30px intervals
    s = Spline(points, tension = 0, knot_spacing = 30)

    # Convert this spline to a polyline with samples every 1 unit length
    poly = s.polyline(sample_spacing = 1)

    # Plot
    import matplotlib.pyplot as plt
    plt.plot([a[0] for a in points],[a[1] for a in points],'-')
    plt.plot([c[0] for c in s.knots],[c[1] for c in s.knots],'X')
    plt.plot([c[0] for c in poly],[c[1] for c in poly],'-')
    plt.show()'''