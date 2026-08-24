
def combine_foci_totals(a, b):
    res = { 0: 0 }

    def add_to_res(x):
        for (key, value) in x.items():
            if key not in res:
                res[key] = 0
            res[key] += value

    add_to_res(a)
    add_to_res(b)
    return res
           
class Stats:
    
    def __init__(self):
        self.total_passed_map_creation = 0
        self.total_failed_grid_creation = 0
        self.total_failed_contour_creation = 0
        self.total_tried_map_creation = 0

        self.totals_foci = { 0: 0 }

        self.total_cells = 0
        return

    def __add__(self, other):
        res = Stats()

        res.total_passed_map_creation = self.total_passed_map_creation + other.total_passed_map_creation
        res.total_failed_grid_creation = self.total_failed_grid_creation + other.total_failed_grid_creation
        res.total_failed_contour_creation = self.total_failed_contour_creation + other.total_failed_contour_creation
        res.total_tried_map_creation = self.total_tried_map_creation + other.total_tried_map_creation

        res.totals_foci = combine_foci_totals(self.totals_foci, other.totals_foci)

        res.total_cells = self.total_cells + other.total_cells
        
        return res

    def to_dict(self):
        
        res = {}

        res["total_cells"] = self.total_cells
        res["total_passed_map_creation"] = self.total_passed_map_creation
        res["total_failed_grid_creation"] = self.total_failed_grid_creation
        res["total_failed_contour_creation"] = self.total_failed_contour_creation
        res["total_tried_map_creation"] = self.total_tried_map_creation
        res["totals_foci"] = self.totals_foci

        return res

    def __repr__(self):
        return repr(self.to_dict())

