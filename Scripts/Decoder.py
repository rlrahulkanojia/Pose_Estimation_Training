import sys
import time
from collections import defaultdict
from queue import PriorityQueue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def scalar_values(field, x, y, default=-1):
    values = np.full((x.shape[0],), default, dtype=np.float32)
    maxx = field.shape[1] - 1
    maxy = field.shape[0] - 1

    for i in range(values.shape[0]):
        if x[i] < 0.0 or y[i] < 0.0 or x[i] > maxx or y[i] > maxy:
            continue
        values[i] = field[int(y[i]), int(x[i])]
    return values
def scalar_square_add_single(field, x, y, sigma, value):
    minx = max(0, int(x - sigma))
    miny = max(0, int(y - sigma))
    maxx = max(minx + 1, min(field.shape[1], int(x + sigma) + 1))
    maxy = max(miny + 1, min(field.shape[0], int(y + sigma) + 1))
    field[miny:maxy, minx:maxx] += value
def scalar_nonzero_clipped_with_reduction(field,x,y,r):
    x = np.clip(x / r, 0.0, field.shape[1] - 1) 
    y = np.clip(y / r, 0.0, field.shape[0] - 1) 
    return field[int(y) , int(x)]
def caf_center_s(caf_field,x, y, sigma):
    result_np = np.empty_like(caf_field)

    result_i = 0
    for i in range(caf_field.shape[1]):
        if caf_field[1, i] < x - sigma:
            continue
        if caf_field[1, i] > x + sigma:
            continue
        if caf_field[2, i] < y - sigma:
            continue
        if caf_field[2, i] > y + sigma:
            continue

        result_np[:, result_i] = caf_field[:, i]
        result_i += 1
    return result_np[:, :result_i]   # remember to custom after calling

def approx_exp(x):
    if x > 2.0 or x < -2.0:
        return 0.0
    x = 1.0 + x / 8.0
    x *= x
    x *= x
    x *= x
    return x
def scalar_square_add_gauss_with_max(field, x,  y, sigma,v,truncate=2.0, max_value=1.0):
    deltax2 = None
    deltay2 = None

    for i in range(x.shape[0]):
        csigma = sigma[i]
        csigma2 = csigma * csigma
        cx = x[i]
        cy = y[i]
        cv = v[i]

        minx = int(np.clip(cx - truncate * csigma, 0, field.shape[1] - 1))
        maxx = int(np.clip(cx + truncate * csigma, minx + 1, field.shape[1]))
        miny = int(np.clip(cy - truncate * csigma, 0, field.shape[0] - 1))
        maxy = int(np.clip(cy + truncate * csigma, miny + 1, field.shape[0]))
        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2

                if deltax2 < 0.25 and deltay2 < 0.25:
                    # this is the closest pixel
                    vv = cv
                else:
                    vv = cv * approx_exp(-0.5 * (deltax2 + deltay2) / csigma2)

                field[yy, xx] += vv
                field[yy, xx] = min(max_value, field[yy, xx])



NOTSET = '__notset__'
class Annotation:
    def __init__(self, keypoints, skeleton, *, category_id=1, suppress_score_index=None):
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.category_id = category_id
        self.suppress_score_index = suppress_score_index

        self.data = np.zeros((len(keypoints), 3), dtype=np.float32)
        self.joint_scales = np.zeros((len(keypoints),), dtype=np.float32)
        self.fixed_score = NOTSET
        self.decoding_order = []
        self.frontier_order = []

        self.skeleton_m1 = (np.asarray(skeleton) - 1).tolist()

        self.score_weights = np.ones((len(keypoints),))
        if self.suppress_score_index:
            self.score_weights[-1] = 0.0
        self.score_weights[:3] = 3.0
        self.score_weights /= np.sum(self.score_weights)

    def add(self, joint_i, xyv):
        self.data[joint_i] = xyv
        return self

    def set(self, data, joint_scales=None, *, fixed_score=NOTSET):
        self.data = data
        if joint_scales is not None:
            self.joint_scales = joint_scales
        else:
            self.joint_scales[:] = 0.0
        self.fixed_score = fixed_score
        return self

    def rescale(self, scale_factor):
        self.data[:, 0:2] *= scale_factor
        if self.joint_scales is not None:
            self.joint_scales *= scale_factor
        for _, __, c1, c2 in self.decoding_order:
            c1[:2] *= scale_factor
            c2[:2] *= scale_factor
        return self

    def fill_joint_scales(self, scales, hr_scale=1.0):
        self.joint_scales = np.zeros((self.data.shape[0],))
        for xyv_i, xyv in enumerate(self.data):
            if xyv[2] == 0.0:
                continue
            scale = scales#scalar_value_clipped(scales[xyv_i], xyv[0] * hr_scale, xyv[1] * hr_scale)
            self.joint_scales[xyv_i] = scale / hr_scale

    def score(self):
        if self.fixed_score != NOTSET:
            return self.fixed_score

        v = self.data[:, 2]
        if self.suppress_score_index is not None:
            v = np.copy(v)
            v[self.suppress_score_index] = 0.0
        # return 0.1 * np.max(v) + 0.9 * np.mean(np.square(v))
        # return np.mean(np.square(v))
        # return np.sum(self.score_weights * np.sort(np.square(v))[::-1])
        return np.sum(self.score_weights * np.sort(v)[::-1])

    def scale(self, v_th=0.5):
        m = self.data[:, 2] > v_th
        if not np.any(m):
            return 0.0
        return max(
            np.max(self.data[m, 0]) - np.min(self.data[m, 0]),
            np.max(self.data[m, 1]) - np.min(self.data[m, 1]),
        )

    def json_data(self):
        """Data ready for json dump."""

        # avoid visible keypoints becoming invisible due to rounding
        v_mask = self.data[:, 2] > 0.0
        keypoints = np.copy(self.data)
        keypoints[v_mask, 2] = np.maximum(0.01, keypoints[v_mask, 2])
        keypoints = np.around(keypoints.astype(np.float64), 2)

        # convert to float64 before rounding because otherwise extra digits
        # will be added when converting to Python type
        data = {
            'keypoints': keypoints.reshape(-1).tolist(),
            'bbox': [round(float(c), 2) for c in self.bbox()],
            'score': max(0.001, round(self.score(), 3)),
            'category_id': self.category_id,
        }

        id_ = getattr(self, 'id_', None)
        if id_:
            data['id_'] = id_

        return data

    def bbox(self):
        return self.bbox_from_keypoints(self.data, self.joint_scales)

    @staticmethod
    def bbox_from_keypoints(kps, joint_scales):
        m = kps[:, 2] > 0
        if not np.any(m):
            return [0, 0, 0, 0]

        x = np.min(kps[:, 0][m] - joint_scales[m])
        y = np.min(kps[:, 1][m] - joint_scales[m])
        w = np.max(kps[:, 0][m] + joint_scales[m]) - x
        h = np.max(kps[:, 1][m] + joint_scales[m]) - y
        return [x, y, w, h]
    
    
class CifHr:
    neighbors = 16
    v_threshold = 0.1
    debug_visualizer = None

    def __init__(self):
        self.accumulated = None

    def accumulate(self, len_cifs, t, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]

        v, x, y, _, scale = p
        x = x * stride
        y = y * stride
        sigma = np.maximum(1.0, 0.5 * scale * stride)

        scalar_square_add_gauss_with_max(
           t, x, y, sigma, v / self.neighbors / len_cifs, truncate=1.0)

    def fill_multiple(self, cifs, stride, min_scale=0.0):


        if self.accumulated is None:
            shape = (
                cifs[0].shape[0],
                int((cifs[0].shape[2] - 1) * stride + 1),
                int((cifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)
        
        
        for cif in cifs:
            
            for t, p in zip(ta, cif):
                self.accumulate(len(cifs), t, p, stride, min_scale)

        if self.accumulated is None:
            self.accumulated = ta
        else:
            self.accumulated = np.maximum(ta, self.accumulated)

        return self

    def fill(self, fields):

        self.fill_multiple([fields[0]], 8, min_scale=0.0)
        return self  
class CifHr:
    neighbors = 16
    v_threshold = 0.1
    debug_visualizer = None

    def __init__(self):
        self.accumulated = None

    def accumulate(self, len_cifs, t, p, stride, min_scale):
        p = p[:, p[0] > self.v_threshold]
        if min_scale:
            p = p[:, p[4] > min_scale / stride]

        v, x, y, _, scale = p
        x = x * stride
        y = y * stride
        sigma = np.maximum(1.0, 0.5 * scale * stride)

        scalar_square_add_gauss_with_max(
           t, x, y, sigma, v / self.neighbors / len_cifs, truncate=1.0)

    def fill_multiple(self, cifs, stride, min_scale=0.0):


        if self.accumulated is None:
            shape = (
                cifs[0].shape[0],
                int((cifs[0].shape[2] - 1) * stride + 1),
                int((cifs[0].shape[3] - 1) * stride + 1),
            )
            ta = np.zeros(shape, dtype=np.float32)
        else:
            ta = np.zeros(self.accumulated.shape, dtype=np.float32)
        
        
        for cif in cifs:
            
            for t, p in zip(ta, cif):
                self.accumulate(len(cifs), t, p, stride, min_scale)

        if self.accumulated is None:
            self.accumulated = ta
        else:
            self.accumulated = np.maximum(ta, self.accumulated)

        return self

    def fill(self, fields):
        self.fill_multiple([fields[0]], 8, min_scale=0.0)
        return self    
class CifSeeds:
    threshold = 0.3 #0.5
    score_scale = 1.0
    debug_visualizer = None

    def __init__(self, cifhr: CifHr):
        self.cifhr = cifhr
        self.seeds = []

    def fill_cif(self, cif, stride, *, min_scale=0.0, seed_mask=None):
        start = time.perf_counter()
        sv = 0.0
        for field_i, p in enumerate(cif):
            if seed_mask is not None and not seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold / 2.0]
            if min_scale:
                p = p[:, p[4] > min_scale / stride]
            _, x, y, _, s = p

            start_sv = time.perf_counter()
            

            v = scalar_values(self.cifhr[field_i], x * stride, y * stride)
            
            sv += time.perf_counter() - start_sv

            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x, y, v, s = x[m] * stride, y[m] * stride, v[m], s[m] * stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))
        return self
    def get(self):
        return sorted(self.seeds, reverse=True)
    def fill(self, fields):
  
        self.fill_cif(fields[0], 8,min_scale=0.0, seed_mask=None)

        return self
    
    
class CafScored:
    default_score_th = 0.1
    def __init__(self, cifhr, skeleton, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.skeleton = skeleton
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        self.forward = None
        self.backward = None
    def directed(self, caf_i, forward):
        if forward:
            return self.forward[caf_i], self.backward[caf_i]

        return self.backward[caf_i], self.forward[caf_i]
    def fill_caf(self, caf, stride, min_distance=0.0, max_distance=None):
        if self.forward is None:
            self.forward = [np.empty((9, 0), dtype=caf.dtype) for _ in caf]
            self.backward = [np.empty((9, 0), dtype=caf.dtype) for _ in caf]

        for caf_i, nine in enumerate(caf):
            assert nine.shape[0] == 9
            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]
            nine = np.copy(nine)
            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= stride
            scores = nine[0]

            j1i = self.skeleton[caf_i][0] - 1
            
            cifhr_b = scalar_values(self.cifhr[j1i], nine[1], nine[2], default=0.0)
            scores_b = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_b)
            
            mask_b = scores_b > self.score_th
            d9_b = np.copy(nine[:, mask_b][(0, 5, 6, 7, 8, 1, 2, 3, 4), :])
            d9_b[0] = scores_b[mask_b]
            self.backward[caf_i] = np.concatenate((self.backward[caf_i], d9_b), axis=1)

            j2i = self.skeleton[caf_i][1] - 1
            
            cifhr_f = scalar_values(self.cifhr[j2i], nine[5], nine[6], default=0.0)
            scores_f = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_f)
            mask_f = scores_f > self.score_th
            d9_f = np.copy(nine[:, mask_f])
            d9_f[0] = scores_f[mask_f]
            self.forward[caf_i] = np.concatenate((self.forward[caf_i], d9_f), axis=1)

        return self
    
    def fill(self, fields):
        self.fill_caf(fields[1], 8,
                          min_distance=0.0, max_distance=None)

        return self
    
class Occupancy():
    def __init__(self, shape, reduction, *, min_scale=None):
        assert len(shape) == 3
        if min_scale is None:
            min_scale = reduction
        assert min_scale >= reduction

        self.reduction = reduction
        self.min_scale = min_scale
        self.min_scale_reduced = min_scale / reduction
        
        self.occupancy = np.zeros((
            shape[0],
            int(shape[1] / reduction),
            int(shape[2] / reduction),
        ), dtype=np.uint8)
        


    def __len__(self):
        return len(self.occupancy)

    def set(self, f, x, y, sigma):
        """Setting needs to be centered at the rounded (x, y)."""
        
        if f >= len(self.occupancy):
            return

        xi = round(x / self.reduction)
        yi = round(y / self.reduction)
        si = round(max(self.min_scale_reduced, sigma / self.reduction))
        scalar_square_add_single(self.occupancy[f], xi, yi, si, value=1)

    def get(self, f, x, y):
        """Getting needs to be done at the floor of (x, y)."""
        if f >= len(self.occupancy):
            return 1.0
        return scalar_nonzero_clipped_with_reduction(self.occupancy[f], x, y, self.reduction)
    
    
    
class Keypoints:
    suppression = 0.0
    instance_threshold = 0.2 #0.2
    keypoint_threshold = 0.2 #0.2
    occupancy_visualizer = None

    def annotations(self, anns):
        start = time.perf_counter()

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score() >= self.instance_threshold]

        if not anns:
            return anns

        occupied = Occupancy((
            len(anns[0].data),
            int(max(np.max(ann.data[:, 1]) for ann in anns) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in anns) + 1),
        ), 2, min_scale=4)

        anns = sorted(anns, key=lambda a: -a.score())
        for ann in anns:
            assert ann.joint_scales is not None
            assert len(occupied) == len(ann.data)
            for f, (xyv, joint_s) in enumerate(zip(ann.data, ann.joint_scales)):
                v = xyv[2]
                if v == 0.0:
                    continue

                if occupied.get(f, xyv[0], xyv[1]):
                    xyv[2] *= self.suppression
                else:
                    occupied.set(f, xyv[0], xyv[1], joint_s)  # joint_s = 2 * sigma

        if self.occupancy_visualizer is not None:

            self.occupancy_visualizer.predicted(occupied)

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score() >= self.instance_threshold]
        anns = sorted(anns, key=lambda a: -a.score())


        return anns
    
class CifCaf(): #Generator
    init_time = time.time()
    connection_method = 'blend'
    occupancy_visualizer = None
    force_complete = False
    greedy = False
    keypoint_threshold = 0.0
    print("Decoder init")
    

    def __init__(self, keypoints,skeleton,out_skeleton=None,confidence_scales=None,nms=True):

        super().__init__()
        if nms is True:
            nms = Keypoints()#nms_module.Keypoints()

#         self.field_config = field_config

        self.keypoints = keypoints
        self.skeleton = skeleton
        self.skeleton_m1 = np.asarray(skeleton) - 1
        self.out_skeleton = out_skeleton or skeleton
        self.confidence_scales = confidence_scales
        self.nms = nms



        # init by_target and by_source
        self.by_target = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_target[j2][j1] = (caf_i, True)
            self.by_target[j1][j2] = (caf_i, False)
        self.by_source = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1][j2] = (caf_i, True)
            self.by_source[j2][j1] = (caf_i, False)
    def decode(self, fields, initial_annotations=None):
        
        if not initial_annotations:
            initial_annotations = []

        cifhr = CifHr().fill(fields)
        seeds = CifSeeds(cifhr.accumulated).fill(fields)
        caf_scored = CafScored(cifhr.accumulated, self.skeleton).fill(fields)
        occupied = Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
        annotations = []

        
        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma
        
        for ann in initial_annotations:
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)
        
        ref_time = time.time()
        for v, f, x, y, s in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = Annotation(self.keypoints, self.out_skeleton).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)
            
        if self.nms is not None:
            annotations = self.nms.annotations(annotations)
        return annotations

    def _grow_connection(self, xy, xy_scale, caf_field):
        assert len(xy) == 2
        assert caf_field.shape[0] == 9

        # source value
        caf_field = caf_center_s(caf_field, xy[0], xy[1], sigma=2.0 * xy_scale)
        if caf_field.shape[1] == 0:
            return 0, 0, 0, 0

        # source distance
        d = np.linalg.norm(((xy[0],), (xy[1],)) - caf_field[1:3], axis=0)

        # combined value and source distance
        v = caf_field[0]
        sigma = 0.5 * xy_scale
        scores = np.exp(-0.5 * d**2 / sigma**2) * v

        return self._target_with_blend(caf_field[5:], scores)
        raise Exception('connection method not known')
    @staticmethod
    def _target_with_blend(target_coordinates, scores):
        assert target_coordinates.shape[1] == len(scores)
        if len(scores) == 1:
            return (
                target_coordinates[0, 0],
                target_coordinates[1, 0],
                target_coordinates[3, 0],
                scores[0] * 0.5,
            )

        sorted_i = np.argsort(scores)
        max_entry_1 = target_coordinates[:, sorted_i[-1]]
        max_entry_2 = target_coordinates[:, sorted_i[-2]]

        score_1 = scores[sorted_i[-1]]
        score_2 = scores[sorted_i[-2]]
        if score_2 < 0.01 or score_2 < 0.5 * score_1:
            return max_entry_1[0], max_entry_1[1], max_entry_1[3], score_1 * 0.5

        # TODO: verify the following three lines have negligible speed impact
        d = np.linalg.norm(max_entry_1[:2] - max_entry_2[:2])
        if d > max_entry_1[3] / 2.0:
            return max_entry_1[0], max_entry_1[1], max_entry_1[3], score_1 * 0.5

        return (
            (score_1 * max_entry_1[0] + score_2 * max_entry_2[0]) / (score_1 + score_2),
            (score_1 * max_entry_1[1] + score_2 * max_entry_2[1]) / (score_1 + score_2),
            (score_1 * max_entry_1[3] + score_2 * max_entry_2[3]) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        )
    def connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):

        caf_i, forward = self.by_source[start_i][end_i]
        caf_f, caf_b = caf_scored.directed(caf_i, forward)
        xyv = ann.data[start_i]
        xy_scale_s = max(0.0, ann.joint_scales[start_i])

        new_xysv = self._grow_connection(xyv[:2], xy_scale_s, caf_f)
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        xy_scale_t = max(0.0, new_xysv[2])

        # reverse match
        if reverse_match:
            reverse_xyv = self._grow_connection(
                new_xysv[:2], xy_scale_t, caf_b)
            if reverse_xyv[2] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    def _grow(self, ann, caf_scored, *, reverse_match=True):
        ref_time = time.time()
        frontier = PriorityQueue()
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[caf_i]
                frontier.put((-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier.qsize():
                entry = frontier.get()
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, caf_scored, start_i, end_i, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.greedy:
                    return (-score, new_xysv, start_i, end_i)
                if self.confidence_scales is not None:
                    caf_i, _ = self.by_source[start_i][end_i]
                    score *= self.confidence_scales[caf_i]
                frontier.put((-score, new_xysv, start_i, end_i))

        # seeding the frontier
        for joint_i, v in enumerate(ann.data[:, 2]):
            if v == 0.0:
                continue
            add_to_frontier(joint_i)

        while True:
            entry = frontier_get()
            if entry is None:
                break

            _, new_xysv, jsi, jti = entry
            if ann.data[jti, 2] > 0.0:
                continue

            ann.data[jti, :2] = new_xysv[:2]
            ann.data[jti, 2] = new_xysv[3]
            ann.joint_scales[jti] = new_xysv[2]
            ann.decoding_order.append(
                (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
            add_to_frontier(jti)
            
            
         
class KeypointPainter:


    def __init__(self, *,
                 xy_scale=1.0, highlight=None, highlight_invisible=False,
                 linewidth=2, markersize=None,
                 color_connections=False,
                 solid_threshold=0.5):
        self.xy_scale = xy_scale
        self.highlight = highlight
        self.highlight_invisible = highlight_invisible
        self.linewidth = linewidth
        self.markersize = markersize
        if self.markersize is None:
            if color_connections:
                self.markersize = max(1, int(linewidth * 0.5))
            else:
                self.markersize = max(linewidth + 1, int(linewidth * 3.0))
        self.color_connections = color_connections
        self.solid_threshold = solid_threshold


    def _draw_skeleton(self, ax, x, y, v, *, skeleton, color=None, **kwargs):
        if not np.any(v > 0):
            return

        # connections
        lines, line_colors, line_styles = [], [], []
        for ci, (j1i, j2i) in enumerate(np.array(skeleton) - 1):
            c = color
            if self.color_connections:
                c = matplotlib.cm.get_cmap('tab20')(ci / len(skeleton))
            if v[j1i] > 0 and v[j2i] > 0:
                lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
                line_colors.append(c)
                if v[j1i] > self.solid_threshold and v[j2i] > self.solid_threshold:
                    line_styles.append('solid')
                else:
                    line_styles.append('dashed')
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            linewidths=kwargs.get('linewidth', self.linewidth),
            linestyles=kwargs.get('linestyle', line_styles),
            capstyle='round',
        ))

        # joints
        ax.scatter(
            x[v > 0.0], y[v > 0.0], s=self.markersize**2, marker='.',
            color='white' if self.color_connections else color,
            edgecolor='k' if self.highlight_invisible else None,
            zorder=2,
        )

        # highlight joints
        if self.highlight is not None:
            highlight_v = np.zeros_like(v)
            highlight_v[self.highlight] = 1
            highlight_v = np.logical_and(v, highlight_v)

            ax.scatter(
                x[highlight_v], y[highlight_v], s=self.markersize**2, marker='.',
                color='white' if self.color_connections else color,
                edgecolor='k' if self.highlight_invisible else None,
                zorder=2,
            )

    def keypoints(self, ax, keypoint_sets, *,
                  skeleton, scores=None, color=None, colors=None, texts=None):
        if keypoint_sets is None:
            return

        if color is None and colors is None:
            colors = range(len(keypoint_sets))

        for i, kps in enumerate(np.asarray(keypoint_sets)):
            assert kps.shape[1] == 3
            x = kps[:, 0] * self.xy_scale
            y = kps[:, 1] * self.xy_scale
            v = kps[:, 2]

            if colors is not None:
                color = colors[i]

            if isinstance(color, (int, np.integer)):
                color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

            self._draw_skeleton(ax, x, y, v, skeleton=skeleton, color=color)

    
    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None):
        for i, ann in enumerate(annotations):
            color = i
            if colors is not None:
                color = colors[i]
            elif hasattr(ann, 'id_'):
                color = ann.id_

            text = None
            text_is_score = False
            if texts is not None:
                text = texts[i]
            elif hasattr(ann, 'id_'):
                text = '{}'.format(ann.id_)
            elif ann.score():
                text = '{:.0%}'.format(ann.score())
                text_is_score = True

            subtext = None
            if subtexts is not None:
                subtext = subtexts[i]
            elif not text_is_score and ann.score():
                subtext = '{:.0%}'.format(ann.score())

            self.annotation(ax, ann, color=color, text=text, subtext=subtext)

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = 0
        if isinstance(color, (int, np.integer)):
            color = matplotlib.cm.get_cmap('tab20')((color % 20 + 0.05) / 20)

        kps = ann.data
        assert kps.shape[1] == 3
        x = kps[:, 0] * self.xy_scale
        y = kps[:, 1] * self.xy_scale
        v = kps[:, 2]

   

        skeleton = ann.skeleton
       
        self._draw_skeleton(ax, x, y, v, color=color, skeleton=skeleton)
        
        
        
from contextlib import contextmanager
@contextmanager
def image_canvas(image, fig_file=None, show=True, dpi_factor=1.0, fig_width=10.0, **kwargs):
    if plt is None:
        raise Exception('please install matplotlib')

    image = np.asarray(image)
    if 'figsize' not in kwargs:
        kwargs['figsize'] = (fig_width, fig_width * image.shape[0] / image.shape[1])

    fig = plt.figure(**kwargs)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    fig.add_axes(ax)
    ax.imshow(image)

    yield ax

    if fig_file:
        fig.savefig(fig_file, dpi=image.shape[1] / kwargs['figsize'][0] * dpi_factor)
    if show:
        plt.show()
    plt.close(fig)

