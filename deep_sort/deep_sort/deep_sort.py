import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        # REID 特征缓存：Governor 判定高负载时可跳过本帧重算，按 IoU 复用上一帧特征
        self._cached_boxes = []          # list of xywh (np.float)
        self._cached_features = np.array([])

    def update(self, bbox_xywh, confidences, ori_img, classes, bbox_3d=None, extract_reid=True):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img, extract_reid=extract_reid)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        
        detections = []
        for i, conf in enumerate(confidences):
            if conf > self.min_confidence:
                b3d = bbox_3d[i] if bbox_3d is not None else None
                detections.append(Detection(bbox_tlwh[i], conf, features[i], classes[i], bbox_3d=b3d))

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        #classes = np.array([d.classes])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            classes1 = track.cls
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id, classes1], dtype=np.int64))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img, extract_reid=True):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if not im_crops:
            return np.array([])

        if extract_reid or not self._cached_features.size:
            features = self.extractor(im_crops)
        else:
            # 高负载：复用上一帧已算好的特征，只有大位移/新目标才重算
            features = self._reuse_or_compute_features(bbox_xywh, im_crops)

        # 更新缓存（与输入框顺序一一对应）
        self._cached_boxes = [np.asarray(b, dtype=float).reshape(-1) for b in bbox_xywh]
        self._cached_features = features
        return features

    def _reuse_or_compute_features(self, bbox_xywh, im_crops):
        cached_boxes = self._cached_boxes
        if not cached_boxes or self._cached_features.size == 0:
            return self.extractor(im_crops)

        n = len(bbox_xywh)
        out = np.zeros((n, self._cached_features.shape[1]), dtype=self._cached_features.dtype)
        used = set()
        needs = []
        for i, box in enumerate(bbox_xywh):
            bx = np.asarray(box, dtype=float).reshape(-1)
            best_j, best_iou = -1, 0.35
            for j, cb in enumerate(cached_boxes):
                if j in used:
                    continue
                iou = self._iou_xywh(bx, cb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                out[i] = self._cached_features[best_j]
                used.add(best_j)
            else:
                needs.append(i)

        if needs:
            crops = [im_crops[i] for i in needs]
            feats = self.extractor(crops)
            for k, i in enumerate(needs):
                out[i] = feats[k]
        return out

    @staticmethod
    def _iou_xywh(a, b):
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        ax1, ay1 = a[0] - a[2] / 2.0, a[1] - a[3] / 2.0
        ax2, ay2 = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bx1, by1 = b[0] - b[2] / 2.0, b[1] - b[3] / 2.0
        bx2, by2 = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        area_a = max(a[2] * a[3], 1e-6)
        area_b = max(b[2] * b[3], 1e-6)
        return inter / (area_a + area_b - inter)


