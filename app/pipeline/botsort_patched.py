"""BoT-SORT subclass with a strict appearance gate.

Stock boxmot BoT-SORT combines IoU and ReID via ``dists = np.minimum(iou, emb)``
in its first association — so a track whose IoU happens to overlap a different
person can be hijacked by IoU alone, even when the ReID embedding clearly says
"different person". This subclass adds **one extra mask**: when
``emb_dist > appearance_thresh`` we also zero out the IoU match (set to 1.0 =
max distance), making appearance a hard veto.

Nothing else in BoT-SORT changes — Kalman, CMC, second pass, lifecycle are all
identical to upstream.
"""
from __future__ import annotations

import numpy as np
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.basetrack import TrackState
from boxmot.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)


class BotSortStrictAppearance(BotSort):
    """BoT-SORT where ReID can veto IoU-only matches.

    Behaviour matches upstream BoT-SORT exactly when there are no
    appearance-mismatched IoU overlaps. The change only kicks in when a track
    has a Kalman-predicted bbox that overlaps a *different person's* detection.
    """

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        from boxmot.trackers.botsort.botsort_track import STrack

        STrack.multi_predict(strack_pool)
        self._apply_camera_motion_compensation(
            dets, img, strack_pool, unconfirmed
        )

        ious_dists = iou_distance(strack_pool, detections, is_obb=self.is_obb)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)
            appearance_bad = emb_dists > self.appearance_thresh
            # === The only deviation from upstream BotSort: appearance is now a
            # hard veto, so an IoU-only match with a wrong-appearance detection
            # cannot hijack the track.
            ious_dists[appearance_bad] = 1.0
            emb_dists[appearance_bad] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        return matches, u_track, u_detection

    def _handle_unconfirmed_tracks(
        self,
        u_detection,
        detections,
        activated_stracks,
        removed_stracks,
        unconfirmed,
    ):
        """Mirror of upstream `_handle_unconfirmed_tracks` with the same
        strict appearance veto."""
        detections = [detections[i] for i in u_detection]

        ious_dists = iou_distance(unconfirmed, detections, is_obb=self.is_obb)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            appearance_bad = emb_dists > self.appearance_thresh
            ious_dists[appearance_bad] = 1.0
            emb_dists[appearance_bad] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=0.7
        )

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return matches, u_unconfirmed, u_detection
