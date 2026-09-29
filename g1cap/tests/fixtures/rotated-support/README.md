# Permitted sensor regression: rotated support ambiguity

Original head RGB-D and camera-time body encoders/derived up from campaign59, diversity-rotated-01,0.28s. No task geometry, evaluator poses, mass, contact or segmentation labels. Static calibration is permitted. Compressed arrays retain original values. Test refits the actual frontend from these measurements; known robot geometry must exclude forearm/thumb points before initial support fitting. Full source recording and diagnostic: runs/sensor-transfer-2026-09-19/diversity-rotated-01/.

The preceding0.24s sensor-estimated cuboid seeds the pose tracker; it is not a simulator object pose. Reinitializing the tracker cold at0.28s cannot see every required box-top edge, a separate limitation from the retained tracking failure being tested.

`hands.json` retains the same0.28s native14-position Dex3 packet. The acquisition geometry recovery regression reuses these recorded pixels/encoders with explicitly synthetic advancing timestamps to exercise estimator lifecycle and tool admission. It does not represent new physics or independently qualify camera-time motion.
