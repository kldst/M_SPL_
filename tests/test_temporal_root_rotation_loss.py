"""Unit tests for supervised temporal root-rotation motion loss."""

import unittest

import torch

from training.loss_smpl import compute_temporal_smpl_smoothness


class TemporalRootRotationLossTest(unittest.TestCase):
    @staticmethod
    def _inputs(pred_root, gt_root, valid=None):
        B, T, P, _ = pred_root.shape
        pred_pose = torch.zeros(B, T, P, 72, dtype=pred_root.dtype)
        gt_pose = torch.zeros_like(pred_pose)
        pred_pose[..., :3] = pred_root
        gt_pose[..., :3] = gt_root
        pred_pose = pred_pose.reshape(B * T, P, 72).requires_grad_()
        gt_pose = gt_pose.reshape(B * T, P, 72)
        if valid is None:
            valid = torch.ones(B, T, P, dtype=pred_root.dtype)
        predictions = {"smpl_pose": pred_pose}
        batch = {
            "smpl_pose": gt_pose,
            "has_smpl": valid.reshape(B * T, P),
            "temporal_shape": torch.tensor([B, T]),
        }
        return predictions, batch

    def test_matching_gt_motion_is_zero_even_during_turn(self):
        root = torch.zeros(1, 3, 2, 3)
        root[0, :, 0, 1] = torch.tensor([0.0, 0.3, 0.8])
        root[0, :, 1, 2] = torch.tensor([0.2, -0.1, -0.5])
        predictions, batch = self._inputs(root, root)

        losses = compute_temporal_smpl_smoothness(predictions, batch)

        self.assertLess(
            float(losses["loss_smpl_temporal_root_rotation"]), 1e-6
        )

    def test_prediction_only_root_jump_is_penalized_and_differentiable(self):
        gt_root = torch.zeros(1, 3, 1, 3)
        gt_root[0, :, 0, 1] = torch.tensor([0.0, 0.1, 0.2])
        pred_root = gt_root.clone()
        pred_root[0, 2, 0, 1] = 0.9
        predictions, batch = self._inputs(pred_root, gt_root)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch
        )["loss_smpl_temporal_root_rotation"]

        # The 0.7-rad final-step error is averaged with the exact first step.
        self.assertGreater(float(loss), 0.3)
        loss.backward()
        self.assertIsNotNone(predictions["smpl_pose"].grad)
        self.assertTrue(torch.isfinite(predictions["smpl_pose"].grad).all())

    def test_invalid_person_is_excluded(self):
        gt_root = torch.zeros(1, 3, 2, 3)
        pred_root = gt_root.clone()
        pred_root[0, 2, 1, 1] = 1.5
        valid = torch.ones(1, 3, 2)
        valid[..., 1] = 0.0
        predictions, batch = self._inputs(pred_root, gt_root, valid)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch
        )["loss_smpl_temporal_root_rotation"]

        self.assertLess(float(loss), 1e-6)

    def test_second_order_matching_motion_is_zero(self):
        root = torch.zeros(1, 4, 1, 3)
        root[0, :, 0, 1] = torch.tensor([0.0, 0.2, 0.5, 0.9])
        predictions, batch = self._inputs(root, root)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch, root_rotation_order=2
        )["loss_smpl_temporal_root_rotation"]

        self.assertLess(float(loss), 1e-6)


if __name__ == "__main__":
    unittest.main()


class TemporalGtDeltaLossTest(unittest.TestCase):
    """pose_use_gt_delta / mesh_translate_use_gt_delta switch the finite
    difference target from 0 to GT's own finite difference."""

    @staticmethod
    def _inputs(pred_pose_BTP72, gt_pose_BTP72, pred_trans=None, gt_trans=None):
        B, T, P, _ = pred_pose_BTP72.shape
        predictions = {
            "smpl_pose": pred_pose_BTP72.reshape(B * T, P, 72).clone().requires_grad_()
        }
        batch = {
            "smpl_pose": gt_pose_BTP72.reshape(B * T, P, 72),
            "has_smpl": torch.ones(B * T, P),
            "temporal_shape": torch.tensor([B, T]),
        }
        if pred_trans is not None:
            predictions["mesh_translate"] = (
                pred_trans.reshape(B * T, P, 3).clone().requires_grad_()
            )
            batch["mesh_translate"] = gt_trans.reshape(B * T, P, 3)
        return predictions, batch

    def test_pose_gt_delta_is_zero_when_prediction_tracks_gt_motion(self):
        # A genuinely articulating (constant-acceleration) elbow: the plain
        # order-2 smoothness prior penalizes it, GT-delta supervision does not.
        pose = torch.zeros(1, 3, 1, 72)
        pose[0, :, 0, 12] = torch.tensor([0.0, 0.2, 0.8])
        predictions, batch = self._inputs(pose, pose)

        smooth = compute_temporal_smpl_smoothness(predictions, batch)
        gt_delta = compute_temporal_smpl_smoothness(
            predictions, batch, pose_use_gt_delta=True
        )

        self.assertGreater(float(smooth["loss_smpl_temporal_pose"]), 1e-4)
        self.assertLess(float(gt_delta["loss_smpl_temporal_pose"]), 1e-6)

    def test_pose_gt_delta_still_penalizes_prediction_only_jitter(self):
        gt_pose = torch.zeros(1, 3, 1, 72)
        gt_pose[0, :, 0, 12] = torch.tensor([0.0, 0.2, 0.4])
        pred_pose = gt_pose.clone()
        pred_pose[0, 1, 0, 12] = 1.1  # off-trajectory middle frame
        predictions, batch = self._inputs(pred_pose, gt_pose)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch, pose_use_gt_delta=True
        )["loss_smpl_temporal_pose"]

        self.assertGreater(float(loss), 1e-3)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions["smpl_pose"].grad).all())

    def test_mesh_translate_gt_delta_ignores_real_acceleration(self):
        pose = torch.zeros(1, 3, 1, 72)
        trans = torch.zeros(1, 3, 1, 3)
        trans[0, :, 0, 0] = torch.tensor([0.0, 0.1, 0.4])  # accelerating walk
        predictions, batch = self._inputs(pose, pose, trans, trans)

        smooth = compute_temporal_smpl_smoothness(predictions, batch)
        gt_delta = compute_temporal_smpl_smoothness(
            predictions, batch, mesh_translate_use_gt_delta=True
        )

        self.assertAlmostEqual(
            float(smooth["loss_smpl_temporal_mesh_translate"]), 0.2 / 3, places=5
        )
        self.assertLess(
            float(gt_delta["loss_smpl_temporal_mesh_translate"]), 1e-7
        )

    def test_mesh_translate_gt_delta_penalizes_wrong_motion(self):
        pose = torch.zeros(1, 3, 1, 72)
        gt_trans = torch.zeros(1, 3, 1, 3)
        gt_trans[0, :, 0, 0] = torch.tensor([0.0, 0.1, 0.2])
        pred_trans = gt_trans.clone()
        pred_trans[0, 1, 0, 0] = 0.7
        predictions, batch = self._inputs(pose, pose, pred_trans, gt_trans)

        loss = compute_temporal_smpl_smoothness(
            predictions, batch, mesh_translate_use_gt_delta=True
        )["loss_smpl_temporal_mesh_translate"]

        self.assertGreater(float(loss), 0.3)
        loss.backward()
        self.assertTrue(
            torch.isfinite(predictions["mesh_translate"].grad).all()
        )


