import torch

from medical_anomaly.supervision import (
    configure_supervision,
    empty_exposure_counts,
    supervision_mode,
    update_exposure_counts,
)


class _Generator:
    def forward(self, features, adapted, mask, labels):
        raise AssertionError("original generator should be replaced")


class _Model:
    def __init__(self):
        self.anomaly_generator = _Generator()


def test_supervision_mode_defaults_to_mixed():
    assert supervision_mode({}) == "mixed"


def test_real_only_duplicates_without_synthetic_noise():
    model = _Model()
    configure_supervision(model, "real_only")
    features = torch.randn(2, 3, 4, 4)
    adapted = torch.randn(2, 3, 4, 4)
    masks = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(-1, -1, 4, 4)
    labels = torch.tensor([0.0, 1.0])
    output_features, output_adapted, output_masks, output_labels = model.anomaly_generator.forward(
        features, adapted, masks, labels
    )
    assert torch.equal(output_features, torch.cat((features, features)))
    assert torch.equal(output_adapted, torch.cat((adapted, adapted)))
    assert torch.equal(output_masks, torch.cat((masks, masks)))
    assert torch.equal(output_labels, torch.cat((labels, labels)))


def test_exposure_counts_separate_real_and_synthetic_pixels():
    counts = empty_exposure_counts("mixed")
    labels = torch.tensor([0.0, 1.0])
    input_masks = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(-1, -1, 2, 2)
    output_masks = torch.cat((input_masks, input_masks)).clone()
    output_masks[0, :, 0, 0] = 1
    update_exposure_counts(counts, labels, input_masks, output_masks)
    assert counts["normal_image_exposures"] == 1
    assert counts["real_defect_image_exposures"] == 1
    assert counts["real_mask_positive_pixels"] == 4
    assert counts["synthetic_anomaly_exposures"] == 1
    assert counts["synthetic_mask_positive_pixels"] == 1
    assert counts["synthetic_noise_mask_positive_pixels"] == 1
