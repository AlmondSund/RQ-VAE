"""Unit tests for the deterministic FWHT PSD codec."""

from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path

import numpy as np

from neural_compression.fwht_codec import (
    CRC32_SIZE_BYTES,
    DatasetConfig,
    EncodedFWHTPayload,
    FWHTCodecConfig,
    FWHTTransportConfig,
    HEADER_SIZE_BYTES,
    MAX_LENGTH_VALUE,
    PsdFrame,
    QuantizationState,
    compute_block_layout,
    compute_component_metrics,
    compute_frame_metrics,
    compute_standardization_stats,
    decode_fwht_frame,
    deserialize_packet,
    deserialize_payload,
    decimate_psd,
    encode_fwht_frame,
    estimate_payload_bits,
    fwht_orthonormal,
    make_frequency_axis_hz,
    materialize_sparse_coefficients,
    quantize_symmetric_uniform,
    required_index_bits,
    SERIALIZATION_VERSION,
    serialize_payload,
    total_occupied_bandwidth_hz,
)


def make_synthetic_frame(
    psd_db: np.ndarray,  # Synthetic PSD values on a uniform grid [dB]
) -> PsdFrame:
    """Build a minimal PSD frame suitable for codec and metric tests."""
    frequencies_hz = make_frequency_axis_hz(88e6, 108e6, psd_db.size)
    return PsdFrame(
        source_name="synthetic",
        psd_db=psd_db.astype(np.float64),
        frequencies_hz=frequencies_hz,
        timestamp_ms=0,
        start_freq_hz=88e6,
        end_freq_hz=108e6,
    )


class FWHTKernelTests(unittest.TestCase):
    """Tests for the mathematical FWHT kernel."""

    def test_fwht_is_orthonormal_and_self_inverse(self) -> None:
        """Applying the orthonormal FWHT twice must recover the input."""
        values = np.array([0.5, -1.0, 2.0, 3.5, -0.75, 1.25, 0.0, -2.5])
        transformed = fwht_orthonormal(values)
        recovered = fwht_orthonormal(transformed)

        np.testing.assert_allclose(recovered, values, atol=1e-12)
        self.assertAlmostEqual(
            float(np.linalg.norm(transformed)),
            float(np.linalg.norm(values)),
            places=12,
        )


class FWHTPayloadTests(unittest.TestCase):
    """Tests for payload-only reconstruction and rate accounting."""

    def test_packet_roundtrip_uses_quantized_side_information(self) -> None:
        """Serialized packets must preserve the quantized payload and decode correctly."""
        frame = make_synthetic_frame(
            np.array(
                [
                    -70.0,
                    -69.5,
                    -68.0,
                    -40.0,
                    -39.0,
                    -67.5,
                    -68.5,
                    -69.0,
                    -70.5,
                    -69.8,
                    -68.7,
                ]
            )
        )
        codec_config = FWHTCodecConfig(
            decimation_factor=3,
            retained_coefficients=4,
            quantization_bits=7,
            aggregation_domain="db",
        )

        payload, diagnostics = encode_fwht_frame(frame, codec_config)
        packet = serialize_payload(payload, codec_config)
        decoded_packet = deserialize_packet(packet)
        deserialized_payload = decoded_packet.payload
        reconstructed_psd_db = decode_fwht_frame(packet)

        decimated_psd_db = decimate_psd(
            frame.psd_db,
            factor=codec_config.decimation_factor,
            aggregation_domain=codec_config.aggregation_domain,
        )
        expected_stats = compute_standardization_stats(
            decimated_psd_db,
            original_length=frame.psd_db.size,
        )
        expected_scale = float(
            np.float32(
                np.max(np.abs(diagnostics.dense_coefficients[payload.retained_indices]))
            )
        )

        np.testing.assert_array_equal(
            deserialized_payload.retained_indices,
            payload.retained_indices,
        )
        np.testing.assert_array_equal(
            deserialized_payload.quantization.quantized_levels,
            payload.quantization.quantized_levels,
        )
        self.assertEqual(
            deserialized_payload.stats.mean_level, expected_stats.mean_level
        )
        self.assertEqual(deserialized_payload.stats.std_level, expected_stats.std_level)
        self.assertEqual(
            deserialized_payload.quantization.coefficient_scale,
            expected_scale,
        )
        self.assertEqual(decoded_packet.serialization_version, SERIALIZATION_VERSION)
        self.assertEqual(
            decoded_packet.transport_config,
            FWHTTransportConfig(
                decimation_factor=codec_config.decimation_factor,
                quantization_bits=codec_config.quantization_bits,
                nonlinear_map=codec_config.nonlinear_map,
                aggregation_domain=codec_config.aggregation_domain,
            ),
        )
        self.assertEqual(reconstructed_psd_db.shape[0], frame.psd_db.size)

    def test_decoder_rejects_duplicate_indices(self) -> None:
        """The payload validator must reject duplicated retained coefficient indices."""
        frame = make_synthetic_frame(np.linspace(-75.0, -35.0, 8))
        codec_config = FWHTCodecConfig(retained_coefficients=3, quantization_bits=6)
        payload, _ = encode_fwht_frame(frame, codec_config)
        bad_payload = EncodedFWHTPayload(
            retained_indices=np.array([1, 1, 3], dtype=np.int32),
            quantization=QuantizationState(
                quantized_levels=np.array([1, 2, 3], dtype=np.int32),
                coefficient_scale=payload.quantization.coefficient_scale,
            ),
            stats=payload.stats,
        )

        with self.assertRaises(ValueError):
            materialize_sparse_coefficients(bad_payload, codec_config)

    def test_payload_bit_estimate_matches_the_serialized_packet_length(self) -> None:
        """Rate accounting must equal the exact serialized packet length in bits."""
        frame = make_synthetic_frame(np.linspace(-80.0, -30.0, 16))
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=4,
            quantization_bits=8,
            aggregation_domain="db",
        )
        payload, _ = encode_fwht_frame(frame, codec_config)
        packet = serialize_payload(payload, codec_config)

        expected_bits = (
            HEADER_SIZE_BYTES * 8
            + 32
            + 8
            * int(
                np.ceil(
                    payload.retained_indices.size
                    * required_index_bits(payload.stats.padded_length)
                    / 8
                )
            )
            + 8
            * int(
                np.ceil(
                    payload.retained_indices.size * codec_config.quantization_bits / 8
                )
            )
            + CRC32_SIZE_BYTES * 8
        )
        self.assertEqual(
            estimate_payload_bits(payload, codec_config),
            expected_bits,
        )
        self.assertEqual(estimate_payload_bits(payload, codec_config), len(packet) * 8)

    def test_deserializer_rejects_corrupted_magic(self) -> None:
        """Malformed packets must fail before decode when the transport header is corrupted."""
        frame = make_synthetic_frame(np.linspace(-80.0, -30.0, 16))
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=4,
            quantization_bits=8,
        )
        payload, _ = encode_fwht_frame(frame, codec_config)
        corrupted_packet = bytearray(serialize_payload(payload, codec_config))
        corrupted_packet[0:4] = b"NOPE"

        with self.assertRaises(ValueError):
            deserialize_payload(corrupted_packet, codec_config)

    def test_deserializer_rejects_crc_corruption(self) -> None:
        """A structurally valid packet with a flipped payload bit must fail the CRC32 check."""
        frame = make_synthetic_frame(np.linspace(-80.0, -30.0, 16))
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=4,
            quantization_bits=8,
        )
        payload, _ = encode_fwht_frame(frame, codec_config)
        corrupted_packet = bytearray(serialize_payload(payload, codec_config))
        corrupted_packet[HEADER_SIZE_BYTES] ^= 0x01

        with self.assertRaises(ValueError):
            deserialize_payload(corrupted_packet)

    def test_one_bit_quantizer_uses_true_binary_levels(self) -> None:
        """The 1-bit quantizer contract must remain binary rather than ternary."""
        quantization = quantize_symmetric_uniform(
            np.array([-3.0, -0.25, 0.25, 2.0], dtype=np.float64),
            quantization_bits=1,
        )

        np.testing.assert_array_equal(
            quantization.quantized_levels,
            np.array([-1, -1, 1, 1], dtype=np.int32),
        )

    def test_serializer_rejects_zero_level_for_one_bit_quantizer(self) -> None:
        """A true 1-bit packet must not silently accept a ternary zero code."""
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=2,
            quantization_bits=1,
        )
        bad_payload = EncodedFWHTPayload(
            retained_indices=np.array([0, 1], dtype=np.int32),
            quantization=QuantizationState(
                quantized_levels=np.array([-1, 0], dtype=np.int32),
                coefficient_scale=1.0,
            ),
            stats=compute_standardization_stats(
                np.array([-75.0, -74.0], dtype=np.float64),
                original_length=4,
            ),
        )

        with self.assertRaises(ValueError):
            serialize_payload(bad_payload, codec_config)

    def test_packet_can_be_decoded_without_external_codec_config(self) -> None:
        """The packet header must carry enough transport information for standalone decode."""
        frame = make_synthetic_frame(np.linspace(-80.0, -35.0, 17))
        codec_config = FWHTCodecConfig(
            decimation_factor=3,
            retained_coefficients=5,
            quantization_bits=7,
            nonlinear_map="asinh",
            aggregation_domain="db",
        )

        payload, _ = encode_fwht_frame(frame, codec_config)
        packet = serialize_payload(payload, codec_config)
        decoded_packet = deserialize_packet(packet)
        reconstructed_from_packet = decode_fwht_frame(packet)
        reconstructed_from_payload = decode_fwht_frame(payload, codec_config)

        self.assertEqual(
            decoded_packet.transport_config,
            FWHTTransportConfig(
                decimation_factor=3,
                quantization_bits=7,
                nonlinear_map="asinh",
                aggregation_domain="db",
            ),
        )
        np.testing.assert_allclose(
            reconstructed_from_packet,
            reconstructed_from_payload,
            atol=1e-12,
        )

    def test_packet_roundtrip_is_stable_across_randomized_codec_settings(self) -> None:
        """Randomized packet roundtrips should preserve payload metadata and decode shape."""
        rng = np.random.default_rng(1234)

        for _ in range(25):
            frame_length = int(rng.integers(5, 33))
            decimation_factor = int(rng.integers(1, 5))
            quantization_bits = int(rng.choice(np.array([1, 3, 7, 11], dtype=np.int64)))
            retained_coefficients = int(rng.integers(0, 9))
            nonlinear_map = "asinh" if bool(rng.integers(0, 2)) else "identity"
            aggregation_domain = "linear_power" if bool(rng.integers(0, 2)) else "db"
            psd_db = rng.normal(loc=-70.0, scale=8.0, size=frame_length)
            frame = make_synthetic_frame(psd_db)
            codec_config = FWHTCodecConfig(
                decimation_factor=decimation_factor,
                retained_coefficients=retained_coefficients,
                quantization_bits=quantization_bits,
                nonlinear_map=nonlinear_map,
                aggregation_domain=aggregation_domain,
            )

            payload, _ = encode_fwht_frame(frame, codec_config)
            packet = serialize_payload(payload, codec_config)
            decoded_packet = deserialize_packet(packet)
            reconstructed = decode_fwht_frame(packet)

            np.testing.assert_array_equal(
                decoded_packet.payload.retained_indices,
                payload.retained_indices,
            )
            np.testing.assert_array_equal(
                decoded_packet.payload.quantization.quantized_levels,
                payload.quantization.quantized_levels,
            )
            self.assertEqual(reconstructed.shape[0], frame_length)

    def test_packet_crc_rejects_deterministic_bit_flips(self) -> None:
        """Single-byte corruptions across the packet body should be rejected deterministically."""
        frame = make_synthetic_frame(np.linspace(-82.0, -28.0, 19))
        codec_config = FWHTCodecConfig(
            decimation_factor=3,
            retained_coefficients=6,
            quantization_bits=7,
        )
        payload, _ = encode_fwht_frame(frame, codec_config)
        packet = serialize_payload(payload, codec_config)

        for byte_index in range(4, len(packet) - CRC32_SIZE_BYTES):
            corrupted_packet = bytearray(packet)
            corrupted_packet[byte_index] ^= 0x01
            with self.assertRaises(ValueError):
                deserialize_payload(corrupted_packet)

    def test_serialized_packet_matches_the_golden_v2_fixture(self) -> None:
        """The v2 packet bytes must remain stable for a fixed payload fixture."""
        payload = EncodedFWHTPayload(
            retained_indices=np.array([0, 1, 3], dtype=np.int32),
            quantization=QuantizationState(
                quantized_levels=np.array([-1, 0, 1], dtype=np.int32),
                coefficient_scale=np.float32(3.5).item(),
            ),
            stats=compute_standardization_stats(
                np.array([-70.0, -68.0, -66.0, -64.0], dtype=np.float64),
                original_length=7,
            ),
        )
        codec_config = FWHTCodecConfig(
            decimation_factor=2,
            retained_coefficients=3,
            quantization_bits=3,
            nonlinear_map="identity",
            aggregation_domain="db",
        )
        packet = serialize_payload(payload, codec_config)

        self.assertEqual(
            packet.hex(),
            "46574854020100000302000700040004000300000086c2bd1b0f4000006040341a0124c8ba88",
        )

    def test_serializer_rejects_metadata_overflow(self) -> None:
        """Metadata that does not fit the fixed-width transport fields must be rejected."""
        frame = make_synthetic_frame(np.linspace(-75.0, -35.0, 8))
        codec_config = FWHTCodecConfig(retained_coefficients=3, quantization_bits=6)
        payload, _ = encode_fwht_frame(frame, codec_config)
        overflowing_payload = EncodedFWHTPayload(
            retained_indices=payload.retained_indices,
            quantization=payload.quantization,
            stats=replace(payload.stats, original_length=MAX_LENGTH_VALUE + 1),
        )

        with self.assertRaises(ValueError):
            serialize_payload(overflowing_payload, codec_config)

    def test_serializer_rejects_quantized_levels_outside_bit_depth(self) -> None:
        """Quantized levels outside the configured signed range must be rejected."""
        frame = make_synthetic_frame(np.linspace(-75.0, -35.0, 8))
        codec_config = FWHTCodecConfig(retained_coefficients=3, quantization_bits=6)
        payload, _ = encode_fwht_frame(frame, codec_config)
        bad_payload = EncodedFWHTPayload(
            retained_indices=payload.retained_indices,
            quantization=QuantizationState(
                quantized_levels=np.array([40, 1, -1], dtype=np.int32),
                coefficient_scale=payload.quantization.coefficient_scale,
            ),
            stats=payload.stats,
        )

        with self.assertRaises(ValueError):
            serialize_payload(bad_payload, codec_config)


class SensingMetricTests(unittest.TestCase):
    """Tests for occupancy and connected-component metrics."""

    def test_occupancy_metric_uses_independent_noise_floor_estimates(self) -> None:
        """A constant dB shift should not change occupancy F1 when the same rule is applied independently."""
        reference_psd_db = np.array(
            [-80.0, -80.0, -60.0, -58.0, -80.0, -80.0, -79.0, -80.0]
        )
        reconstructed_psd_db = reference_psd_db + 10.0
        frame = make_synthetic_frame(reference_psd_db)
        dataset_config = DatasetConfig(dataset_dir=Path("."))

        metrics = compute_frame_metrics(
            frame=frame,
            reconstructed_psd_db=reconstructed_psd_db,
            dataset_config=dataset_config,
            payload_bits=128,
            codec_config=FWHTCodecConfig(),
        )

        self.assertAlmostEqual(metrics.occupancy_f1, 1.0, places=12)

    def test_connected_component_bandwidth_sums_each_region_width(self) -> None:
        """Bandwidth must be the sum of connected occupied widths, not the span over the full band."""
        frequencies_hz = np.arange(10, dtype=np.float64) + 0.5
        occupancy = np.array(
            [False, True, True, False, False, False, True, True, False, False]
        )

        self.assertEqual(total_occupied_bandwidth_hz(frequencies_hz, occupancy), 4.0)

    def test_component_matching_catches_station_merging_that_total_width_misses(
        self,
    ) -> None:
        """Station-aware metrics must penalize merged occupied regions even at equal total width."""
        frequencies_hz = np.arange(10, dtype=np.float64) + 0.5
        reference_occupancy = np.array(
            [False, True, True, False, False, False, True, True, False, False]
        )
        reconstructed_occupancy = np.array(
            [False, True, True, True, True, False, False, False, False, False]
        )

        component_metrics = compute_component_metrics(
            frequencies_hz,
            reference_occupancy,
            reconstructed_occupancy,
        )

        self.assertEqual(
            total_occupied_bandwidth_hz(frequencies_hz, reference_occupancy),
            total_occupied_bandwidth_hz(frequencies_hz, reconstructed_occupancy),
        )
        self.assertLess(component_metrics.component_f1, 1.0)
        self.assertEqual(component_metrics.missed_component_count, 1)
        self.assertEqual(component_metrics.hallucinated_component_count, 0)

    def test_non_uniform_frequency_grid_is_rejected(self) -> None:
        """The codec must fail fast when a PSD frame violates the uniform-grid assumption."""
        psd_db = np.array([-80.0, -79.0, -78.0], dtype=np.float64)
        frequencies_hz = np.array([88.5e6, 89.5e6, 90.7e6], dtype=np.float64)

        with self.assertRaises(ValueError):
            PsdFrame(
                source_name="nonuniform",
                psd_db=psd_db,
                frequencies_hz=frequencies_hz,
                timestamp_ms=0,
                start_freq_hz=88e6,
                end_freq_hz=91e6,
            )

    def test_block_layout_matches_the_last_partial_decimation_block(self) -> None:
        """The matched resampling geometry must preserve the shorter last block."""
        block_starts, block_lengths = compute_block_layout(num_bins=11, factor=3)

        np.testing.assert_array_equal(block_starts, np.array([0, 3, 6, 9]))
        np.testing.assert_array_equal(block_lengths, np.array([3, 3, 3, 2]))


if __name__ == "__main__":
    unittest.main()
