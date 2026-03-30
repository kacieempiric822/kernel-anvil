"""Tests for the C header code generation module."""

import pytest

from kernel_anvil.codegen import (
    BUCKET_BOUNDARIES,
    DEFAULT_CONFIG,
    GGML_TYPE_COUNT,
    GGML_TYPE_MAP,
    NUM_BUCKETS,
    ShapeConfig,
    bucket_index,
    build_config_tables,
    generate_config_header,
)


# ---------------------------------------------------------------------------
# bucket_index tests
# ---------------------------------------------------------------------------


class TestBucketIndex:
    def test_below_first_boundary(self):
        assert bucket_index(1) == 0
        assert bucket_index(64) == 0
        assert bucket_index(128) == 0

    def test_exact_boundaries(self):
        for i, boundary in enumerate(BUCKET_BOUNDARIES):
            assert bucket_index(boundary) == i

    def test_just_above_boundaries(self):
        assert bucket_index(129) == 1
        assert bucket_index(1025) == 2
        assert bucket_index(4097) == 3
        assert bucket_index(16385) == 4

    def test_large_value(self):
        assert bucket_index(151936) == len(BUCKET_BOUNDARIES)

    def test_value_one(self):
        assert bucket_index(1) == 0

    def test_between_boundaries(self):
        assert bucket_index(500) == 1
        assert bucket_index(2048) == 2
        assert bucket_index(8192) == 3


# ---------------------------------------------------------------------------
# ShapeConfig tests
# ---------------------------------------------------------------------------


class TestShapeConfig:
    def test_frozen(self):
        cfg = ShapeConfig(nwarps=4, rows_per_block=2)
        with pytest.raises(AttributeError):
            cfg.nwarps = 8

    def test_fields(self):
        cfg = ShapeConfig(nwarps=8, rows_per_block=4)
        assert cfg.nwarps == 8
        assert cfg.rows_per_block == 4

    def test_equality(self):
        a = ShapeConfig(nwarps=4, rows_per_block=1)
        b = ShapeConfig(nwarps=4, rows_per_block=1)
        assert a == b

    def test_default_config(self):
        assert DEFAULT_CONFIG.nwarps == 4
        assert DEFAULT_CONFIG.rows_per_block == 1


# ---------------------------------------------------------------------------
# build_config_tables tests
# ---------------------------------------------------------------------------


class TestBuildConfigTables:
    def test_empty_configs(self):
        tables = build_config_tables({})
        assert tables == {}

    def test_single_entry(self):
        configs = {("Q4_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 2}}
        tables = build_config_tables(configs)
        assert "Q4_K" in tables
        table = tables["Q4_K"]
        assert table[2][2] == ShapeConfig(nwarps=8, rows_per_block=2)

    def test_defaults_fill_empty_buckets(self):
        configs = {("Q4_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 2}}
        tables = build_config_tables(configs)
        table = tables["Q4_K"]
        for ni in range(NUM_BUCKETS):
            for ki in range(NUM_BUCKETS):
                if (ni, ki) != (2, 2):
                    assert table[ni][ki] == DEFAULT_CONFIG

    def test_multiple_quant_types(self):
        configs = {
            ("Q4_K", 4096, 4096): {"nwarps": 4, "rows_per_block": 2},
            ("Q6_K", 4096, 4096): {"nwarps": 8, "rows_per_block": 4},
        }
        tables = build_config_tables(configs)
        assert "Q4_K" in tables
        assert "Q6_K" in tables
        assert tables["Q4_K"][2][2] == ShapeConfig(nwarps=4, rows_per_block=2)
        assert tables["Q6_K"][2][2] == ShapeConfig(nwarps=8, rows_per_block=4)

    def test_table_dimensions(self):
        configs = {("Q4_K", 64, 64): {"nwarps": 2, "rows_per_block": 1}}
        tables = build_config_tables(configs)
        table = tables["Q4_K"]
        assert len(table) == NUM_BUCKETS
        for row in table:
            assert len(row) == NUM_BUCKETS

    def test_large_dimensions_go_to_last_bucket(self):
        configs = {("Q4_K", 151936, 4096): {"nwarps": 8, "rows_per_block": 4}}
        tables = build_config_tables(configs)
        assert tables["Q4_K"][4][2] == ShapeConfig(nwarps=8, rows_per_block=4)

    def test_multiple_entries_same_bucket(self):
        configs = {
            ("Q4_K", 3000, 3000): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 4000, 4000): {"nwarps": 8, "rows_per_block": 4},
        }
        tables = build_config_tables(configs)
        cfg = tables["Q4_K"][2][2]
        assert cfg.nwarps in (2, 8)


# ---------------------------------------------------------------------------
# generate_config_header tests (GGML_TYPE_COUNT indexed format)
# ---------------------------------------------------------------------------


class TestGenerateConfigHeader:
    @pytest.fixture
    def sample_configs(self):
        return {
            ("Q4_K", 4096, 4096): {"nwarps": 4, "rows_per_block": 2},
            ("Q4_K", 4096, 1024): {"nwarps": 2, "rows_per_block": 1},
            ("Q4_K", 12288, 4096): {"nwarps": 8, "rows_per_block": 4},
            ("Q6_K", 4096, 1024): {"nwarps": 2, "rows_per_block": 1},
        }

    def test_header_contains_pragma_once(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "#pragma once" in header

    def test_header_contains_struct(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "struct smithy_shape_config {" in header
        assert "int nwarps;" in header
        assert "int rows_per_block;" in header

    def test_header_contains_gpu_name(self, sample_configs):
        header = generate_config_header(sample_configs, gpu_name="gfx1100 (7900 XTX)")
        assert "gfx1100 (7900 XTX)" in header

    def test_header_contains_model_name(self, sample_configs):
        header = generate_config_header(sample_configs, model_name="Qwen3-8B-Q4_K_M")
        assert "Qwen3-8B-Q4_K_M" in header

    def test_header_contains_bucket_enum(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "smithy_get_bucket" in header
        assert "SMITHY_NUM_BUCKETS" in header
        for boundary in BUCKET_BOUNDARIES:
            assert str(boundary) in header

    def test_header_has_ggml_type_count_array(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert f"smithy_configs[{GGML_TYPE_COUNT}]" in header
        assert "SMITHY_CONFIG_TABLE" in header

    def test_header_has_type_comments(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "GGML_TYPE_Q4_K" in header
        assert "GGML_TYPE_Q6_K" in header

    def test_specific_config_values_in_output(self, sample_configs):
        header = generate_config_header(sample_configs)
        # Q4_K (4096, 1024) -> nwarps=2, rows_per_block=1 -> N bucket 2, K bucket 1
        assert "{ 2, 1}" in header
        # Q4_K (12288, 4096) -> nwarps=8, rows_per_block=4 -> N bucket 3, K bucket 2
        assert "{ 8, 4}" in header

    def test_empty_configs(self):
        header = generate_config_header({})
        assert "#pragma once" in header
        assert "struct smithy_shape_config {" in header

    def test_single_quant_type(self):
        configs = {("Q8_0", 4096, 4096): {"nwarps": 4, "rows_per_block": 2}}
        header = generate_config_header(configs)
        # Q8_0 is ggml_type 8, should have data at index [8]
        assert f"[{GGML_TYPE_MAP['Q8_0']}] = GGML_TYPE_Q8_0" in header

    def test_header_is_valid_looking_c(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert header.count("{") == header.count("}")

    def test_unfilled_buckets_are_zero(self, sample_configs):
        header = generate_config_header(sample_configs)
        # Most entries should be {0, 0} (no override)
        assert header.count("{ 0, 0}") > 10

    def test_unused_types_are_empty(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "unused" in header

    def test_ggml_type_count_dimensions(self, sample_configs):
        header = generate_config_header(sample_configs)
        assert "[SMITHY_NUM_BUCKETS][SMITHY_NUM_BUCKETS]" in header


# ---------------------------------------------------------------------------
# Integration: GGUF -> codegen pipeline
# ---------------------------------------------------------------------------


class TestGGUFToCodegen:
    def test_gguf_shapes_to_configs(self):
        unique_shapes = {
            ("Q4_K", 4096, 1024): 54,
            ("Q4_K", 4096, 4096): 72,
            ("Q4_K", 4096, 12288): 72,
            ("Q4_K", 4096, 151936): 1,
            ("Q4_K", 12288, 4096): 18,
            ("Q6_K", 4096, 1024): 18,
            ("Q6_K", 4096, 151936): 1,
            ("Q6_K", 12288, 4096): 18,
        }

        configs = {}
        for (qt, n, k), _count in unique_shapes.items():
            configs[(qt, n, k)] = {"nwarps": 4, "rows_per_block": 1}

        header = generate_config_header(
            configs,
            gpu_name="gfx1100 (7900 XTX)",
            model_name="Qwen3-8B-Q4_K_M",
        )

        assert "Qwen3-8B-Q4_K_M" in header
        assert "GGML_TYPE_Q4_K" in header
        assert "GGML_TYPE_Q6_K" in header
        assert "#pragma once" in header
        assert f"smithy_configs[{GGML_TYPE_COUNT}]" in header
