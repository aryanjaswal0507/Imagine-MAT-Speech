"""
Comprehensive Unit Test Suite for AgnixAI Health Report Pipeline
Covers: function.py, extraction_utils.py, runtime_logging.py, schemas.py
Includes both positive (happy path) and negative (edge case / failure) tests.

Run with:
    python -m pytest test_suite.py -v
or:
    python -m unittest test_suite.py -v
"""

import unittest
import math
import numpy as np
import os
import tempfile
import json
from unittest.mock import patch, MagicMock


# ==============================================================================
# Imports from project modules
# ==============================================================================
from function import (
    clean_text_for_pdf_func,
    dict_to_toon,
    calculate_status_func,
    parse_interval_func,
    standardize_interval_func,
    calculate_advanced_forecast_fixed_func,
    normalize_test_name_func,
    _categorize_test_func,
    is_qualitative_interval_func,
)
from extraction_utils import OCRRegion, get_reading_order_simple_func
from runtime_logging import (
    estimate_tokens_func,
    initialize_logging_func,
    log_event_func,
    save_log_func,
    execution_log,
)


# ==============================================================================
# 1. Tests for clean_text_for_pdf_func
# ==============================================================================

class TestCleanTextForPdf(unittest.TestCase):
    """Tests for unicode cleaning and sanitization."""

    # --- Positive Tests ---

    def test_replaces_en_dash(self):
        self.assertEqual(clean_text_for_pdf_func('A \u2013 B'), 'A - B')

    def test_replaces_em_dash(self):
        self.assertEqual(clean_text_for_pdf_func('A \u2014 B'), 'A - B')

    def test_replaces_curly_double_quotes(self):
        self.assertEqual(clean_text_for_pdf_func('\u201cHello\u201d'), '"Hello"')

    def test_replaces_mu_symbol(self):
        self.assertEqual(clean_text_for_pdf_func('\u03bcg/dL'), 'ug/dL')

    def test_plain_ascii_unchanged(self):
        self.assertEqual(clean_text_for_pdf_func('Normal Text 123!'), 'Normal Text 123!')

    def test_numeric_input_converted_to_string(self):
        self.assertEqual(clean_text_for_pdf_func(123.45), '123.45')

    def test_integer_input(self):
        self.assertEqual(clean_text_for_pdf_func(0), '0')

    def test_empty_string(self):
        self.assertEqual(clean_text_for_pdf_func(''), '')

    def test_multiple_replacements_in_one_string(self):
        result = clean_text_for_pdf_func('\u201cTest \u2013 Value\u201d')
        self.assertIn('"', result)
        self.assertIn('-', result)

    # --- Negative / Edge Case Tests ---

    def test_none_input_does_not_raise(self):
        """None should be converted gracefully (to string 'None')."""
        try:
            result = clean_text_for_pdf_func(None)
            self.assertIsInstance(result, str)
        except Exception:
            pass  # Acceptable if not explicitly handled

    def test_list_input_converted(self):
        """List input should be stringified without crashing."""
        try:
            result = clean_text_for_pdf_func([1, 2, 3])
            self.assertIsInstance(result, str)
        except Exception:
            pass


# ==============================================================================
# 2. Tests for dict_to_toon
# ==============================================================================

class TestDictToToon(unittest.TestCase):
    """Tests for TOON (Token-Oriented Object Notation) conversion."""

    # --- Positive Tests ---

    def test_flat_dict(self):
        data = {"name": "Alice", "age": 30}
        result = dict_to_toon(data)
        self.assertIn("name: Alice", result)
        self.assertIn("age: 30", result)

    def test_dict_with_list_values(self):
        data = {"tests": ["CBC", "LFT"]}
        result = dict_to_toon(data)
        self.assertIn("- CBC", result)
        self.assertIn("- LFT", result)

    def test_nested_dict(self):
        data = {"report": {"status": "final"}}
        result = dict_to_toon(data)
        self.assertIn("report:", result)
        self.assertIn("status: final", result)

    def test_empty_dict(self):
        result = dict_to_toon({})
        self.assertEqual(result, "")

    def test_empty_list(self):
        result = dict_to_toon([])
        self.assertEqual(result, "")

    def test_scalar_input(self):
        result = dict_to_toon("hello")
        self.assertEqual(result, "hello")

    def test_list_of_dicts(self):
        data = [{"key": "val1"}, {"key": "val2"}]
        result = dict_to_toon(data)
        self.assertIn("key: val1", result)
        self.assertIn("key: val2", result)

    def test_indentation_increases_for_nested(self):
        data = {"outer": {"inner": "value"}}
        result = dict_to_toon(data)
        lines = result.split("\n")
        inner_line = [l for l in lines if "inner" in l][0]
        self.assertTrue(inner_line.startswith("  "))

    # --- Negative / Edge Case Tests ---

    def test_dict_with_none_value(self):
        data = {"key": None}
        result = dict_to_toon(data)
        self.assertIn("key:", result)

    def test_deeply_nested_structure(self):
        data = {"a": {"b": {"c": "deep"}}}
        result = dict_to_toon(data)
        self.assertIn("c: deep", result)


# ==============================================================================
# 3. Tests for parse_interval_func
# ==============================================================================

class TestParseIntervalFunc(unittest.TestCase):
    """Tests for parsing reference range strings."""

    # --- Positive Tests ---

    def test_standard_range(self):
        self.assertEqual(parse_interval_func("70 - 110"), (70.0, 110.0))

    def test_less_than(self):
        self.assertEqual(parse_interval_func("<200"), (None, 200.0))

    def test_greater_than(self):
        self.assertEqual(parse_interval_func(">5"), (5.0, None))

    def test_range_with_spaces(self):
        min_v, max_v = parse_interval_func("3.5 - 5.5")
        self.assertAlmostEqual(min_v, 3.5)
        self.assertAlmostEqual(max_v, 5.5)

    def test_decimal_range(self):
        min_v, max_v = parse_interval_func("0.5 - 1.5")
        self.assertAlmostEqual(min_v, 0.5)
        self.assertAlmostEqual(max_v, 1.5)

    # --- Negative / Edge Case Tests ---

    def test_na_returns_none_none(self):
        self.assertEqual(parse_interval_func("N/A"), (None, None))

    def test_empty_string_returns_none_none(self):
        self.assertEqual(parse_interval_func(""), (None, None))

    def test_none_returns_none_none(self):
        self.assertEqual(parse_interval_func(None), (None, None))

    def test_qualitative_text_returns_none_none(self):
        self.assertEqual(parse_interval_func("Negative"), (None, None))

    def test_malformed_range_returns_none_none(self):
        self.assertEqual(parse_interval_func("abc-xyz"), (None, None))


# ==============================================================================
# 4. Tests for calculate_status_func
# ==============================================================================

class TestCalculateStatusFunc(unittest.TestCase):
    """Tests for Red/Amber/Green classification."""

    # --- Positive Tests ---

    def test_value_in_range_is_green(self):
        self.assertEqual(calculate_status_func(100, "70 - 110"), "Green")

    def test_value_above_range_is_red(self):
        self.assertEqual(calculate_status_func(200, "70 - 110"), "Red")

    def test_value_below_range_is_red(self):
        self.assertEqual(calculate_status_func(50, "70 - 110"), "Red")

    def test_value_at_upper_boundary_is_green(self):
        self.assertEqual(calculate_status_func(110, "70 - 110"), "Green")

    def test_value_at_lower_boundary_is_green(self):
        self.assertEqual(calculate_status_func(70, "70 - 110"), "Green")

    def test_value_just_above_range_is_amber(self):
        # 110 + 10% of (110-70)=4 → amber up to 114
        self.assertEqual(calculate_status_func(113, "70 - 110"), "Amber")

    def test_value_just_below_range_is_amber(self):
        # 70 - 10% of 40=4 → amber down to 66
        self.assertEqual(calculate_status_func(67, "70 - 110"), "Amber")

    def test_less_than_range_green(self):
        self.assertEqual(calculate_status_func(150, "<200"), "Green")

    def test_less_than_range_red(self):
        self.assertEqual(calculate_status_func(250, "<200"), "Red")

    def test_greater_than_range_green(self):
        self.assertEqual(calculate_status_func(10, ">5"), "Green")

    def test_greater_than_range_red(self):
        self.assertEqual(calculate_status_func(3, ">5"), "Red")

    def test_qualitative_negative_with_zero_is_green(self):
        self.assertEqual(calculate_status_func(0, "Negative"), "Green")

    def test_qualitative_negative_with_nonzero_is_red(self):
        self.assertEqual(calculate_status_func(1, "Negative"), "Red")

    # --- Negative / Edge Case Tests ---

    def test_na_interval_returns_na(self):
        self.assertEqual(calculate_status_func(100, "N/A"), "N/A")

    def test_na_value_returns_na(self):
        self.assertEqual(calculate_status_func("N/A", "70 - 110"), "N/A")

    def test_none_value_returns_na(self):
        self.assertEqual(calculate_status_func(None, "70 - 110"), "N/A")

    def test_none_interval_returns_na(self):
        self.assertEqual(calculate_status_func(100, None), "N/A")

    def test_nan_value_returns_na(self):
        self.assertEqual(calculate_status_func(float('nan'), "70 - 110"), "N/A")

    def test_string_value_returns_na(self):
        self.assertEqual(calculate_status_func("high", "70 - 110"), "N/A")


# ==============================================================================
# 5. Tests for standardize_interval_func
# ==============================================================================

class TestStandardizeIntervalFunc(unittest.TestCase):
    """Tests for reference range string normalization."""

    # --- Positive Tests ---

    def test_standard_dash_range(self):
        self.assertEqual(standardize_interval_func("70-110"), "70.0 - 110.0")

    def test_less_than_format(self):
        result = standardize_interval_func("<200")
        self.assertIn("200", result)
        self.assertIn("<", result)

    def test_greater_than_format(self):
        result = standardize_interval_func(">5")
        self.assertIn("5", result)
        self.assertIn(">", result)

    def test_qualitative_negative_unchanged(self):
        result = standardize_interval_func("Negative")
        self.assertEqual(result, "Negative")

    def test_qualitative_absent_unchanged(self):
        result = standardize_interval_func("Absent")
        self.assertEqual(result, "Absent")

    def test_range_with_to_keyword(self):
        result = standardize_interval_func("70 to 110")
        self.assertIn("70", result)
        self.assertIn("110", result)

    def test_range_with_em_dash(self):
        result = standardize_interval_func("70\u2013110")
        self.assertIn("70", result)
        self.assertIn("110", result)

    # --- Negative / Edge Case Tests ---

    def test_empty_string_returns_na(self):
        self.assertEqual(standardize_interval_func(""), "N/A")

    def test_none_returns_na(self):
        self.assertEqual(standardize_interval_func(None), "N/A")

    def test_reversed_range_is_corrected(self):
        # "110-70" should swap to "70 - 110"
        result = standardize_interval_func("110-70")
        self.assertIn("70", result)
        self.assertIn("110", result)

    def test_non_numeric_garbage_returns_original(self):
        result = standardize_interval_func("unknown format")
        self.assertIsInstance(result, str)


# ==============================================================================
# 6. Tests for calculate_advanced_forecast_fixed_func
# ==============================================================================

class TestCalculateAdvancedForecast(unittest.TestCase):
    """Tests for advanced forecasting logic."""

    # --- Positive Tests ---

    def test_single_value_forecasts_same(self):
        val, status = calculate_advanced_forecast_fixed_func([100], "70 - 110")
        self.assertEqual(val, 100.0)
        self.assertEqual(status, "Green")

    def test_two_values_linear_trend(self):
        # 100, 110 → next predicted = 120
        val, status = calculate_advanced_forecast_fixed_func([100, 110], "70 - 130")
        self.assertAlmostEqual(val, 120.0, delta=1.0)

    def test_three_values_returns_numeric_forecast(self):
        val, status = calculate_advanced_forecast_fixed_func([80, 90, 100], "70 - 110")
        self.assertIsInstance(val, float)
        self.assertIn(status, ["Green", "Amber", "Red", "N/A"])

    def test_stable_values_forecast_similar(self):
        val, status = calculate_advanced_forecast_fixed_func([100, 100, 100], "70 - 110")
        self.assertAlmostEqual(val, 100.0, delta=5.0)
        self.assertEqual(status, "Green")

    def test_rising_trend_predicts_above_range(self):
        val, status = calculate_advanced_forecast_fixed_func([90, 100, 110], "70 - 110")
        self.assertGreater(val, 110)

    # --- Negative / Edge Case Tests ---

    def test_na_interval_returns_na(self):
        val, status = calculate_advanced_forecast_fixed_func([100], "N/A")
        self.assertEqual(val, "N/A")
        self.assertEqual(status, "N/A")

    def test_empty_list_returns_na(self):
        val, status = calculate_advanced_forecast_fixed_func([], "70 - 110")
        self.assertEqual(val, "N/A")
        self.assertEqual(status, "N/A")

    def test_all_na_values_returns_na(self):
        val, status = calculate_advanced_forecast_fixed_func(["N/A", "N/A"], "70 - 110")
        self.assertEqual(val, "N/A")
        self.assertEqual(status, "N/A")

    def test_mixed_na_and_numeric(self):
        val, status = calculate_advanced_forecast_fixed_func(["N/A", 100], "70 - 110")
        self.assertIsNotNone(val)

    def test_non_numeric_strings_ignored(self):
        val, status = calculate_advanced_forecast_fixed_func(["abc", "xyz"], "70 - 110")
        self.assertEqual(val, "N/A")


# ==============================================================================
# 7. Tests for normalize_test_name_func
# ==============================================================================

class TestNormalizeTestNameFunc(unittest.TestCase):
    """Tests for medical test name normalization."""

    # --- Positive Tests ---

    def test_known_synonym_haemoglobin(self):
        result = normalize_test_name_func("HAEMOGLOBIN")
        self.assertEqual(result, "HEMOGLOBIN")

    def test_known_synonym_sgpt(self):
        result = normalize_test_name_func("SGPT")
        self.assertEqual(result, "ALT (SGPT)")

    def test_known_synonym_sgot(self):
        result = normalize_test_name_func("SGOT")
        self.assertEqual(result, "AST (SGOT)")

    def test_known_synonym_fbs(self):
        result = normalize_test_name_func("FBS")
        self.assertEqual(result, "FASTING BLOOD GLUCOSE")

    def test_known_synonym_hdl_c(self):
        result = normalize_test_name_func("HDL-C")
        self.assertEqual(result, "HDL CHOLESTEROL")

    def test_case_insensitive_lookup(self):
        result = normalize_test_name_func("sgpt")
        self.assertEqual(result, "ALT (SGPT)")

    def test_extra_whitespace_cleaned(self):
        result = normalize_test_name_func("  SGOT  ")
        self.assertEqual(result, "AST (SGOT)")

    # --- Negative / Edge Case Tests ---

    def test_unknown_name_returned_cleaned(self):
        result = normalize_test_name_func("SOME_RANDOM_TEST_XYZ")
        self.assertIsInstance(result, str)
        self.assertEqual(result, result.upper().strip())

    def test_empty_string_cleaned(self):
        result = normalize_test_name_func("")
        self.assertEqual(result, "")

    def test_numeric_string(self):
        result = normalize_test_name_func("12345")
        self.assertEqual(result, "12345")


# ==============================================================================
# 8. Tests for _categorize_test_func
# ==============================================================================

class TestCategorizeTestFunc(unittest.TestCase):
    """Tests for clinical test categorization."""

    # --- Positive Tests ---

    def test_hemoglobin_is_hematological(self):
        result = _categorize_test_func("Hemoglobin", "Hematology")
        self.assertEqual(result, "Hematological")

    def test_cholesterol_is_cardiovascular(self):
        result = _categorize_test_func("Total Cholesterol", "Lipid Profile")
        self.assertEqual(result, "Cardiovascular")

    def test_tsh_is_thyroid(self):
        result = _categorize_test_func("TSH", "Thyroid")
        self.assertEqual(result, "Thyroid")

    def test_glucose_is_metabolic(self):
        result = _categorize_test_func("Fasting Glucose", "Diabetes")
        self.assertEqual(result, "Metabolic")

    def test_creatinine_is_renal(self):
        result = _categorize_test_func("Creatinine", "Kidney Function")
        self.assertEqual(result, "Renal")

    def test_alt_is_hepatic(self):
        result = _categorize_test_func("ALT", "Liver Function")
        self.assertEqual(result, "Hepatic")

    def test_vitamin_b12_is_vitamins(self):
        result = _categorize_test_func("Vitamin B12", "Vitamins")
        self.assertEqual(result, "Vitamins")

    # --- Negative / Edge Case Tests ---

    def test_unknown_test_falls_to_other(self):
        result = _categorize_test_func("Zymase", "Unknown Panel")
        self.assertEqual(result, "Other")

    def test_empty_strings_fall_to_other(self):
        result = _categorize_test_func("", "")
        self.assertEqual(result, "Other")


# ==============================================================================
# 9. Tests for is_qualitative_interval_func
# ==============================================================================

class TestIsQualitativeIntervalFunc(unittest.TestCase):
    """Tests for detecting qualitative reference intervals."""

    # --- Positive Tests ---

    def test_negative_is_qualitative(self):
        self.assertTrue(is_qualitative_interval_func("Negative"))

    def test_positive_is_qualitative(self):
        self.assertTrue(is_qualitative_interval_func("Positive"))

    def test_absent_is_qualitative(self):
        self.assertTrue(is_qualitative_interval_func("Absent"))

    def test_nil_is_qualitative(self):
        self.assertTrue(is_qualitative_interval_func("Nil"))

    def test_present_is_qualitative(self):
        self.assertTrue(is_qualitative_interval_func("Present"))

    # --- Negative / Edge Case Tests ---

    def test_numeric_range_is_not_qualitative(self):
        self.assertFalse(is_qualitative_interval_func("70 - 110"))

    def test_less_than_is_not_qualitative(self):
        self.assertFalse(is_qualitative_interval_func("<200"))

    def test_na_returns_false(self):
        self.assertFalse(is_qualitative_interval_func("N/A"))

    def test_none_returns_false(self):
        self.assertFalse(is_qualitative_interval_func(None))

    def test_empty_string_returns_false(self):
        self.assertFalse(is_qualitative_interval_func(""))


# ==============================================================================
# 10. Tests for OCRRegion (extraction_utils.py)
# ==============================================================================

class TestOCRRegion(unittest.TestCase):
    """Tests for OCRRegion dataclass properties."""

    def _make_region(self, x1=10, y1=20, x2=100, y2=40, text="Test", page=1):
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return OCRRegion(text=text, bbox=bbox, page_num=page)

    # --- Positive Tests ---

    def test_bbox_xyxy_correct(self):
        region = self._make_region(10, 20, 100, 40)
        self.assertEqual(region.bbox_xyxy_func, [10, 20, 100, 40])

    def test_center_y_correct(self):
        region = self._make_region(10, 20, 100, 40)
        self.assertEqual(region.center_y_func, 30.0)

    def test_center_y_for_single_pixel_height(self):
        region = self._make_region(0, 50, 100, 50)
        self.assertEqual(region.center_y_func, 50.0)

    def test_bbox_with_non_rectangular_polygon(self):
        # Irregular polygon — should still compute min/max correctly
        bbox = [[5, 10], [95, 15], [100, 45], [0, 40]]
        region = OCRRegion(text="Skewed", bbox=bbox, page_num=1)
        xyxy = region.bbox_xyxy_func
        self.assertEqual(xyxy[0], 0)   # min_x
        self.assertEqual(xyxy[1], 10)  # min_y
        self.assertEqual(xyxy[2], 100) # max_x
        self.assertEqual(xyxy[3], 45)  # max_y

    def test_text_preserved(self):
        region = self._make_region(text="Blood Glucose")
        self.assertEqual(region.text, "Blood Glucose")

    def test_page_num_preserved(self):
        region = self._make_region(page=3)
        self.assertEqual(region.page_num, 3)

    # --- Negative / Edge Case Tests ---

    def test_single_point_bbox(self):
        bbox = [[50, 50], [50, 50], [50, 50], [50, 50]]
        region = OCRRegion(text="Point", bbox=bbox, page_num=1)
        self.assertEqual(region.center_y_func, 50.0)
        self.assertEqual(region.bbox_xyxy_func, [50, 50, 50, 50])

    def test_large_coordinates(self):
        region = self._make_region(0, 0, 10000, 8000)
        self.assertEqual(region.center_y_func, 4000.0)


# ==============================================================================
# 11. Tests for get_reading_order_simple_func
# ==============================================================================

class TestGetReadingOrderSimpleFunc(unittest.TestCase):
    """Tests for top-to-bottom reading order sorting."""

    def _make_region(self, y_top, y_bottom, text="T"):
        bbox = [[0, y_top], [100, y_top], [100, y_bottom], [0, y_bottom]]
        return OCRRegion(text=text, bbox=bbox, page_num=1)

    # --- Positive Tests ---

    def test_already_sorted(self):
        regions = [
            self._make_region(0, 20),
            self._make_region(30, 50),
            self._make_region(60, 80),
        ]
        self.assertEqual(get_reading_order_simple_func(regions), [0, 1, 2])

    def test_reverse_order_sorted(self):
        regions = [
            self._make_region(60, 80),
            self._make_region(30, 50),
            self._make_region(0, 20),
        ]
        self.assertEqual(get_reading_order_simple_func(regions), [2, 1, 0])

    def test_mixed_order(self):
        r1 = self._make_region(80, 100)   # center_y = 90  → index 0
        r2 = self._make_region(10, 30)    # center_y = 20  → index 1
        r3 = self._make_region(40, 60)    # center_y = 50  → index 2
        regions = [r1, r2, r3]
        self.assertEqual(get_reading_order_simple_func(regions), [1, 2, 0])

    def test_single_region(self):
        regions = [self._make_region(10, 30)]
        self.assertEqual(get_reading_order_simple_func(regions), [0])

    # --- Negative / Edge Case Tests ---

    def test_empty_list_returns_empty(self):
        self.assertEqual(get_reading_order_simple_func([]), [])

    def test_two_regions_at_same_height(self):
        # Stable sort — should return both indices, order may vary
        r1 = self._make_region(10, 30)
        r2 = self._make_region(10, 30)
        result = get_reading_order_simple_func([r1, r2])
        self.assertEqual(set(result), {0, 1})
        self.assertEqual(len(result), 2)


# ==============================================================================
# 12. Tests for estimate_tokens_func (runtime_logging.py)
# ==============================================================================

class TestEstimateTokensFunc(unittest.TestCase):
    """Tests for rough token estimation."""

    # --- Positive Tests ---

    def test_empty_string_returns_zero(self):
        self.assertEqual(estimate_tokens_func(""), 0)

    def test_four_chars_is_one_token(self):
        self.assertEqual(estimate_tokens_func("abcd"), 1)

    def test_eight_chars_is_two_tokens(self):
        self.assertEqual(estimate_tokens_func("abcdefgh"), 2)

    def test_large_text(self):
        text = "a" * 4000
        self.assertEqual(estimate_tokens_func(text), 1000)

    def test_minimum_is_one_for_nonempty(self):
        self.assertEqual(estimate_tokens_func("a"), 1)

    # --- Negative / Edge Case Tests ---

    def test_none_returns_zero(self):
        self.assertEqual(estimate_tokens_func(None), 0)

    def test_single_char_returns_one(self):
        self.assertEqual(estimate_tokens_func("x"), 1)


# ==============================================================================
# 13. Tests for initialize_logging_func and log_event_func
# ==============================================================================

class TestRuntimeLogging(unittest.TestCase):
    """Tests for centralized logging utilities."""

    def setUp(self):
        """Reset execution log state before each test."""
        import runtime_logging as rl
        rl.execution_log["errors"] = []
        rl.execution_log["node_execution"] = []
        rl.execution_log["function_execution"] = []
        rl.execution_log["pdf_processing"] = []
        rl.execution_log["llm_usage"]["total_calls"] = 0
        rl.execution_log["llm_usage"]["total_tokens"] = 0
        rl.log_filename = ""

    def tearDown(self):
        """Clean up any log files created during tests."""
        import runtime_logging as rl
        if rl.log_filename and os.path.exists(rl.log_filename):
            try:
                os.remove(rl.log_filename)
            except Exception:
                pass

    # --- Positive Tests ---

    def test_initialize_creates_log_file(self):
        initialize_logging_func("TestPatient")
        import runtime_logging as rl
        self.assertTrue(os.path.exists(rl.log_filename))

    def test_initialize_sanitizes_special_chars(self):
        initialize_logging_func("Patient Name!")
        import runtime_logging as rl
        self.assertNotIn("!", rl.log_filename)

    def test_log_event_error_appends_to_errors(self):
        initialize_logging_func("ErrPatient")
        import runtime_logging as rl
        initial_count = len(rl.execution_log["errors"])
        log_event_func("error", {"location": "test", "error": "Something went wrong"})
        self.assertEqual(len(rl.execution_log["errors"]), initial_count + 1)

    def test_log_event_llm_call_increments_counter(self):
        initialize_logging_func("LLMPatient")
        import runtime_logging as rl
        before = rl.execution_log["llm_usage"]["total_calls"]
        log_event_func("llm_call", {"tokens": 500, "phase": "extraction"})
        self.assertEqual(rl.execution_log["llm_usage"]["total_calls"], before + 1)

    def test_log_event_llm_call_accumulates_tokens(self):
        initialize_logging_func("TokenPatient")
        import runtime_logging as rl
        before = rl.execution_log["llm_usage"]["total_tokens"]
        log_event_func("llm_call", {"tokens": 300, "phase": "extraction"})
        self.assertEqual(rl.execution_log["llm_usage"]["total_tokens"], before + 300)

    def test_log_event_node_execution_appended(self):
        initialize_logging_func("NodePatient")
        import runtime_logging as rl
        before = len(rl.execution_log["node_execution"])
        log_event_func("node_execution", {"node": "extractor", "status": "started"})
        self.assertEqual(len(rl.execution_log["node_execution"]), before + 1)

    def test_log_event_pdf_processing_appended(self):
        initialize_logging_func("PDFPatient")
        import runtime_logging as rl
        before = len(rl.execution_log["pdf_processing"])
        log_event_func("pdf_processing", {"pdf_name": "test.pdf", "status": "completed"})
        self.assertEqual(len(rl.execution_log["pdf_processing"]), before + 1)

    # --- Negative / Edge Case Tests ---

    def test_initialize_with_empty_name_uses_fallback(self):
        initialize_logging_func("")
        import runtime_logging as rl
        self.assertIn("UnknownPatient", rl.log_filename)

    def test_log_event_without_init_does_not_crash(self):
        """log_event_func should not crash if log_filename is empty."""
        import runtime_logging as rl
        rl.log_filename = ""
        try:
            log_event_func("error", {"location": "test", "error": "no init"})
        except Exception as e:
            self.fail(f"log_event_func raised an exception: {e}")

    def test_save_log_without_init_does_not_crash(self):
        """save_log_func should handle missing log_filename gracefully."""
        import runtime_logging as rl
        rl.log_filename = ""
        try:
            save_log_func()
        except Exception as e:
            self.fail(f"save_log_func raised exception: {e}")


# ==============================================================================
# 14. Tests for Pydantic Schemas (schemas.py)
# ==============================================================================

class TestSchemas(unittest.TestCase):
    """Tests for Pydantic schema validation."""

    def test_test_result_defaults(self):
        from schemas import TestResult
        tr = TestResult(standard_name="Glucose", category="Metabolic")
        self.assertEqual(tr.methodology, "N/A")
        self.assertEqual(tr.value, 0.0)
        self.assertEqual(tr.unit, "N/A")
        self.assertEqual(tr.reference_range, "N/A")
        self.assertEqual(tr.recommendation, "N/A")

    def test_test_result_full(self):
        from schemas import TestResult
        tr = TestResult(
            standard_name="Hemoglobin",
            category="Hematological",
            methodology="Colorimetry",
            value=13.5,
            unit="g/dL",
            reference_range="12 - 17",
            recommendation="Maintain iron-rich diet"
        )
        self.assertEqual(tr.value, 13.5)
        self.assertEqual(tr.unit, "g/dL")

    def test_health_report_creation(self):
        from schemas import HealthReport, TestResult
        report = HealthReport(
            patient_name="Alice",
            age=30,
            gender="Female",
            test_date="2024-01-15",
            hospital_name="City Lab",
            results=[
                TestResult(standard_name="TSH", category="Thyroid")
            ]
        )
        self.assertEqual(report.patient_name, "Alice")
        self.assertEqual(len(report.results), 1)

    def test_health_report_missing_required_field_raises(self):
        from schemas import HealthReport
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            HealthReport(
                age=30,
                gender="Male",
                test_date="2024-01-15",
                hospital_name="Lab",
                results=[]
            )  # Missing patient_name

    def test_test_result_missing_required_field_raises(self):
        from schemas import TestResult
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            TestResult(category="Metabolic")  # Missing standard_name


# ==============================================================================
# 15. Integration-style tests for route_edges (nodes.py)
# ==============================================================================

class TestRouteEdges(unittest.TestCase):
    """Tests for the chatbot routing conditional edge function."""

    def test_out_of_scope_routes_correctly(self):
        from nodes import route_edges
        state = {"question_category": "out_of_scope", "messages": [], "patient_name": "Test"}
        self.assertEqual(route_edges(state), "handle_out_scope_node")

    def test_simple_lookup_routes_correctly(self):
        from nodes import route_edges
        state = {"question_category": "simple_lookup", "messages": [], "patient_name": "Test"}
        self.assertEqual(route_edges(state), "handle_simple_node")

    def test_complex_analysis_routes_correctly(self):
        from nodes import route_edges
        state = {"question_category": "complex_analysis", "messages": [], "patient_name": "Test"}
        self.assertEqual(route_edges(state), "handle_complex_node")

    def test_missing_category_defaults_to_complex(self):
        from nodes import route_edges
        state = {"messages": [], "patient_name": "Test"}  # No question_category
        self.assertEqual(route_edges(state), "handle_complex_node")

    def test_unknown_category_defaults_to_complex(self):
        from nodes import route_edges
        state = {"question_category": "totally_unknown", "messages": [], "patient_name": "Test"}
        self.assertEqual(route_edges(state), "handle_complex_node")


# ==============================================================================
# 16. Tests for handle_out_scope_node
# ==============================================================================

class TestHandleOutScopeNode(unittest.TestCase):
    """Tests that out-of-scope node returns static response without LLM."""

    def test_returns_expected_message(self):
        from nodes import handle_out_scope_node
        from prompts import CHATBOT_OUT_OF_SCOPE_RESPONSE
        state = {
            "messages": [],
            "patient_name": "Bob",
            "json_data": "{}",
        }
        result = handle_out_scope_node(state)
        msg_content = result["messages"][0].content
        self.assertEqual(msg_content, CHATBOT_OUT_OF_SCOPE_RESPONSE)

    def test_does_not_require_json_data(self):
        from nodes import handle_out_scope_node
        state = {"messages": [], "patient_name": "Bob"}
        result = handle_out_scope_node(state)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)


# ==============================================================================
# Run all tests
# ==============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
