"""No model needed — pure string parsing tests for helpers/tool_parser.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.tool_parser import parse_tool_call, strip_tool_call, validate_tool_call

SCHEMAS = {
    "get_weather": {"parameters": {"required": ["city"]}},
    "calculator": {"parameters": {"required": ["expression"]}},
}


def test_parse_well_formed_call():
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "กรุงเทพ\n</parameter>\n</function>\n</tool_call>"
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.args == {"city": "กรุงเทพ"}


def test_parse_multi_param_call():
    text = (
        "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
        "<parameter=cwd>\n/tmp\n</parameter>\n</function>\n</tool_call>"
    )
    call = parse_tool_call(text)
    assert call.name == "bash"
    assert call.args["command"] == "ls -la"
    assert call.args["cwd"] == "/tmp"


def test_parse_plain_text_no_call():
    assert parse_tool_call("สวัสดีครับ วันนี้อากาศดี") is None


def test_parse_malformed_missing_closing_tag():
    text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBangkok"
    assert parse_tool_call(text) is None


def test_strip_tool_call_keeps_reasoning():
    text = "ให้ผมเช็คให้นะครับ\n<tool_call>\n<function=x>\n</function>\n</tool_call>"
    assert strip_tool_call(text) == "ให้ผมเช็คให้นะครับ"


def test_validate_unknown_tool():
    call = parse_tool_call(
        "<tool_call>\n<function=nonexistent_tool>\n</function>\n</tool_call>"
    )
    err = validate_tool_call(call, SCHEMAS)
    assert err is not None
    assert "nonexistent_tool" in err


def test_validate_missing_required_param():
    call = parse_tool_call(
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
    )
    err = validate_tool_call(call, SCHEMAS)
    assert err is not None
    assert "city" in err


def test_validate_ok_call():
    call = parse_tool_call(
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n1+1\n</parameter>\n</function>\n</tool_call>"
    )
    assert validate_tool_call(call, SCHEMAS) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
