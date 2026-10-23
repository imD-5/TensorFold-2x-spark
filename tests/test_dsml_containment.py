"""DeepSeek-V4.1 DSML tool calls are contained: a call counts only when its DSML calls block is complete and well formed
at the top level of the reply. Text in an open block is never a call, and a call inside an argument value is argument
text. This holds for the whole reply, for the stream, and with TF_DROP_UNOFFERED_TOOL_CALLS=1.

The tests use unittest, so ``python -I -m unittest tests.test_dsml_containment`` runs them without PyTorch or pytest."""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tensorfold.cuda.http import unsent_call_deltas
from tensorfold.cuda.reply_text import answer_without_tools, hide_tool_calls, parse_tool_calls
from tensorfold.dsml_calls import (dsml_block_end, parse_calls_block, reset_suppressed, suppressed_counts,
                                   top_level_blocks)
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server import metrics

D = "｜DSML｜"
OPEN, CLOSE = f"<{D} calls>", f"</{D} calls>"
FLAG = "TF_DROP_UNOFFERED_TOOL_CALLS"


def tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


TOOLS = [tool("read"), tool("write")]


def param(key: str, value: str, *, string: bool = True) -> str:
    return f'<{D} parameter name="{key}" string="{"true" if string else "false"}">{value}</{D} parameter>\n'


def invoke(name: str, *params: str) -> str:
    return f'<{D} invoke name="{name}">\n{"".join(params)}</{D} invoke>\n'


def calls(*invokes: str) -> str:
    return f"{OPEN}\n{''.join(invokes)}{CLOSE}"


# the shapes of a nested call that a model writes as code or as an example
NESTED_JSON = '<tool_call>{"name": "write", "arguments": {"path": "x"}}</tool_call>'
NESTED_XML = "<tool_call><function=write>\n<parameter=path>x</parameter>\n</function></tool_call>"
NESTED_GLM = "<tool_call>write<arg_key>path</arg_key><arg_value>x</arg_value></tool_call>"
NESTED_DSML = calls(invoke("write", param("path", "x")))
NESTED = [NESTED_JSON, NESTED_XML, NESTED_GLM, NESTED_DSML]

VALID = "Checking.\n\n" + calls(invoke("read", param("path", "a.py"), param("limit", "3", string=False)))


def names(found) -> list[str]:
    return [c["function"]["name"] for c in found or []]


def arguments(found) -> list[dict]:
    return [json.loads(c["function"]["arguments"]) for c in found or []]


class Flag(unittest.TestCase):
    """Runs a test with the unoffered-call flag off. A test turns it on with ``drop()``."""

    def setUp(self):
        patch = mock.patch.dict(os.environ)
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop(FLAG, None)
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))   # the warnings of the parser stay out of the run
        self.addCleanup(quiet.close)

    def drop(self):
        os.environ[FLAG] = "1"


class ValidCalls(Flag):
    def test_one_valid_call_keeps_its_exact_output(self):
        content, found = parse_tool_calls(VALID, TOOLS)
        self.assertEqual(content, "Checking.")
        self.assertEqual(names(found), ["read"])
        self.assertEqual(found[0]["function"]["arguments"], '{"path":"a.py","limit":3}')
        self.assertEqual(found[0]["type"], "function")
        self.assertRegex(found[0]["id"], r"^call_[0-9a-f]{24}$")

    def test_a_non_ascii_value_and_a_json_value_keep_their_bytes(self):
        text = calls(invoke("write", param("path", "café/日本.txt"),
                            param("data", '{"a": [1, 2.50, null]}', string=False),
                            param("bad", "{not json", string=False)))
        _, found = parse_tool_calls(text, TOOLS)
        self.assertEqual(found[0]["function"]["arguments"],
                         '{"path":"café/日本.txt","data":{"a":[1,2.5,null]},"bad":"{not json"}')

    def test_several_calls_in_one_block_keep_their_order(self):
        text = calls(invoke("read", param("path", "a")), invoke("write", param("path", "b")), invoke("read"))
        content, found = parse_tool_calls(text, TOOLS)
        self.assertEqual((content, names(found)), ("", ["read", "write", "read"]))
        self.assertEqual(arguments(found), [{"path": "a"}, {"path": "b"}, {}])

    def test_several_blocks_and_prose_between_them(self):
        text = "One.\n" + calls(invoke("read", param("path", "a"))) + "\nTwo.\n" + calls(
            invoke("write", param("path", "b"))) + "\nDone."
        content, found = parse_tool_calls(text, TOOLS)
        self.assertEqual(content, "One.\n\nTwo.\n\nDone.")
        self.assertEqual(names(found), ["read", "write"])

    def test_a_block_next_to_a_tool_call_block_keeps_the_text_order(self):
        text = (calls(invoke("read", param("path", "a"))) + NESTED_JSON.replace("write", "read")
                + calls(invoke("write")))
        _, found = parse_tool_calls(text, TOOLS)
        self.assertEqual(names(found), ["read", "read", "write"])

    def test_a_parameter_without_a_string_attribute_is_a_string(self):
        text = f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path">a</{D} parameter>\n</{D} invoke>\n{CLOSE}'
        self.assertEqual(arguments(parse_tool_calls(text, TOOLS)[1]), [{"path": "a"}])

    def test_the_single_call_mode_keeps_the_first_call(self):
        text = calls(invoke("read"), invoke("write"))
        content, found = parse_tool_calls(text, TOOLS, max_calls=1)
        self.assertEqual((content, names(found)), ("", ["read"]))


class OpenBlocks(Flag):
    def test_a_reply_cut_at_every_byte_makes_a_call_only_when_the_block_is_complete(self):
        start = VALID.index(OPEN)
        end = VALID.index(CLOSE) + len(CLOSE)
        for cut in range(len(VALID) + 1):
            content, found = parse_tool_calls(VALID[:cut], TOOLS)
            if cut >= end:
                self.assertEqual(names(found), ["read"], cut)
                self.assertEqual(content, "Checking.", cut)
            else:
                self.assertIsNone(found, cut)
                # an open block is removed: no part of it stays in the content. A cut inside the opener tag is text.
                opened = cut >= start + len(OPEN)
                self.assertEqual(content, (VALID[:start] if opened else VALID[:cut]).strip(), cut)
                if opened:
                    self.assertNotIn(D, content, cut)

    def test_a_nested_call_in_a_cut_off_block_is_text(self):
        for nested in NESTED:
            for cut in (f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">' + nested,
                        f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">' + nested
                        + f"</{D} parameter>\n",
                        f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">x'
                        + f"</{D} parameter>\n</{D} invoke>\n" + nested):
                text = "Before.\n" + cut
                self.assertEqual(parse_tool_calls(text, TOOLS), ("Before.", None), nested)
                self.assertEqual(parse_tool_calls(text, TOOLS, max_calls=1), ("Before.", None), nested)

    def test_a_nested_complete_block_does_not_close_a_cut_off_outer_block(self):
        outer = f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">see '
        text = "Before.\n" + outer + NESTED_DSML + "\n"
        self.assertEqual(parse_tool_calls(text, TOOLS), ("Before.", None))
        text = outer + NESTED_DSML + "\n"
        self.assertEqual(parse_tool_calls(text, TOOLS), ("", None))
        self.assertEqual(dsml_block_end(text, 0), -1)

    def test_an_open_block_after_a_valid_one_is_removed_and_the_valid_call_stays(self):
        text = VALID + "\n" + f'{OPEN}\n<{D} invoke name="write">\n' + NESTED_JSON
        content, found = parse_tool_calls(text, TOOLS)
        self.assertEqual(names(found), ["read"])
        self.assertEqual(content, "Checking.")

    def test_a_tool_call_tag_that_never_closes_opens_a_block_that_holds_the_rest(self):
        text = "The <tool_call> tag starts a call.\n" + calls(invoke("read", param("path", "a")))
        self.assertEqual(parse_tool_calls(text, TOOLS), (text, None))
        self.assertEqual(hide_tool_calls(text, finished=True), "The ")

    def test_an_unfinished_xml_call_is_not_followed_by_a_dsml_call(self):
        text = "<tool_call><function=write></function>\n" + calls(invoke("read", param("path", "a")))
        self.assertEqual(parse_tool_calls(text, TOOLS), (text, None))


class NestedCalls(Flag):
    def test_a_nested_call_in_an_argument_is_argument_text(self):
        for nested in NESTED:
            text = "Go.\n" + calls(invoke("read", param("path", nested)))
            content, found = parse_tool_calls(text, TOOLS)
            self.assertEqual((content, names(found), arguments(found)), ("Go.", ["read"], [{"path": nested}]), nested)

    def test_a_nested_call_in_a_json_argument_is_argument_text(self):
        value = json.dumps({"q": NESTED_JSON})
        _, found = parse_tool_calls(calls(invoke("read", param("q", value, string=False))), TOOLS)
        self.assertEqual(arguments(found), [{"q": {"q": NESTED_JSON}}])

    def test_an_unfinished_nested_call_in_an_argument_is_argument_text(self):
        for nested in ("<tool_call><function=write>", '<tool_call>{"name": "write", "arguments": {}', "<tool_call>"):
            _, found = parse_tool_calls(calls(invoke("read", param("path", nested))), TOOLS)
            self.assertEqual(arguments(found), [{"path": nested}], nested)

    def test_a_call_next_to_the_block_still_runs(self):
        adjacent = NESTED_JSON.replace("write", "read")
        for text in (calls(invoke("read", param("path", NESTED_JSON))) + adjacent,
                     adjacent + calls(invoke("read", param("path", NESTED_JSON)))):
            content, found = parse_tool_calls(text, TOOLS)
            self.assertEqual((content, names(found)), ("", ["read", "read"]))
            self.assertIn({"path": NESTED_JSON}, arguments(found))
            _, one = parse_tool_calls(text, TOOLS, max_calls=1)
            self.assertEqual(len(one), 1)

    def test_a_nested_complete_block_in_a_value_is_argument_text(self):
        text = calls(invoke("read", param("path", "see " + NESTED_DSML + " and " + NESTED_DSML)))
        _, found = parse_tool_calls(text, TOOLS)
        self.assertEqual(arguments(found), [{"path": "see " + NESTED_DSML + " and " + NESTED_DSML}])

    def test_a_nested_call_in_a_tool_call_block_is_argument_text(self):
        nested = NESTED_JSON.replace("write", "read")
        json_call = '<tool_call>{"name": "read", "arguments": {"q": ' + json.dumps(nested) + "}}</tool_call>"
        xml_call = f"<tool_call><function=read>\n<parameter=q>{nested}</parameter>\n</function></tool_call>"
        mixed_case = '<TOOL_CALL>{"name": "read", "arguments": {"q": ' + json.dumps(nested) + "}}</Tool_Call>"
        for text in (json_call, xml_call, mixed_case):
            for quota in (None, 1):
                content, found = parse_tool_calls("Go.\n" + text + "\n" + json_call, TOOLS, max_calls=quota)
                self.assertEqual(names(found), ["read"] * (2 if quota is None else 1), text)
                self.assertEqual(arguments(found)[0], {"q": nested}, text)
                if quota is None:
                    self.assertEqual(content, "Go.", text)

    def test_a_nested_call_in_a_cut_off_tool_call_block_is_text(self):
        nested = NESTED_JSON.replace("write", "read")
        for text in ('<tool_call>{"name": "read", "arguments": {"q": ' + json.dumps(nested),
                     '<tool_call>{"name": "read", "arguments": {"q": ' + json.dumps(nested) + "}}",
                     "<tool_call><function=read>\n<parameter=q>" + nested + "</parameter>\n</function>"):
            self.assertEqual(parse_tool_calls("Go.\n" + text, TOOLS), ("Go.\n" + text, None), text)

    def test_a_dsml_marker_in_a_tool_call_block_is_argument_text(self):
        text = '<tool_call>{"name": "read", "arguments": {"path": "' + OPEN + '"}}</tool_call>'
        content, found = parse_tool_calls(text, TOOLS)
        self.assertEqual((content, arguments(found)), ("", [{"path": OPEN}]))


class IllFormedBlocks(Flag):
    def assert_text(self, text: str):
        self.assertEqual(parse_tool_calls("Go.\n" + text, TOOLS), ("Go.\n" + text, None), text)

    def test_a_block_that_is_not_well_formed_is_not_a_call(self):
        valid = invoke("read", param("path", "a"))
        self.assert_text(calls())                                                  # no invoke
        self.assert_text(f"{OPEN}\nnotes {valid}{CLOSE}")                         # text before an invoke
        self.assert_text(f"{OPEN}\n{valid}notes\n{CLOSE}")                        # text after an invoke
        self.assert_text(f"{OPEN}\n{valid}{NESTED_JSON}\n{CLOSE}")                # a call beside the invokes
        self.assert_text(calls(valid.replace(f"</{D} invoke>", "")))             # an invoke that never closes
        self.assert_text(calls(invoke("read", "x\n", param("path", "a"))))        # text beside a parameter
        self.assert_text(calls(invoke("read", param("path", "a").replace(f"</{D} parameter>", ""))))   # a parameter that never closes
        self.assert_text(calls(f'<{D} invoke name="">\n</{D} invoke>\n'))          # no name

    def test_one_bad_invoke_keeps_the_whole_block_from_running(self):
        self.assert_text(calls(invoke("read", param("path", "a")), "junk\n", invoke("write")))

    def test_a_block_that_is_not_well_formed_does_not_hide_a_later_valid_block(self):
        later = calls(invoke("read", param("path", "a")))
        content, found = parse_tool_calls(calls() + "\n" + later, TOOLS)
        self.assertEqual(names(found), ["read"])
        self.assertEqual(content, calls())

    def test_the_parser_of_a_block_returns_none_for_text_and_an_empty_block(self):
        self.assertIsNone(parse_calls_block(""))
        self.assertIsNone(parse_calls_block(" \n "))
        self.assertIsNone(parse_calls_block("x"))
        self.assertEqual(parse_calls_block(f'\n<{D} invoke name="f">\n</{D} invoke>\n'), [("f", {})])


class StreamedText(Flag):
    def test_the_visible_text_never_shows_a_call_and_only_grows(self):
        for text in (VALID + " After.", "Go.\n" + calls(invoke("read", param("path", NESTED_JSON))) + " After.",
                     "Go.\n" + OPEN + f'\n<{D} invoke name="read">\n<{D} parameter name="path">' + NESTED_JSON):
            before = ""
            for cut in range(len(text) + 1):
                shown = hide_tool_calls(text[:cut], finished=False)
                self.assertTrue(shown.startswith(before), (cut, shown, before))
                self.assertNotIn("tool_call", shown)
                self.assertNotIn(D, shown)
                self.assertNotIn("<function", shown)
                before = shown
            final = hide_tool_calls(text, finished=True)
            self.assertTrue(final.startswith(before))
            self.assertEqual(final, hide_tool_calls(text, finished=True, any_case=True))

    def test_the_visible_text_is_what_the_parser_leaves(self):
        for text in (VALID + " After.", calls(invoke("read")) + "\nA " + NESTED_JSON.replace("write", "read") + " B",
                     "Go.\n" + calls(invoke("read", param("path", NESTED_JSON))) + " After."):
            content, _ = parse_tool_calls(text, TOOLS)
            self.assertEqual(hide_tool_calls(text, finished=True).strip(), content)

    def test_a_block_that_is_not_well_formed_stays_in_the_streamed_text(self):
        for block in (calls("junk\n"), calls(invoke("read"), "junk\n"), calls()):
            text = "A" + block + "B"
            content, found = parse_tool_calls(text, TOOLS)
            self.assertEqual((content, found), (text, None))
            before = ""
            for cut in range(len(text) + 1):
                shown = hide_tool_calls(text[:cut], finished=False, keep_ill_formed=True)
                self.assertTrue(shown.startswith(before), (cut, shown, before))
                before = shown
            self.assertEqual(hide_tool_calls(text, finished=True, keep_ill_formed=True), content)
            self.assertEqual(hide_tool_calls(text, finished=True), "AB")      # without the flag it is hidden

    def test_an_open_block_hides_the_rest_of_the_reply(self):
        text = "Go.\n" + OPEN + "\nrest " + NESTED_JSON
        self.assertEqual(hide_tool_calls(text, finished=True), "Go.\n")
        self.assertEqual(hide_tool_calls(text, finished=False), "Go.\n")


class StreamedCalls(Flag):
    """The streamed call deltas agree with the whole-reply parse at every chunk split: a call is streamed only from a
    tag that the parser counts at the top level, and the streamed calls are the first calls of the parse."""

    STREAM_TOOLS = [tool("read"), tool("write")]
    XML_WRITE = "<tool_call><function=write>\n<parameter=path>y</parameter>\n</function></tool_call>"

    def streamed(self, text, cuts):
        streamer = ToolCallStreamer(self.STREAM_TOOLS)
        found: dict[int, list] = {}
        for cut in cuts:
            for delta in streamer.feed(text[:cut]):
                for call in delta["tool_calls"]:
                    entry = found.setdefault(call["index"], ["", ""])
                    entry[0] = call["function"].get("name") or entry[0]
                    entry[1] += call["function"]["arguments"]
            yield [name for name, _ in found.values()], found
        self.last = [(name, json.loads(args)) for name, args in found.values()]

    def check(self, text, expect_streamed):
        _, parsed = parse_tool_calls(text, self.STREAM_TOOLS)
        whole = list(zip(names(parsed), arguments(parsed)))
        runs = [[len(text)], *([cut, len(text)] for cut in range(len(text) + 1)), range(1, len(text) + 1)]
        for cuts in runs:
            for so_far, _ in self.streamed(text, cuts):
                self.assertEqual(so_far, names(parsed)[:len(so_far)], (text, list(cuts)[:2]))
            self.assertEqual(self.last, whole[:len(self.last)], (text, list(cuts)[:2]))
            self.assertEqual(len(self.last), expect_streamed, (text, list(cuts)[:2]))

    def test_a_top_level_xml_call_streams_and_equals_the_parse(self):
        self.check("Go.\n" + self.XML_WRITE, 1)

    def test_a_tag_inside_a_dsml_block_is_never_streamed(self):
        for nested in (NESTED_XML, NESTED_XML.replace("</tool_call>", "")):
            opened = f'{OPEN}\n<{D} invoke name="read">\n<{D} parameter name="path" string="true">{nested}'
            closed = opened + f"</{D} parameter>\n</{D} invoke>\n{CLOSE}"
            for text in (opened, closed, "Go.\n" + closed + " After."):
                self.check(text, 0)

    def test_a_tag_inside_another_tag_block_is_never_streamed(self):
        self.check("<tool_call>note <tool_call><function=write>\n<parameter=path>y</parameter>\n</function>"
                   "</tool_call>", 0)
        self.check("<tool_call><tool_call><function=write>\n<parameter=path>y</parameter>\n</function></tool_call>", 0)
        self.check("Use <tool_call> tags. " + self.XML_WRITE, 0)

    def test_a_call_with_a_nested_tag_in_its_argument_streams_as_one_call(self):
        nested = NESTED_JSON.replace("write", "read")
        call = f"<tool_call><function=write>\n<parameter=path>{nested}</parameter>\n</function></tool_call>"
        self.check(call, 1)
        self.check(call + "\n" + self.XML_WRITE, 2)

    def test_the_streamer_scans_again_only_when_a_block_tag_arrives(self):
        from tensorfold.engine import tool_draft

        scans = []

        def counted(text, **kwargs):
            scans.append(len(text))
            return top_level_blocks(text, **kwargs)

        with mock.patch.object(tool_draft, "top_level_blocks", counted):
            streamer = ToolCallStreamer(self.STREAM_TOOLS)
            text = ""
            for _ in range(300):                     # prose with no tag, then a DSML block that stays open
                text += "plain words and more words. "
                streamer.feed(text)
            self.assertEqual(len(scans), 1)
            for piece in (OPEN, "\n<", D, " invoke name=\"read\">\n") + ("x " * 20,) * 300:
                text += piece
                self.assertEqual(streamer.feed(text), [])
            self.assertLessEqual(len(scans), 6)
            text += NESTED_XML
            self.assertEqual(streamer.feed(text), [])
            self.assertLessEqual(len(scans), 7)

    def test_a_block_cut_before_its_closing_tag_streams_no_complete_call(self):
        text = "Go.\n" + self.XML_WRITE
        for cut in range(len(text) + 1):
            streamer = ToolCallStreamer(self.STREAM_TOOLS)
            arguments_so_far = "".join(call["function"]["arguments"] for delta in streamer.feed(text[:cut])
                                       for call in delta["tool_calls"])
            if cut < len(text):
                with self.assertRaises(ValueError, msg=cut):
                    json.loads(arguments_so_far)
                self.assertEqual(parse_tool_calls(text[:cut], self.STREAM_TOOLS), (text[:cut].strip(), None))
            else:
                self.assertEqual(json.loads(arguments_so_far), {"path": "y"})

    def test_the_closing_tag_in_other_letter_case_closes_a_streamed_call(self):
        text = self.XML_WRITE[:-len("</tool_call>")] + "</TOOL_CALL>"
        streamer = ToolCallStreamer(self.STREAM_TOOLS)
        sent = "".join(call["function"]["arguments"] for delta in streamer.feed(text) for call in delta["tool_calls"])
        self.assertEqual(json.loads(sent), {"path": "y"})
        self.assertEqual(arguments(parse_tool_calls(text, self.STREAM_TOOLS)[1]), [{"path": "y"}])

    def test_an_unfinished_xml_call_never_sends_complete_arguments_when_a_dsml_call_follows(self):
        text = "<tool_call><function=write></function>\n" + calls(invoke("read", param("path", "a")))
        for cut in range(len(text) + 1):
            streamer = ToolCallStreamer(self.STREAM_TOOLS)
            sent = "".join(call["function"]["arguments"] for delta in streamer.feed(text[:cut])
                           for call in delta["tool_calls"])
            with self.assertRaises(ValueError, msg=cut):
                json.loads(sent)
        self.assertEqual(parse_tool_calls(text, self.STREAM_TOOLS), (text, None))

    def test_a_lone_closing_tag_in_a_value_never_completes_the_streamed_arguments(self):
        for value in ("x</tool_call>y", "x</TOOL_CALL>y"):
            text = f"<tool_call><function=write>\n<parameter=path>{value}</parameter>\n</function></tool_call>"
            for cut in range(len(text) + 1):
                streamer = ToolCallStreamer(self.STREAM_TOOLS)
                sent = "".join(call["function"]["arguments"] for delta in streamer.feed(text[:cut])
                               for call in delta["tool_calls"])
                with self.assertRaises(ValueError, msg=(value, cut)):
                    json.loads(sent)
            self.assertEqual(parse_tool_calls(text, self.STREAM_TOOLS)[1], None)

    def test_a_call_that_never_closed_keeps_its_index_and_the_next_call_is_sent_after_it(self):
        lone = "<tool_call><function=write>\n<parameter=path>x</tool_call>y</parameter>\n</function></tool_call>\n"
        stuck = "<tool_call><function=write>\n</function>junk</tool_call>\n"
        for first in (lone, stuck):
            text = first + calls(invoke("read", param("path", "a")))
            _, found = parse_tool_calls(text, self.STREAM_TOOLS)
            self.assertEqual(names(found), ["read"])
            streamer = ToolCallStreamer(self.STREAM_TOOLS)
            started = [call for delta in streamer.feed(text) for call in delta["tool_calls"] if "id" in call]
            self.assertEqual([call["index"] for call in started], [0])
            self.assertEqual((streamer.completed, streamer.abandoned), (0, 1))
            result = {"calls": found, "calls_streamed": streamer.completed, "calls_offset": streamer.abandoned}
            sent = unsent_call_deltas(result)
            self.assertEqual([d["tool_calls"][0]["index"] for d in sent], [1])
            self.assertEqual(sent[0]["tool_calls"][0]["function"]["name"], "read")

    def test_a_closed_streamed_call_is_not_sent_again(self):
        text = self.XML_WRITE + "\n" + OPEN + "\n"
        streamer = ToolCallStreamer(self.STREAM_TOOLS)
        streamer.feed(text)
        self.assertEqual((streamer.completed, streamer.abandoned), (1, 0))
        _, found = parse_tool_calls(self.XML_WRITE, self.STREAM_TOOLS)
        result = {"calls": found, "calls_streamed": streamer.completed, "calls_offset": streamer.abandoned}
        self.assertEqual(unsent_call_deltas(result), [])
        self.assertEqual(unsent_call_deltas({"calls": None}), [])
        self.assertEqual([d["tool_calls"][0]["index"] for d in unsent_call_deltas({"calls": found * 2})], [0, 1])

    def test_the_inner_tags_of_a_streamed_call_match_in_any_case_as_the_parser_does(self):
        for text in ("<tool_call><function=write>\n<parameter=path>y</parameter>\n</FUNCTION></tool_call>",
                     "<tool_call><function=write>\n<PARAMETER=path>y</Parameter>\n</function></tool_call>",
                     "<tool_call><FUNCTION=write>\n<parameter=path>y</parameter>\n</function></Tool_Call>"):
            self.check(text, 1)
            streamer = ToolCallStreamer(self.STREAM_TOOLS)
            streamer.feed(text)
            self.assertEqual((streamer.completed, streamer.abandoned), (1, 0))

    def test_a_tag_in_other_letter_case_is_not_streamed(self):
        self.check("<TOOL_CALL>" + self.XML_WRITE[len("<tool_call>"):-len("</tool_call>")] + "</tool_call>", 0)

    def test_a_dsml_call_before_a_tag_keeps_the_streamed_calls_a_prefix_of_the_parse(self):
        self.check(calls(invoke("read", param("path", "a"))) + "\n" + self.XML_WRITE, 0)

    def test_a_call_before_a_dsml_block_streams_and_the_block_does_not(self):
        self.check(self.XML_WRITE + "\n" + calls(invoke("read", param("path", NESTED_XML))), 1)

    def test_an_open_dsml_block_after_a_call_streams_nothing_more(self):
        self.check(self.XML_WRITE + "\n" + OPEN + "\n" + NESTED_XML, 1)


class UnofferedCalls(Flag):
    def test_by_default_an_unoffered_call_stays_text(self):
        text = "Go.\n" + calls(invoke("apply_patch", param("p", "x")))
        content, found = parse_tool_calls(text, TOOLS)
        self.assertIsNone(found)
        self.assertIn("apply_patch", content)

    def test_the_flag_drops_an_unoffered_block_and_logs_it(self):
        self.drop()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = parse_tool_calls("Go.\n" + calls(invoke("apply_patch", param("p", "x"))), TOOLS)
        self.assertEqual(result, ("Go.", None))
        self.assertIn("apply_patch", out.getvalue())

    def test_the_flag_removes_an_open_block_and_does_not_drop_its_nested_call(self):
        self.drop()
        unoffered = NESTED_JSON.replace("write", "apply_patch")
        for text in ("Go.\n" + OPEN + f'\n<{D} invoke name="read">\n<{D} parameter name="p">' + unoffered,
                     "Go.\n" + OPEN + "\n" + unoffered + "\n"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = parse_tool_calls(text, TOOLS)
            self.assertEqual(result, ("Go.", None))
            self.assertNotIn("dropped", out.getvalue())          # only the warning for the open block

    def test_the_flag_keeps_an_unoffered_nested_call_as_argument_text(self):
        self.drop()
        unoffered = NESTED_JSON.replace("write", "apply_patch")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            content, found = parse_tool_calls(calls(invoke("read", param("p", unoffered))), TOOLS)
        self.assertEqual((content, arguments(found)), ("", [{"p": unoffered}]))
        self.assertEqual(out.getvalue(), "")

    def test_the_flag_keeps_the_offered_invoke_next_to_an_unoffered_one(self):
        self.drop()
        with contextlib.redirect_stdout(io.StringIO()):
            content, found = parse_tool_calls(calls(invoke("apply_patch"), invoke("read")), TOOLS)
        self.assertEqual((content, names(found)), ("", ["read"]))

    def test_the_flag_keeps_a_block_that_is_not_well_formed(self):
        self.drop()
        text = "Go.\n" + calls(invoke("apply_patch"), "junk\n")
        self.assertEqual(parse_tool_calls(text, TOOLS), (text, None))

    def test_the_flag_streams_no_markup_of_an_open_or_closed_block(self):
        self.drop()
        unoffered = NESTED_JSON.replace("write", "apply_patch")
        for text in ("Go.\n" + calls(invoke("read", param("p", unoffered))) + " Bye.",
                     "Go.\n" + OPEN + "\n" + unoffered):
            for cut in range(len(text) + 1):
                shown = hide_tool_calls(text[:cut], finished=False, any_case=True)
                self.assertNotIn("tool_call", shown)
                self.assertNotIn("DSML", shown)

    def test_without_tools_the_flag_keeps_a_block_that_is_not_well_formed(self):
        self.drop()
        text = "A" + calls("junk\n") + "B"
        self.assertEqual(answer_without_tools(text, finished=True), text)
        for cut in range(len(text) + 1):
            shown = answer_without_tools(text[:cut], finished=False)
            self.assertTrue(text.startswith(shown))

    def test_without_tools_the_flag_drops_each_block_whole(self):
        self.drop()
        text = "Go.\n" + calls(invoke("read", param("p", NESTED_JSON))) + " Bye."
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(answer_without_tools(text, finished=True), "Go.\n Bye.")
            for cut in range(len(text) + 1):
                shown = answer_without_tools(text[:cut], finished=False)
                self.assertNotIn("tool_call", shown)
                self.assertNotIn("DSML", shown)
            self.assertEqual(answer_without_tools(text[:text.index(CLOSE)], finished=True), "Go.\n")


class Suppressed(Flag):
    """A block that does not count and keeps a later call as text is logged once for the reply and counted."""

    def setUp(self):
        super().setUp()
        reset_suppressed()
        self.addCleanup(reset_suppressed)

    def parse(self, text):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = parse_tool_calls(text, TOOLS)
        return result, out.getvalue()

    def test_a_valid_call_and_a_cut_off_block_with_no_later_call_are_silent(self):
        for text in (VALID, "Go.\n" + OPEN + "\n", "Go. <tool_call> only", f'{OPEN}\n<{D} invoke name="read">\n'):
            _, log = self.parse(text)
            self.assertEqual(log, "", text)
        self.assertEqual(suppressed_counts(), {})
        self.assertNotIn("tool_calls_suppressed_total", metrics.render(SimpleNamespace()))

    def test_a_lone_tool_call_tag_before_a_dsml_call_is_logged_and_counted(self):
        text = "Use <tool_call> tags.\n" + calls(invoke("read", param("path", "SECRET")))
        (content, found), log = self.parse(text)
        self.assertEqual((content, found), (text, None))
        self.assertEqual(log.count("\n"), 1)
        self.assertIn("WARNING", log)
        self.assertIn("open_tool_call_tag at character 4", log)
        self.assertNotIn("SECRET", log)
        self.assertNotIn("Use", log)
        self.assertEqual(suppressed_counts(), {"open_tool_call_tag": 1})

    def test_a_cut_off_dsml_block_that_holds_a_complete_call_is_logged_and_counted(self):
        text = "Go.\n" + OPEN + f'\n<{D} invoke name="read">\n<{D} parameter name="path">' + NESTED_JSON
        _, log = self.parse(text)
        self.assertIn("open_dsml_block at character 4", log)
        self.assertEqual(suppressed_counts(), {"open_dsml_block": 1})

    def test_a_lone_dsml_opener_before_a_dsml_call_is_logged_and_counted(self):
        _, log = self.parse("The opener is " + OPEN + " here.\n" + calls(invoke("read")))
        self.assertIn("open_dsml_block at character 14", log)

    def test_a_block_that_is_not_well_formed_is_logged_and_counted(self):
        _, log = self.parse("Go.\n" + calls(invoke("read"), "junk\n"))
        self.assertIn("ill_formed_dsml_block at character 4", log)
        self.assertEqual(suppressed_counts(), {"ill_formed_dsml_block": 1})

    def test_one_line_and_one_count_for_each_reason_in_a_reply(self):
        text = calls("junk\n") + calls("more junk\n") + "<tool_call> " + calls(invoke("read"))
        _, log = self.parse(text)
        self.assertEqual(log.count("\n"), 1)
        self.assertIn("ill_formed_dsml_block at character 0", log)
        self.assertIn("(3 such blocks)", log)
        self.assertEqual(suppressed_counts(), {"ill_formed_dsml_block": 1, "open_tool_call_tag": 1})
        self.parse(calls("junk\n"))
        self.assertEqual(suppressed_counts()["ill_formed_dsml_block"], 2)

    def test_the_metrics_text_shows_the_counter_by_reason(self):
        self.parse("Use <tool_call> tags.\n" + calls(invoke("read")))
        self.parse("Go.\n" + calls("junk\n"))
        self.parse("Go.\n" + calls("junk\n"))
        body = metrics.render(SimpleNamespace())
        name = f"{metrics.PREFIX}tool_calls_suppressed_total"
        self.assertIn(f"# TYPE {name} counter", body)
        self.assertIn(f'{name}{{reason="ill_formed_dsml_block"}} 2', body)
        self.assertIn(f'{name}{{reason="open_tool_call_tag"}} 1', body)


class Blocks(unittest.TestCase):
    def test_the_blocks_at_the_top_level(self):
        text = "a " + calls(invoke("f", param("k", NESTED_JSON))) + " b " + NESTED_JSON + " c"
        blocks, open_at = top_level_blocks(text)
        self.assertEqual([b.kind for b in blocks], ["dsml", "tag"])
        self.assertEqual(open_at, -1)
        self.assertEqual(text[blocks[1].start:blocks[1].end], NESTED_JSON)

    def test_the_start_of_the_first_open_block(self):
        text = "a " + NESTED_JSON + " b " + OPEN + " c"
        blocks, open_at = top_level_blocks(text)
        self.assertEqual([b.kind for b in blocks], ["tag"])
        self.assertEqual(open_at, text.index(OPEN))
        self.assertEqual(top_level_blocks("x <tool_call> y")[1], 2)

    def test_the_tags_in_any_case_are_blocks_only_when_asked(self):
        text = "<TOOL_CALL>x</Tool_Call>"
        self.assertEqual(len(top_level_blocks(text)[0]), 1)
        self.assertEqual(top_level_blocks(text, any_case=False), ([], -1))

    def test_a_large_reply_is_parsed_in_linear_time(self):
        parts = ["<tool_call>" + "x" * 20 + "</tool_call> " + OPEN + " " + "<tool_call> " * 5 for _ in range(4000)]
        for text in ("".join(parts), OPEN * 30000, "<tool_call>" * 30000, calls(invoke("f", param("k", OPEN * 30000)))):
            start = time.perf_counter()
            top_level_blocks(text)
            parse_tool_calls(text, TOOLS)
            hide_tool_calls(text, finished=False)
            self.assertLess(time.perf_counter() - start, 20.0)


if __name__ == "__main__":
    unittest.main()
