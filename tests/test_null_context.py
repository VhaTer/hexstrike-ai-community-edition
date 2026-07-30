"""Tests for NullContext — mock MCP context for internal calls."""

import pytest

from pulse.tools.null_context import NullContext, _DummyContent, _DummyResource, _DummySample


class TestDummyContent:
    def test_default_text(self):
        d = _DummyContent()
        assert d.content == ""

    def test_custom_text(self):
        d = _DummyContent("hello")
        assert d.content == "hello"


class TestDummyResource:
    def test_empty_contents(self):
        r = _DummyResource()
        assert r.contents == []


class TestDummySample:
    def test_default_text(self):
        s = _DummySample()
        assert s.text == ""


class TestNullContext:
    @pytest.fixture
    def ctx(self):
        return NullContext()

    def test_session_id(self, ctx):
        assert ctx.session_id == ""

    def test_request_id(self, ctx):
        assert ctx.request_id is None

    async def test_info(self, ctx):
        await ctx.info("test")

    async def test_warning(self, ctx):
        await ctx.warning("test")

    async def test_error(self, ctx):
        await ctx.error("test")

    async def test_report_progress(self, ctx):
        await ctx.report_progress(1, 10)

    async def test_read_resource(self, ctx):
        r = await ctx.read_resource("test://uri")
        assert isinstance(r, _DummyResource)

    async def test_sample(self, ctx):
        s = await ctx.sample([])
        assert isinstance(s, _DummySample)

    async def test_set_state(self, ctx):
        await ctx.set_state("key", "value")

    async def test_get_state(self, ctx):
        val = await ctx.get_state("key")
        assert val == {}
