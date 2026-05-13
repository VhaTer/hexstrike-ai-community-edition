import logging
import pytest
from server_core.setup_logging import setup_logging


class TestSuppressWerkzeugBanner:
    @pytest.fixture
    def banner_filter(self):
        root = setup_logging()
        werkzeug_logger = logging.getLogger('werkzeug')
        filters = [f for f in werkzeug_logger.filters
                   if type(f).__name__ == '_SuppressWerkzeugBanner']
        return filters[-1]

    def test_suppresses_dev_server_warning(self, banner_filter):
        record = logging.LogRecord("werkzeug", logging.WARNING, "", 0,
                                   "WARNING: This is a development server", (), None)
        assert banner_filter.filter(record) is False

    def test_passes_other_messages(self, banner_filter):
        record = logging.LogRecord("werkzeug", logging.INFO, "", 0,
                                   "GET / HTTP/1.1 200", (), None)
        assert banner_filter.filter(record) is True
