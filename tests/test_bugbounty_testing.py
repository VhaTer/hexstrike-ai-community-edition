from server_core.workflows.bugbounty.testing import FileUploadTestingFramework


class TestFileUploadTestingFramework:
    def test_add_test_case(self):
        fw = FileUploadTestingFramework()
        fw.add_test_case({"file": "test.txt", "payload": "<?php ..."})
        assert len(fw.upload_tests) == 1

    def test_run_tests(self):
        fw = FileUploadTestingFramework()
        fw.add_test_case({"file": "test.txt"})
        result = fw.run_tests("http://target.com/upload")
        assert result["success"] is True
        assert result["tests_run"] == 1

    def test_get_results(self):
        fw = FileUploadTestingFramework()
        fw.results.append({"status": "ok"})
        assert fw.get_results() == [{"status": "ok"}]
