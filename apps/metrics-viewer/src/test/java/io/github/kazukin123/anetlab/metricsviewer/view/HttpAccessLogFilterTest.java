package io.github.kazukin123.anetlab.metricsviewer.view;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

@ExtendWith(OutputCaptureExtension.class)
class HttpAccessLogFilterTest {

	@Test
	void logsRequestStartAndCompletionAtInfo(CapturedOutput output) throws Exception {
		final MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/runs.json");
		request.setRemoteAddr("127.0.0.1");
		final MockHttpServletResponse response = new MockHttpServletResponse();

		new HttpAccessLogFilter().doFilter(request, response, new MockFilterChain());

		assertTrue(output.getOut().contains(
				"HTTP request started: method=GET path=/api/runs.json remote=127.0.0.1"));
		assertTrue(output.getOut().contains(
				"HTTP request completed: method=GET path=/api/runs.json status=200"));
		assertTrue(output.getOut().contains("outcome=completed"));
	}
}
