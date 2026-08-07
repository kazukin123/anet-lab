package io.github.kazukin123.anetlab.metricsviewer.view;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class HttpAccessLogFilter extends OncePerRequestFilter {

	private static final Logger log = LoggerFactory.getLogger(HttpAccessLogFilter.class);

	@Override
	protected void doFilterInternal(
			HttpServletRequest request,
			HttpServletResponse response,
			FilterChain filterChain) throws ServletException, IOException {
		final long startedAt = System.nanoTime();
		final String method = request.getMethod();
		final String path = request.getRequestURI();
		log.info("HTTP request started: method={} path={} remote={}",
				method, path, request.getRemoteAddr());

		boolean completed = false;
		try {
			filterChain.doFilter(request, response);
			completed = true;
		} finally {
			log.info(
					"HTTP request completed: method={} path={} status={} durationMs={} outcome={}",
					method,
					path,
					response.getStatus(),
					(System.nanoTime() - startedAt) / 1_000_000L,
					completed ? "completed" : "failed");
		}
	}
}
